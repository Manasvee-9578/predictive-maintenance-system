"""Production-style LSTM training pipeline for RUL prediction."""

# import sys
# from pathlib import Path

# ROOT_DIR = Path(__file__).resolve().parents[2]
# sys.path.append(str(ROOT_DIR))
from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.settings import Settings
from src.rul_prediction.prepare_sequences import SequencePreparer, load_processed_csv
from src.rul_prediction.rul_utils import (
    add_prediction_columns,
    compute_metrics,
    plot_degradation_curve,
    plot_engine_degradation_trends,
    plot_error_distribution,
    plot_failure_risk,
    plot_predicted_vs_actual,
    plot_training_history,
    save_figure,
    save_metrics_csv,
    save_predictions_csv,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RULTrainer:
    """End-to-end LSTM trainer for NASA C-MAPSS FD001 RUL forecasting."""

    def __init__(
        self,
        sequence_length: int | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        lstm_units: int | None = None,
        dropout: float | None = None,
        random_state: int = 42,
    ) -> None:
        self.sequence_length = int(sequence_length or Settings.SEQUENCE_LENGTH)
        self.epochs = int(epochs or Settings.LSTM_EPOCHS)
        self.batch_size = int(batch_size or Settings.LSTM_BATCH_SIZE)
        self.learning_rate = float(learning_rate or Settings.LSTM_LEARNING_RATE)
        self.lstm_units = int(lstm_units or Settings.LSTM_UNITS)
        self.dropout = float(dropout if dropout is not None else Settings.LSTM_DROPOUT)
        self.random_state = random_state

        self.model = None
        self.history = None
        self.feature_columns: list[str] = []
        self.n_features = 0

        self.model_dir = Settings.MODEL_DIR
        self.predictions_dir = Settings.OUTPUT_DIR / "predictions"
        self.plots_dir = Settings.OUTPUT_DIR / "plots"

    def run(
        self,
        train_path: Path | None = None,
        test_path: Path | None = None,
        rul_path: Path | None = None,
    ) -> dict:
        """Train, evaluate, save artifacts, and generate dashboard outputs."""
        try:
            Settings.ensure_directories()
            self._set_random_seeds()
            logger.info("Starting RUL LSTM training pipeline")

            preparer = SequencePreparer(
                sequence_length=self.sequence_length,
                random_state=self.random_state,
            )
            X_train, X_val, y_train, y_val = preparer.prepare_train_sequences(
                train_path
            )
            self.feature_columns = preparer.feature_columns
            self.n_features = preparer.n_features

            self.model = self.build_model()
            callbacks = self._get_callbacks()

            self.history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1,
            )

            val_pred = self._predict_array(X_val)
            val_metrics = compute_metrics(y_val, val_pred, "RUL LSTM Validation")
            validation_df = self._build_prediction_frame(
                engine_ids=None,
                actual=y_val,
                predicted=val_pred,
                dataset="validation",
            )

            test_df = None
            test_metrics = None
            if self._test_files_available(test_path, rul_path):
                X_test, y_test, test_engine_ids = preparer.prepare_test_sequences(
                    test_path, rul_path
                )
                test_pred = self._predict_array(X_test)
                test_metrics = compute_metrics(y_test, test_pred, "RUL LSTM Test")
                test_df = self._build_prediction_frame(
                    engine_ids=test_engine_ids,
                    actual=y_test,
                    predicted=test_pred,
                    dataset="test",
                )

            trend_df = self._build_training_trends(preparer, train_path)

            self._save_model(preparer)
            self._save_outputs(
                validation_df, val_metrics, test_df, test_metrics, trend_df
            )
            self._save_plots(validation_df, test_df, trend_df)

            logger.success("RUL LSTM training pipeline completed")
            return {
                "model": self.model,
                "history": self.history.history,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
                "feature_columns": self.feature_columns,
                "trend_df": trend_df,
            }
        except Exception as exc:
            logger.exception("RUL LSTM training failed")
            raise exc

    def build_model(self):
        """Build a compact, maintainable Keras LSTM regressor."""
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import (
            BatchNormalization,
            Dense,
            Dropout,
            Input,
            LSTM,
        )

        if self.n_features <= 0:
            raise ValueError("n_features must be set before building the model.")

        model = Sequential(
            [
                Input(shape=(self.sequence_length, self.n_features)),
                LSTM(self.lstm_units, return_sequences=True),
                Dropout(self.dropout),
                BatchNormalization(),
                LSTM(max(self.lstm_units // 2, 8), return_sequences=False),
                Dropout(self.dropout),
                Dense(64, activation="relu"),
                Dropout(self.dropout),
                Dense(32, activation="relu"),
                Dense(1, activation="linear"),
            ],
            name="rul_lstm_regressor",
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        model.summary(print_fn=logger.info)
        return model

    def _get_callbacks(self) -> list:
        from tensorflow.keras.callbacks import (
            EarlyStopping,
            ModelCheckpoint,
            ReduceLROnPlateau,
        )

        self.model_dir.mkdir(parents=True, exist_ok=True)
        return [
            EarlyStopping(
                monitor="val_loss",
                patience=Settings.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(self.model_dir / "best_rul_lstm.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=Settings.REDUCE_LR_PATIENCE,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

    def _predict_array(self, X: np.ndarray) -> np.ndarray:
        prediction = self.model.predict(X, verbose=0).reshape(-1)
        return np.clip(prediction, 0, Settings.MAX_RUL)

    def _build_prediction_frame(
        self,
        engine_ids: np.ndarray | None,
        actual: np.ndarray,
        predicted: np.ndarray,
        dataset: str,
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "dataset": dataset,
                "actual_rul": np.asarray(actual, dtype=float),
                "predicted_rul": np.round(np.asarray(predicted, dtype=float), 3),
            }
        )
        if engine_ids is not None:
            df.insert(0, "engine_id", engine_ids)
        else:
            df.insert(0, "sequence_id", np.arange(1, len(df) + 1))
        return add_prediction_columns(df)

    def _build_training_trends(
        self,
        preparer: SequencePreparer,
        train_path: Path | None,
        max_engines: int = 5,
    ) -> pd.DataFrame:
        """Predict full trajectories for a small set of training engines."""
        path = train_path or Settings.OUTPUT_DIR / "processed" / "train_processed.csv"
        train_df = load_processed_csv(path)
        sample_engines = sorted(train_df["engine_id"].unique())[:max_engines]
        sample_df = train_df[train_df["engine_id"].isin(sample_engines)].copy()
        trajectories = preparer.prepare_full_trajectories(sample_df)

        rows: list[pd.DataFrame] = []
        for engine_id, (X_engine, y_engine, cycles) in trajectories.items():
            predicted = self._predict_array(X_engine)
            engine_rows = pd.DataFrame(
                {
                    "engine_id": engine_id,
                    "cycle": cycles,
                    "predicted_rul": np.round(predicted, 3),
                }
            )
            if y_engine is not None:
                engine_rows["actual_rul"] = y_engine

            source_cols = ["engine_id", "cycle"] + [
                col for col in self.feature_columns[:8] if col in sample_df.columns
            ]
            source = sample_df[source_cols].drop_duplicates(["engine_id", "cycle"])
            engine_rows = engine_rows.merge(
                source, on=["engine_id", "cycle"], how="left"
            )
            rows.append(add_prediction_columns(engine_rows, actual_col="actual_rul"))

        trend_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        logger.info(f"Generated trajectory predictions: {trend_df.shape}")
        return trend_df

    def _save_model(self, preparer: SequencePreparer) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / "rul_lstm_model.keras"
        self.model.save(model_path)

        metadata = {
            "model_type": "keras_lstm",
            "dataset": "NASA CMAPSS FD001",
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "feature_columns": self.feature_columns,
            "lstm_units": self.lstm_units,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs_requested": self.epochs,
            "epochs_trained": len(self.history.history.get("loss", [])),
            "random_state": self.random_state,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "sequence_preparer": preparer.summary(),
        }
        metadata_path = self.model_dir / "rul_lstm_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info(f"Saved RUL model to {model_path}")
        logger.info(f"Saved RUL metadata to {metadata_path}")

    def _save_outputs(
        self,
        validation_df: pd.DataFrame,
        val_metrics: dict,
        test_df: pd.DataFrame | None,
        test_metrics: dict | None,
        trend_df: pd.DataFrame,
    ) -> None:
        save_predictions_csv(
            validation_df, self.predictions_dir, "rul_validation_predictions.csv"
        )
        if test_df is not None:
            save_predictions_csv(
                test_df, self.predictions_dir, "rul_test_predictions.csv"
            )
            save_predictions_csv(test_df, self.predictions_dir, "rul_predictions.csv")
        if not trend_df.empty:
            save_predictions_csv(
                trend_df, self.predictions_dir, "rul_training_trends.csv"
            )

        metrics = [val_metrics]
        if test_metrics is not None:
            metrics.append(test_metrics)
        save_metrics_csv(metrics, self.predictions_dir, "rul_training_metrics.csv")

        history_df = pd.DataFrame(self.history.history)
        save_predictions_csv(
            history_df, self.predictions_dir, "rul_training_history.csv"
        )

    def _save_plots(
        self,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame | None,
        trend_df: pd.DataFrame,
    ) -> None:
        save_figure(
            plot_training_history(self.history.history),
            self.plots_dir,
            "rul_training_history.html",
        )
        save_figure(
            plot_predicted_vs_actual(
                validation_df["actual_rul"].to_numpy(),
                validation_df["predicted_rul"].to_numpy(),
                "Validation Predicted vs Actual RUL",
            ),
            self.plots_dir,
            "rul_predicted_vs_actual_validation.html",
        )
        save_figure(
            plot_error_distribution(
                validation_df["actual_rul"].to_numpy(),
                validation_df["predicted_rul"].to_numpy(),
                "Validation RUL Error Distribution",
            ),
            self.plots_dir,
            "rul_error_distribution_validation.html",
        )

        risk_source = test_df if test_df is not None else validation_df
        save_figure(
            plot_failure_risk(risk_source), self.plots_dir, "rul_failure_risk.html"
        )

        if test_df is not None:
            save_figure(
                plot_predicted_vs_actual(
                    test_df["actual_rul"].to_numpy(),
                    test_df["predicted_rul"].to_numpy(),
                    "Test Predicted vs Actual RUL",
                ),
                self.plots_dir,
                "rul_predicted_vs_actual_test.html",
            )

        if not trend_df.empty:
            save_figure(
                plot_engine_degradation_trends(trend_df),
                self.plots_dir,
                "rul_engine_degradation_trends.html",
            )

            sensor_cols = [
                col for col in self.feature_columns if col in trend_df.columns
            ][:3]
            for engine_id in sorted(trend_df["engine_id"].unique())[:3]:
                save_figure(
                    plot_degradation_curve(trend_df, int(engine_id), sensor_cols),
                    self.plots_dir,
                    f"rul_degradation_engine_{int(engine_id)}.html",
                )

    def _set_random_seeds(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        try:
            import tensorflow as tf

            tf.random.set_seed(self.random_state)
        except Exception:
            logger.warning("TensorFlow seed could not be set before import.")

    @staticmethod
    def _test_files_available(test_path: Path | None, rul_path: Path | None) -> bool:
        test_file = (
            test_path or Settings.OUTPUT_DIR / "processed" / "test_processed.csv"
        )
        rul_file = rul_path or Settings.OUTPUT_DIR / "processed" / "rul_processed.csv"
        return test_file.exists() and rul_file.exists()


if __name__ == "__main__":
    trainer = RULTrainer()
    trainer.run()
