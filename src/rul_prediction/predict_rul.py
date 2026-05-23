"""Inference pipeline for trained RUL LSTM models."""

from __future__ import annotations

import json
import os
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
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    URGENCY_IMMEDIATE,
    URGENCY_PRIORITY,
    add_prediction_columns,
    classify_risk,
    compute_metrics,
    estimate_confidence,
    plot_confidence_heatmap,
    plot_degradation_curve,
    plot_engine_degradation_trends,
    plot_error_distribution,
    plot_failure_risk,
    plot_predicted_vs_actual,
    plot_risk_distribution,
    plot_rul_trend,
    plot_urgency_summary,
    predict_maintenance_urgency,
    save_figure,
    save_metrics_csv,
    save_predictions_csv,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class RULPredictor:
    """Load a trained LSTM model and generate RUL forecasts."""

    def __init__(
        self,
        model_path: Path | None = None,
        metadata_path: Path | None = None,
    ) -> None:
        self.model_dir = Settings.MODEL_DIR
        self.predictions_dir = Settings.OUTPUT_DIR / "predictions"
        self.plots_dir = Settings.OUTPUT_DIR / "plots"

        self.model_path = model_path or self.model_dir / "rul_lstm_model.keras"
        self.metadata_path = metadata_path or self.model_dir / "rul_lstm_metadata.json"

        self.model = None
        self.sequence_length = Settings.SEQUENCE_LENGTH
        self.feature_columns: list[str] = []
        self.n_features = 0
        self.metadata: dict = {}

    def load_model(
        self,
        model_path: Path | None = None,
        meta_path: Path | None = None,
    ) -> "RULPredictor":
        """Load the Keras model and its feature metadata."""
        try:
            import tensorflow as tf

            self.model_path = model_path or self.model_path
            self.metadata_path = meta_path or self.metadata_path

            if not self.model_path.exists():
                raise FileNotFoundError(f"RUL model not found: {self.model_path}")

            logger.info(f"Loading RUL model from {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)

            if self.metadata_path.exists():
                self.metadata = json.loads(
                    self.metadata_path.read_text(encoding="utf-8")
                )
                self.sequence_length = int(
                    self.metadata.get("sequence_length", self.sequence_length)
                )
                self.feature_columns = list(self.metadata.get("feature_columns", []))
                self.n_features = int(
                    self.metadata.get("n_features", len(self.feature_columns))
                )
            else:
                logger.warning(f"Metadata file not found: {self.metadata_path}")

            if not self.feature_columns:
                raise ValueError("Model metadata does not contain feature_columns.")

            logger.info(
                f"Loaded RUL model with window={self.sequence_length}, "
                f"features={len(self.feature_columns)}"
            )
            return self
        except Exception as exc:
            logger.exception("Failed to load RUL model")
            raise exc

    def predict(
        self,
        test_path: Path | None = None,
        rul_path: Path | None = None,
        trend_path: Path | None = None,
        generate_trends: bool = True,
    ) -> dict:
        """Run batch inference and save dashboard-ready outputs."""
        try:
            if self.model is None:
                self.load_model()

            Settings.ensure_directories()
            preparer = SequencePreparer(
                sequence_length=self.sequence_length,
                feature_columns=self.feature_columns,
            )
            X_test, y_test, engine_ids = preparer.prepare_test_sequences(
                test_path, rul_path
            )
            predicted = self._predict_array(X_test)
            results_df = self._build_results(engine_ids, y_test, predicted)

            metrics = None
            if np.isfinite(y_test).any():
                metrics = compute_metrics(y_test, predicted, "RUL LSTM Inference")

            trend_df = None
            if generate_trends:
                trend_df = self._predict_trajectories(preparer, trend_path)

            self._save_outputs(results_df, metrics, trend_df)
            self._save_plots(results_df, metrics, trend_df)
            self._log_summary(results_df)

            logger.success("RUL inference pipeline completed")
            return {
                "results_df": results_df,
                "metrics": metrics,
                "trend_df": trend_df,
            }
        except Exception as exc:
            logger.exception("RUL inference failed")
            raise exc

    def predict_single(self, sensor_sequence: np.ndarray) -> dict:
        """Predict RUL for one already-processed sensor sequence."""
        if self.model is None:
            self.load_model()

        sequence = np.asarray(sensor_sequence, dtype=np.float32)
        if sequence.ndim == 2:
            sequence = sequence[np.newaxis, :, :]
        if sequence.ndim != 3:
            raise ValueError(
                "sensor_sequence must have shape (seq_len, features) or (1, seq_len, features)."
            )
        if sequence.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {sequence.shape[1]}."
            )
        if sequence.shape[2] != len(self.feature_columns):
            raise ValueError(
                f"Expected {len(self.feature_columns)} features, got {sequence.shape[2]}."
            )

        predicted = float(self._predict_array(sequence)[0])
        confidence = float(estimate_confidence(np.asarray([predicted]))[0])
        risk = str(classify_risk(np.asarray([predicted]))[0])
        urgency = str(
            predict_maintenance_urgency(
                np.asarray([predicted]), np.asarray([confidence])
            )[0]
        )
        return {
            "predicted_rul": round(predicted, 3),
            "confidence": round(confidence, 4),
            "risk_category": risk,
            "maintenance_urgency": urgency,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def _predict_array(self, X: np.ndarray) -> np.ndarray:
        predicted = self.model.predict(X, verbose=0).reshape(-1)
        return np.clip(predicted, 0, Settings.MAX_RUL)

    def _build_results(
        self,
        engine_ids: np.ndarray,
        y_true: np.ndarray,
        predicted: np.ndarray,
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "engine_id": engine_ids,
                "actual_rul": y_true,
                "predicted_rul": np.round(predicted, 3),
                "prediction_timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
        return add_prediction_columns(df)

    def _predict_trajectories(
        self,
        preparer: SequencePreparer,
        trend_path: Path | None = None,
        max_engines: int = 5,
    ) -> pd.DataFrame:
        """Predict RUL through selected engine trajectories for trend charts."""
        data_path = (
            trend_path or Settings.OUTPUT_DIR / "processed" / "train_processed.csv"
        )
        if not data_path.exists():
            logger.warning(
                f"Trend data not found, skipping trajectory plots: {data_path}"
            )
            return pd.DataFrame()

        source_df = load_processed_csv(data_path)
        sample_engines = sorted(source_df["engine_id"].unique())[:max_engines]
        sample_df = source_df[source_df["engine_id"].isin(sample_engines)].copy()
        trajectories = preparer.prepare_full_trajectories(sample_df)

        rows = []
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

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _save_outputs(
        self,
        results_df: pd.DataFrame,
        metrics: dict | None,
        trend_df: pd.DataFrame | None,
    ) -> None:
        save_predictions_csv(
            results_df, self.predictions_dir, "rul_inference_results.csv"
        )
        save_predictions_csv(results_df, self.predictions_dir, "rul_predictions.csv")
        if metrics is not None:
            save_metrics_csv(
                [metrics], self.predictions_dir, "rul_inference_metrics.csv"
            )
        if trend_df is not None and not trend_df.empty:
            save_predictions_csv(
                trend_df, self.predictions_dir, "rul_inference_trends.csv"
            )

    def _save_plots(
        self,
        results_df: pd.DataFrame,
        metrics: dict | None,
        trend_df: pd.DataFrame | None,
    ) -> None:
        save_figure(
            plot_failure_risk(results_df), self.plots_dir, "rul_failure_risk.html"
        )
        save_figure(
            plot_risk_distribution(results_df["risk_category"].to_numpy()),
            self.plots_dir,
            "rul_risk_distribution.html",
        )
        save_figure(
            plot_urgency_summary(results_df["maintenance_urgency"].to_numpy()),
            self.plots_dir,
            "rul_urgency_summary.html",
        )
        save_figure(
            plot_confidence_heatmap(
                results_df["engine_id"].to_numpy(),
                results_df["predicted_rul"].to_numpy(),
                results_df["confidence"].to_numpy(),
                results_df["risk_category"].to_numpy(),
            ),
            self.plots_dir,
            "rul_confidence_heatmap.html",
        )

        if metrics is not None:
            save_figure(
                plot_predicted_vs_actual(
                    results_df["actual_rul"].to_numpy(),
                    results_df["predicted_rul"].to_numpy(),
                    "Inference Predicted vs Actual RUL",
                ),
                self.plots_dir,
                "rul_inference_predicted_vs_actual.html",
            )
            save_figure(
                plot_error_distribution(
                    results_df["actual_rul"].to_numpy(),
                    results_df["predicted_rul"].to_numpy(),
                    "Inference RUL Error Distribution",
                ),
                self.plots_dir,
                "rul_inference_error_distribution.html",
            )

        if trend_df is not None and not trend_df.empty:
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
                    plot_rul_trend(trend_df, int(engine_id)),
                    self.plots_dir,
                    f"rul_trend_engine_{int(engine_id)}.html",
                )
                if sensor_cols:
                    save_figure(
                        plot_degradation_curve(trend_df, int(engine_id), sensor_cols),
                        self.plots_dir,
                        f"rul_degradation_engine_{int(engine_id)}.html",
                    )

    def _log_summary(self, df: pd.DataFrame) -> None:
        total = len(df)
        high = int((df["risk_category"] == RISK_HIGH).sum())
        medium = int((df["risk_category"] == RISK_MEDIUM).sum())
        low = int((df["risk_category"] == RISK_LOW).sum())
        immediate = int((df["maintenance_urgency"] == URGENCY_IMMEDIATE).sum())
        priority = int((df["maintenance_urgency"] == URGENCY_PRIORITY).sum())

        logger.info(
            "RUL inference summary: "
            f"engines={total}, low={low}, medium={medium}, high={high}, "
            f"priority={priority}, immediate={immediate}, "
            f"avg_predicted_rul={df['predicted_rul'].mean():.2f}"
        )


if __name__ == "__main__":
    predictor = RULPredictor()
    predictor.load_model()
    predictor.predict()
