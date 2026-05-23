"""
╔══════════════════════════════════════════════════════════════╗
║   Predict Anomalies                                          ║
║   Real-Time & Batch Inference with Saved Model               ║
╚══════════════════════════════════════════════════════════════╝

Load a previously trained Isolation Forest model and run anomaly
prediction on new / unseen sensor data.

Designed for two usage modes:

    ▸ **Batch mode** — score an entire CSV of engine data at once
    ▸ **Real-time mode** — score one row (or a small DataFrame)
      representing a single sensor snapshot from a live stream

Usage:
    >>> from src.anomaly_detection.predict_anomalies import AnomalyPredictor
    >>> predictor = AnomalyPredictor()
    >>> predictor.load_model()
    >>> results = predictor.predict_batch()  # batch CSV
    >>> single  = predictor.predict_realtime(sensor_dict)  # real-time

    # Or run as script:
    $ python -m src.anomaly_detection.predict_anomalies
"""

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.settings import Settings
from src.utils.logger import setup_logger
from src.anomaly_detection.anomaly_utils import (
    get_sensor_feature_columns,
    normalize_anomaly_scores,
    classify_severity,
    compute_health_score,
    plot_anomaly_scores_distribution,
    plot_health_score_heatmap,
    plot_anomaly_timeline,
    plot_multi_sensor_anomaly_panel,
    save_anomaly_results,
    save_health_report,
    save_plotly_figure,
    SEVERITY_NORMAL,
    SEVERITY_WARNING,
    SEVERITY_CRITICAL,
)

logger = setup_logger(__name__)


class AnomalyPredictor:
    """
    Inference wrapper for trained Isolation Forest models.

    Supports batch CSV prediction and single-row real-time scoring.
    """

    def __init__(self):
        self.model = None
        self.feature_columns: list[str] = []
        self.contamination: float = Settings.ANOMALY_CONTAMINATION
        self.warning_threshold: float = 0.55
        self.critical_threshold: float = 0.80

        # Output paths
        self.predictions_dir = Settings.OUTPUT_DIR / "predictions"
        self.plots_dir = Settings.OUTPUT_DIR / "plots"
        self.model_dir = Settings.MODEL_DIR

    # ── Model Loading ───────────────────────────────────────

    def load_model(self, model_path: Path | None = None) -> "AnomalyPredictor":
        """
        Load a saved Isolation Forest model bundle.

        The bundle contains the model, feature columns, thresholds,
        and training metadata.

        Returns:
            ``self`` for method chaining.
        """
        model_path = model_path or (self.model_dir / "anomaly_isolation_forest.joblib")
        logger.info(f"Loading anomaly model from {model_path}")

        bundle = joblib.load(model_path)

        self.model = bundle["model"]
        self.feature_columns = bundle["feature_columns"]
        self.contamination = bundle.get("contamination", self.contamination)
        self.warning_threshold = bundle.get("warning_threshold", self.warning_threshold)
        self.critical_threshold = bundle.get(
            "critical_threshold", self.critical_threshold
        )

        logger.info(f"Model loaded — trained at {bundle.get('trained_at', 'N/A')}")
        logger.info(f"Feature columns: {len(self.feature_columns)}")
        return self

    # ── Batch Prediction ────────────────────────────────────

    def predict_batch(
        self,
        data_path: Path | None = None,
        df: pd.DataFrame | None = None,
        save: bool = True,
        generate_plots: bool = True,
    ) -> pd.DataFrame:
        """
        Run anomaly prediction on an entire dataset.

        Args:
            data_path: Path to CSV (defaults to test_processed.csv).
            df: Alternatively, pass a DataFrame directly.
            save: Whether to export results & plots.
            generate_plots: Whether to generate visualization charts.

        Returns:
            Annotated DataFrame with anomaly scores and severities.
        """
        self._assert_model_loaded()

        logger.info("═" * 60)
        logger.info("  ANOMALY DETECTION — BATCH PREDICTION")
        logger.info("═" * 60)

        # Load data
        if df is None:
            data_path = data_path or (
                Settings.OUTPUT_DIR / "processed" / "test_processed.csv"
            )
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)

        logger.info(f"Input shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

        # Predict
        df = self._score_dataframe(df)

        # Health scores
        health_df = None
        if "engine_id" in df.columns:
            health_df = compute_health_score(
                df["anomaly_score_norm"].values,
                df["engine_id"].values,
            )

        # Summary
        self._print_summary(df, health_df)

        # Export
        if save:
            save_anomaly_results(
                df,
                self.predictions_dir,
                "anomaly_predictions_batch.csv",
            )
            if health_df is not None:
                save_health_report(
                    health_df,
                    self.predictions_dir,
                    "health_scores_prediction.csv",
                )

        if generate_plots:
            self._generate_prediction_plots(df, health_df)

        logger.success("Batch prediction completed ✅")
        return df

    # ── Real-Time Prediction ────────────────────────────────

    def predict_realtime(
        self,
        sensor_data: dict | pd.DataFrame | pd.Series,
    ) -> dict:
        """
        Score a **single observation** (or small batch) in real-time.

        Accepts a dict, Series, or single-row DataFrame of sensor
        readings.  Returns a results dict suitable for API responses
        or dashboard consumption.

        Args:
            sensor_data: One row of sensor readings.

        Returns:
            dict with keys: ``anomaly_score``, ``is_anomaly``,
            ``severity``, ``severity_color``, ``health_estimate``,
            ``timestamp``.
        """
        self._assert_model_loaded()

        # Normalise input
        if isinstance(sensor_data, dict):
            row_df = pd.DataFrame([sensor_data])
        elif isinstance(sensor_data, pd.Series):
            row_df = sensor_data.to_frame().T
        else:
            row_df = sensor_data.copy()

        # Ensure all required features present
        missing = [c for c in self.feature_columns if c not in row_df.columns]
        if missing:
            logger.warning(f"Missing features (will zero-fill): {missing}")
            for c in missing:
                row_df[c] = 0.0

        X = row_df[self.feature_columns].values

        # Score
        raw_pred = self.model.predict(X)
        raw_score = self.model.decision_function(X)
        norm_score = float(normalize_anomaly_scores(raw_score)[0])
        severity_labels = classify_severity(
            np.array([norm_score]),
            self.warning_threshold,
            self.critical_threshold,
        )
        severity = str(severity_labels[0])

        from src.anomaly_detection.anomaly_utils import severity_color

        color = severity_color(severity)

        result = {
            "anomaly_score": round(norm_score, 4),
            "is_anomaly": bool(raw_pred[0] == -1),
            "severity": severity,
            "severity_color": color,
            "health_estimate": round(100 * (1 - norm_score), 1),
            "timestamp": datetime.now().isoformat(),
        }

        level = "warning" if result["is_anomaly"] else "info"
        getattr(logger, level)(
            f"Real-time prediction: score={result['anomaly_score']:.4f}  "
            f"severity={severity}  health≈{result['health_estimate']}%"
        )
        return result

    # ── Internals ───────────────────────────────────────────

    def _assert_model_loaded(self):
        if self.model is None:
            raise RuntimeError(
                "No model loaded. Call `load_model()` before prediction."
            )

    def _score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add anomaly scores, predictions, and severity to DataFrame."""
        # Align features (zero-fill any missing)
        for c in self.feature_columns:
            if c not in df.columns:
                df[c] = 0.0

        X = df[self.feature_columns].values

        df["anomaly_pred"] = self.model.predict(X)
        df["anomaly_score_raw"] = self.model.decision_function(X)
        df["anomaly_score_norm"] = normalize_anomaly_scores(
            df["anomaly_score_raw"].values,
        )
        df["is_anomaly"] = (df["anomaly_pred"] == -1).astype(np.int8)
        df["severity"] = classify_severity(
            df["anomaly_score_norm"].values,
            self.warning_threshold,
            self.critical_threshold,
        )
        return df

    def _print_summary(
        self,
        df: pd.DataFrame,
        health_df: pd.DataFrame | None,
    ):
        """Log prediction summary."""
        total = len(df)
        n_normal = (df["severity"] == SEVERITY_NORMAL).sum()
        n_warning = (df["severity"] == SEVERITY_WARNING).sum()
        n_critical = (df["severity"] == SEVERITY_CRITICAL).sum()

        logger.info("─" * 60)
        logger.info("  PREDICTION SUMMARY")
        logger.info("─" * 60)
        logger.info(f"  Total samples        : {total:>10,}")
        logger.info(
            f"  🟢 Normal            : {n_normal:>10,}  ({n_normal/total*100:5.1f}%)"
        )
        logger.info(
            f"  🟡 Warning           : {n_warning:>10,}  ({n_warning/total*100:5.1f}%)"
        )
        logger.info(
            f"  🔴 Critical          : {n_critical:>10,}  ({n_critical/total*100:5.1f}%)"
        )

        if health_df is not None:
            avg_h = health_df["health_score"].mean()
            worst = health_df.iloc[0]
            logger.info(f"  Avg health score     : {avg_h:>10.1f}")
            logger.info(
                f"  Worst engine         :    #{int(worst['engine_id'])}  "
                f"(score={worst['health_score']:.1f})"
            )
        logger.info("─" * 60)

    def _generate_prediction_plots(
        self,
        df: pd.DataFrame,
        health_df: pd.DataFrame | None,
    ):
        """Generate plots for prediction results."""
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Score distribution
        fig = plot_anomaly_scores_distribution(
            df["anomaly_score_norm"].values,
            df["severity"].values,
            title="Prediction — Anomaly Score Distribution",
        )
        save_plotly_figure(fig, self.plots_dir, "pred_anomaly_distribution.html")

        # Health heatmap
        if health_df is not None:
            fig = plot_health_score_heatmap(health_df)
            save_plotly_figure(fig, self.plots_dir, "pred_health_scores.html")

            # Timeline for worst 3
            worst = health_df.head(3)
            sensors = self.feature_columns[:4]
            for _, row in worst.iterrows():
                eid = int(row["engine_id"])
                fig = plot_anomaly_timeline(df, eid)
                save_plotly_figure(
                    fig,
                    self.plots_dir,
                    f"pred_timeline_engine_{eid}.html",
                )
                if sensors:
                    fig = plot_multi_sensor_anomaly_panel(df, eid, sensors)
                    save_plotly_figure(
                        fig,
                        self.plots_dir,
                        f"pred_sensor_panel_engine_{eid}.html",
                    )

        logger.info(f"Prediction plots saved to {self.plots_dir}")


# ═══════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    Settings.ensure_directories()
    predictor = AnomalyPredictor()
    predictor.load_model()
    predictor.predict_batch()
