"""
╔══════════════════════════════════════════════════════════════╗
║   Train Anomaly Model                                        ║
║   Isolation Forest Training & Health Scoring Pipeline        ║
╚══════════════════════════════════════════════════════════════╝

Production-grade training pipeline for anomaly detection:

    1. Load preprocessed sensor data
    2. Train Isolation Forest with configurable contamination
    3. Compute normalised anomaly scores
    4. Classify severity (Normal / Warning / Critical)
    5. Calculate per-engine health scores (0 – 100)
    6. Generate dashboard-ready Plotly visualizations
    7. Persist model (joblib), results (CSV), and plots (HTML)

Usage:
    >>> from src.anomaly_detection.train_anomaly_model import AnomalyTrainer
    >>> trainer = AnomalyTrainer(contamination=0.05)
    >>> results = trainer.run()

    # Or run as script:
    $ python -m src.anomaly_detection.train_anomaly_model
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
import sys
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import IsolationForest

# ── Ensure project root is on path ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.settings import Settings
from src.utils.logger import setup_logger
from src.anomaly_detection.anomaly_utils import (
    get_sensor_feature_columns,
    normalize_anomaly_scores,
    classify_severity,
    compute_health_score,
    plot_anomaly_scores_distribution,
    plot_sensor_with_anomalies,
    plot_health_score_gauge,
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


# ═══════════════════════════════════════════════════════════════
#  Configuration Defaults
# ═══════════════════════════════════════════════════════════════

DEFAULT_CONTAMINATION = Settings.ANOMALY_CONTAMINATION  # 0.05
DEFAULT_N_ESTIMATORS = 200
DEFAULT_MAX_SAMPLES = "auto"
DEFAULT_RANDOM_STATE = 42
DEFAULT_WARNING_THRESH = 0.55
DEFAULT_CRIT_THRESH = 0.80


class AnomalyTrainer:
    """
    End-to-end Isolation Forest anomaly training pipeline.

    Attributes:
        contamination: Expected proportion of anomalies.
        n_estimators:  Number of isolation trees.
        model:         Trained ``IsolationForest`` instance (populated
                       after ``train()``).
    """

    def __init__(
        self,
        contamination: float = DEFAULT_CONTAMINATION,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_samples: str | int = DEFAULT_MAX_SAMPLES,
        random_state: int = DEFAULT_RANDOM_STATE,
        warning_threshold: float = DEFAULT_WARNING_THRESH,
        critical_threshold: float = DEFAULT_CRIT_THRESH,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        self.model: IsolationForest | None = None
        self.feature_columns: list[str] = []

        # Output paths
        self.model_dir = Settings.MODEL_DIR
        self.predictions_dir = Settings.OUTPUT_DIR / "predictions"
        self.plots_dir = Settings.OUTPUT_DIR / "plots"

    # ── Public API ──────────────────────────────────────────

    def run(self, data_path: Path | None = None) -> dict:
        """
        Execute the full training pipeline.

        1. Load data
        2. Train model
        3. Score & classify
        4. Compute health
        5. Visualize
        6. Export everything

        Returns:
            dict with keys: ``df``, ``health_df``, ``model``,
            ``feature_columns``, ``stats``.
        """
        logger.info("═" * 60)
        logger.info("  ANOMALY DETECTION — TRAINING PIPELINE")
        logger.info("═" * 60)

        # 1. Load
        df = self._load_data(data_path)

        # 2. Train
        df = self.train(df)

        # 3. Health scores
        health_df = compute_health_score(
            df["anomaly_score_norm"].values,
            df["engine_id"].values,
        )

        # 4. Visualise & save
        self._generate_visualizations(df, health_df)

        # 5. Export
        self._save_model()
        save_anomaly_results(df, self.predictions_dir, "anomaly_results.csv")
        save_health_report(health_df, self.predictions_dir, "health_scores.csv")

        # 6. Summary
        stats = self._summary(df, health_df)

        logger.success("Anomaly training pipeline completed ✅")
        return {
            "df": df,
            "health_df": health_df,
            "model": self.model,
            "feature_columns": self.feature_columns,
            "stats": stats,
        }

    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Train the Isolation Forest and annotate *df* in-place with:

        - ``anomaly_pred``       — raw prediction (1 = inlier, −1 = anomaly)
        - ``anomaly_score_raw``  — raw decision-function output
        - ``anomaly_score_norm`` — normalised score [0, 1]
        - ``is_anomaly``         — boolean flag
        - ``severity``           — Normal / Warning / Critical
        """
        self.feature_columns = get_sensor_feature_columns(df)
        logger.info(f"Feature columns: {len(self.feature_columns)}")
        logger.info(
            f"Training Isolation Forest  "
            f"(contamination={self.contamination}, "
            f"n_estimators={self.n_estimators})"
        )

        X = df[self.feature_columns].values

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
            warm_start=False,
        )

        df["anomaly_pred"] = self.model.fit_predict(X)
        df["anomaly_score_raw"] = self.model.decision_function(X)
        df["anomaly_score_norm"] = normalize_anomaly_scores(
            df["anomaly_score_raw"].values,
        )
        df["is_anomaly"] = (df["anomaly_pred"] == -1).astype(np.int8)
        df["severity"] = classify_severity(
            df["anomaly_score_norm"].values,
            warning_threshold=self.warning_threshold,
            critical_threshold=self.critical_threshold,
        )

        n_anom = df["is_anomaly"].sum()
        logger.info(
            f"Anomalies detected: {n_anom:,} / {len(df):,} "
            f"({n_anom / len(df) * 100:.1f}%)"
        )
        return df

    # ── Data Loading ────────────────────────────────────────

    def _load_data(self, path: Path | None = None) -> pd.DataFrame:
        """Load the preprocessed training CSV."""
        path = path or (Settings.OUTPUT_DIR / "processed" / "train_processed.csv")
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df

    # ── Model Persistence ───────────────────────────────────

    def _save_model(self):
        """Persist the trained Isolation Forest to disk."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / "anomaly_isolation_forest.joblib"
        joblib.dump(
            {
                "model": self.model,
                "feature_columns": self.feature_columns,
                "contamination": self.contamination,
                "warning_threshold": self.warning_threshold,
                "critical_threshold": self.critical_threshold,
                "trained_at": datetime.now().isoformat(),
            },
            model_path,
        )
        logger.info(f"Model saved → {model_path}")

    # ── Visualizations ──────────────────────────────────────

    def _generate_visualizations(
        self,
        df: pd.DataFrame,
        health_df: pd.DataFrame,
    ):
        """Create and save all dashboard-ready Plotly charts."""
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Generating anomaly visualizations …")

        # 1. Score distribution
        fig = plot_anomaly_scores_distribution(
            df["anomaly_score_norm"].values,
            df["severity"].values,
        )
        save_plotly_figure(fig, self.plots_dir, "anomaly_score_distribution.html")

        # 2. Health score bar chart (all engines)
        fig = plot_health_score_heatmap(health_df)
        save_plotly_figure(fig, self.plots_dir, "health_scores_all_engines.html")

        # 3. Per-engine deep-dives (top 3 most critical)
        worst = health_df.head(3)
        sensors_for_plot = self.feature_columns[:4]  # first 4 sensors

        for _, row in worst.iterrows():
            eid = int(row["engine_id"])

            # Gauge
            fig = plot_health_score_gauge(row["health_score"], eid)
            save_plotly_figure(fig, self.plots_dir, f"health_gauge_engine_{eid}.html")

            # Anomaly timeline
            fig = plot_anomaly_timeline(df, eid)
            save_plotly_figure(
                fig, self.plots_dir, f"anomaly_timeline_engine_{eid}.html"
            )

            # Multi-sensor panel
            if sensors_for_plot:
                fig = plot_multi_sensor_anomaly_panel(
                    df,
                    eid,
                    sensors_for_plot,
                )
                save_plotly_figure(
                    fig,
                    self.plots_dir,
                    f"sensor_panel_engine_{eid}.html",
                )

            # Single sensor highlight (first sensor)
            if sensors_for_plot:
                fig = plot_sensor_with_anomalies(
                    df,
                    sensors_for_plot[0],
                    eid,
                )
                save_plotly_figure(
                    fig,
                    self.plots_dir,
                    f"sensor_anomaly_engine_{eid}_{sensors_for_plot[0]}.html",
                )

        logger.info(f"All plots saved to {self.plots_dir}")

    # ── Summary ─────────────────────────────────────────────

    def _summary(self, df: pd.DataFrame, health_df: pd.DataFrame) -> dict:
        """Log and return pipeline statistics."""
        total = len(df)
        n_normal = (df["severity"] == SEVERITY_NORMAL).sum()
        n_warning = (df["severity"] == SEVERITY_WARNING).sum()
        n_critical = (df["severity"] == SEVERITY_CRITICAL).sum()

        avg_health = health_df["health_score"].mean()
        worst_eng = health_df.iloc[0]

        stats = {
            "total_samples": total,
            "normal": n_normal,
            "warning": n_warning,
            "critical": n_critical,
            "avg_health_score": round(avg_health, 1),
            "worst_engine_id": int(worst_eng["engine_id"]),
            "worst_engine_health": round(worst_eng["health_score"], 1),
        }

        logger.info("─" * 60)
        logger.info("  ANOMALY DETECTION SUMMARY")
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
        logger.info(f"  Avg health score     : {avg_health:>10.1f}")
        logger.info(
            f"  Worst engine         :    #{int(worst_eng['engine_id'])}  "
            f"(score={worst_eng['health_score']:.1f})"
        )
        logger.info("─" * 60)
        return stats


# ═══════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    Settings.ensure_directories()
    trainer = AnomalyTrainer()
    trainer.run()
