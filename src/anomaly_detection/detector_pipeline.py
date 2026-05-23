"""
╔══════════════════════════════════════════════════════════════╗
║   Anomaly Detection Pipeline — Orchestrator                 ║
╚══════════════════════════════════════════════════════════════╝

Runs all anomaly detection methods and aggregates results.
"""

import pandas as pd
import numpy as np

from src.anomaly_detection.statistical import StatisticalDetector
from src.anomaly_detection.ml_detector import MLAnomalyDetector
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DetectorPipeline:
    """Orchestrate multiple anomaly detection methods."""

    def __init__(self):
        self.stat_detector = StatisticalDetector()
        self.ml_detector = MLAnomalyDetector()

    def run(self, df: pd.DataFrame) -> dict:
        """
        Run all anomaly detection methods on the input data.

        Args:
            df: Preprocessed DataFrame with sensor columns

        Returns:
            dict with results from each detection method
        """
        logger.info("Running anomaly detection pipeline...")
        results = {}

        # 1. Statistical methods
        results["zscore_anomalies"] = self.stat_detector.detect_zscore(df)
        results["iqr_anomalies"] = self.stat_detector.detect_iqr(df)

        # 2. ML-based methods
        iso_preds, iso_scores = self.ml_detector.fit_isolation_forest(df)
        results["isolation_forest"] = {
            "predictions": iso_preds,
            "scores": iso_scores,
        }

        ocsvm_preds = self.ml_detector.fit_one_class_svm(df)
        results["one_class_svm"] = {"predictions": ocsvm_preds}

        # 3. Save models
        self.ml_detector.save_models()

        # Summary
        logger.success("Anomaly detection pipeline complete ✅")
        self._print_summary(results, len(df))

        return results

    def _print_summary(self, results: dict, n_total: int):
        """Log a summary of anomaly detection results."""
        logger.info("─" * 50)
        logger.info("Anomaly Detection Summary")
        logger.info("─" * 50)

        zscore_count = results["zscore_anomalies"].sum().sum()
        iqr_count = results["iqr_anomalies"].sum().sum()
        iso_count = (results["isolation_forest"]["predictions"] == -1).sum()
        svm_count = (results["one_class_svm"]["predictions"] == -1).sum()

        logger.info(f"  Z-Score:          {zscore_count:>8,} anomalous readings")
        logger.info(f"  IQR:              {iqr_count:>8,} anomalous readings")
        logger.info(
            f"  Isolation Forest: {iso_count:>8,} anomalous samples ({iso_count/n_total*100:.1f}%)"
        )
        logger.info(
            f"  One-Class SVM:    {svm_count:>8,} anomalous samples ({svm_count/n_total*100:.1f}%)"
        )
        logger.info("─" * 50)
