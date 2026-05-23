"""
╔══════════════════════════════════════════════════════════════╗
║   Statistical Anomaly Detection                             ║
╚══════════════════════════════════════════════════════════════╝

Implements classical statistical methods for detecting sensor anomalies:
- Z-Score method
- IQR (Interquartile Range) method
"""

import pandas as pd
import numpy as np

from configs.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class StatisticalDetector:
    """Statistical anomaly detection using Z-Score and IQR methods."""

    def __init__(self, zscore_threshold: float = None, iqr_multiplier: float = None):
        self.zscore_threshold = zscore_threshold or Settings.ZSCORE_THRESHOLD
        self.iqr_multiplier = iqr_multiplier or Settings.IQR_MULTIPLIER

    def detect_zscore(self, df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """
        Detect anomalies using Z-Score method.

        Args:
            df: Input DataFrame with sensor columns
            columns: Columns to check (defaults to sensor columns)

        Returns:
            DataFrame with boolean anomaly flags per column
        """
        columns = columns or [
            c
            for c in Settings.SENSOR_COLUMNS
            if c in df.columns and c not in Settings.DROP_SENSORS
        ]

        logger.info(
            f"Z-Score anomaly detection on {len(columns)} columns "
            f"(threshold={self.zscore_threshold})"
        )

        anomalies = pd.DataFrame(index=df.index)

        for col in columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            anomalies[f"{col}_anomaly"] = z_scores > self.zscore_threshold

        total = anomalies.sum().sum()
        logger.info(f"Z-Score detected {total:,} anomalous readings")
        return anomalies

    def detect_iqr(self, df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """
        Detect anomalies using IQR (Interquartile Range) method.

        Args:
            df: Input DataFrame with sensor columns
            columns: Columns to check

        Returns:
            DataFrame with boolean anomaly flags per column
        """
        columns = columns or [
            c
            for c in Settings.SENSOR_COLUMNS
            if c in df.columns and c not in Settings.DROP_SENSORS
        ]

        logger.info(
            f"IQR anomaly detection on {len(columns)} columns "
            f"(multiplier={self.iqr_multiplier})"
        )

        anomalies = pd.DataFrame(index=df.index)

        for col in columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            anomalies[f"{col}_anomaly"] = (df[col] < lower) | (df[col] > upper)

        total = anomalies.sum().sum()
        logger.info(f"IQR detected {total:,} anomalous readings")
        return anomalies
