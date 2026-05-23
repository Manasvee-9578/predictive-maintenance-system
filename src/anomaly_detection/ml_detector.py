"""
╔══════════════════════════════════════════════════════════════╗
║   ML-Based Anomaly Detection                                ║
╚══════════════════════════════════════════════════════════════╝

Implements machine learning methods for detecting sensor anomalies:
- Isolation Forest
- One-Class SVM
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import joblib
from pathlib import Path

from configs.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MLAnomalyDetector:
    """ML-based anomaly detection using Isolation Forest and One-Class SVM."""

    def __init__(self, contamination: float = None):
        self.contamination = contamination or Settings.ANOMALY_CONTAMINATION
        self.models = {}

    def fit_isolation_forest(
        self, df: pd.DataFrame, columns: list = None
    ) -> np.ndarray:
        """
        Fit and predict using Isolation Forest.

        Args:
            df: Input DataFrame
            columns: Feature columns to use

        Returns:
            Array of predictions (-1 = anomaly, 1 = normal)
        """
        columns = columns or [
            c
            for c in Settings.SENSOR_COLUMNS
            if c in df.columns and c not in Settings.DROP_SENSORS
        ]

        logger.info(f"Training Isolation Forest (contamination={self.contamination})")

        model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )

        X = df[columns].values
        predictions = model.fit_predict(X)
        anomaly_scores = model.decision_function(X)

        self.models["isolation_forest"] = model

        n_anomalies = (predictions == -1).sum()
        logger.info(
            f"Isolation Forest: {n_anomalies:,} anomalies detected "
            f"({n_anomalies / len(predictions) * 100:.1f}%)"
        )

        return predictions, anomaly_scores

    def fit_one_class_svm(
        self, df: pd.DataFrame, columns: list = None, sample_size: int = 5000
    ) -> np.ndarray:
        """
        Fit and predict using One-Class SVM.
        Uses sampling for large datasets (SVM doesn't scale well).

        Args:
            df: Input DataFrame
            columns: Feature columns to use
            sample_size: Max samples for fitting (SVM scaling)

        Returns:
            Array of predictions (-1 = anomaly, 1 = normal)
        """
        columns = columns or [
            c
            for c in Settings.SENSOR_COLUMNS
            if c in df.columns and c not in Settings.DROP_SENSORS
        ]

        logger.info(f"Training One-Class SVM (sample_size={sample_size})")

        X = df[columns].values

        # Sub-sample for training if dataset is too large
        if len(X) > sample_size:
            indices = np.random.RandomState(42).choice(
                len(X), sample_size, replace=False
            )
            X_train = X[indices]
        else:
            X_train = X

        model = OneClassSVM(kernel="rbf", gamma="auto", nu=self.contamination)
        model.fit(X_train)

        predictions = model.predict(X)
        self.models["one_class_svm"] = model

        n_anomalies = (predictions == -1).sum()
        logger.info(
            f"One-Class SVM: {n_anomalies:,} anomalies detected "
            f"({n_anomalies / len(predictions) * 100:.1f}%)"
        )

        return predictions

    def save_models(self, output_dir: Path = None):
        """Serialize trained anomaly detection models."""
        output_dir = output_dir or Settings.MODEL_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            path = output_dir / f"anomaly_{name}.joblib"
            joblib.dump(model, path)
            logger.info(f"Saved {name} model to {path}")

    def load_model(self, model_name: str, model_dir: Path = None):
        """Load a previously saved anomaly detection model."""
        model_dir = model_dir or Settings.MODEL_DIR
        path = model_dir / f"anomaly_{model_name}.joblib"
        self.models[model_name] = joblib.load(path)
        logger.info(f"Loaded {model_name} from {path}")
        return self.models[model_name]
