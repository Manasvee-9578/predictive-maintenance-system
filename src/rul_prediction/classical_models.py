"""
╔══════════════════════════════════════════════════════════════╗
║   Classical ML Models — Baseline RUL Prediction             ║
╚══════════════════════════════════════════════════════════════╝

Implements classical machine learning models as baselines:
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import joblib
from pathlib import Path

from configs.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ClassicalModels:
    """Classical ML models for RUL prediction."""

    def __init__(self):
        self.models = {}

    def build_random_forest(self) -> RandomForestRegressor:
        """Build a Random Forest Regressor."""
        logger.info(
            f"Building Random Forest (n_estimators={Settings.RF_N_ESTIMATORS}, "
            f"max_depth={Settings.RF_MAX_DEPTH})"
        )

        model = RandomForestRegressor(
            n_estimators=Settings.RF_N_ESTIMATORS,
            max_depth=Settings.RF_MAX_DEPTH,
            random_state=42,
            n_jobs=-1,
        )
        self.models["random_forest"] = model
        return model

    def build_gradient_boosting(self) -> GradientBoostingRegressor:
        """Build a Gradient Boosting Regressor."""
        logger.info(
            f"Building Gradient Boosting (n_estimators={Settings.GB_N_ESTIMATORS}, "
            f"lr={Settings.GB_LEARNING_RATE})"
        )

        model = GradientBoostingRegressor(
            n_estimators=Settings.GB_N_ESTIMATORS,
            learning_rate=Settings.GB_LEARNING_RATE,
            max_depth=Settings.GB_MAX_DEPTH,
            random_state=42,
        )
        self.models["gradient_boosting"] = model
        return model

    def build_svr(self) -> SVR:
        """Build a Support Vector Regressor."""
        logger.info("Building SVR (kernel=rbf)")

        model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
        self.models["svr"] = model
        return model

    def build_all(self) -> dict:
        """Build all classical models."""
        self.build_random_forest()
        self.build_gradient_boosting()
        self.build_svr()
        return self.models

    def train(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray):
        """Train a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Build it first.")

        logger.info(f"Training {model_name} on {X_train.shape[0]:,} samples...")
        self.models[model_name].fit(X_train, y_train)
        logger.success(f"{model_name} training complete ✅")

    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Generate predictions from a trained model."""
        return self.models[model_name].predict(X)

    def save_models(self, output_dir: Path = None):
        """Save all trained models to disk."""
        output_dir = output_dir or Settings.MODEL_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            path = output_dir / f"rul_{name}.joblib"
            joblib.dump(model, path)
            logger.info(f"Saved {name} → {path}")

    def load_model(self, model_name: str, model_dir: Path = None):
        """Load a saved model from disk."""
        model_dir = model_dir or Settings.MODEL_DIR
        path = model_dir / f"rul_{model_name}.joblib"
        self.models[model_name] = joblib.load(path)
        logger.info(f"Loaded {model_name} from {path}")
        return self.models[model_name]
