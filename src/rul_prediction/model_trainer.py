"""
╔══════════════════════════════════════════════════════════════╗
║   Model Trainer — Training Orchestrator                     ║
╚══════════════════════════════════════════════════════════════╝

Handles the training workflow for both deep learning and
classical ML models, including data splitting and sequencing.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from configs.settings import Settings
from src.rul_prediction.lstm_model import LSTMModel
from src.rul_prediction.classical_models import ClassicalModels
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelTrainer:
    """Orchestrate training of all RUL prediction models."""

    def __init__(self):
        self.lstm_model = None
        self.classical_models = ClassicalModels()
        self.history = None

    def prepare_lstm_data(self, train_df: pd.DataFrame) -> tuple:
        """
        Prepare sequence data for LSTM training.

        Returns:
            tuple: (X_train, X_val, y_train, y_val) as numpy arrays
        """
        feature_cols = [
            c
            for c in Settings.SENSOR_COLUMNS
            if c in train_df.columns and c not in Settings.DROP_SENSORS
        ]

        # Build sequences per engine
        X_all, y_all = [], []

        for engine_id in train_df["engine_id"].unique():
            engine_data = train_df[train_df["engine_id"] == engine_id]
            features = engine_data[feature_cols].values
            labels = engine_data["rul"].values

            if len(features) >= Settings.SEQUENCE_LENGTH:
                X_seq, y_seq = LSTMModel.create_sequences(features, labels)
                X_all.append(X_seq)
                y_all.append(y_seq)

        X = np.concatenate(X_all, axis=0)
        y = np.concatenate(y_all, axis=0)

        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info(f"LSTM data: Train={X_train.shape}, Val={X_val.shape}")
        return X_train, X_val, y_train, y_val

    def prepare_classical_data(self, train_df: pd.DataFrame) -> tuple:
        """
        Prepare flat feature data for classical ML models.
        Uses the last cycle per engine as the representative sample.

        Returns:
            tuple: (X_train, X_val, y_train, y_val)
        """
        feature_cols = [
            c
            for c in Settings.SENSOR_COLUMNS
            if c in train_df.columns and c not in Settings.DROP_SENSORS
        ]

        # Use last cycle per engine (most degraded state)
        last_cycles = train_df.groupby("engine_id").last().reset_index()

        X = last_cycles[feature_cols].values
        y = last_cycles["rul"].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info(f"Classical data: Train={X_train.shape}, Val={X_val.shape}")
        return X_train, X_val, y_train, y_val

    def train_lstm(self, train_df: pd.DataFrame) -> dict:
        """Train the LSTM model."""
        logger.info("Training LSTM model...")

        X_train, X_val, y_train, y_val = self.prepare_lstm_data(train_df)

        n_features = X_train.shape[2]
        self.lstm_model = LSTMModel(n_features=n_features)
        self.lstm_model.build_lstm()

        self.history = self.lstm_model.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=Settings.LSTM_EPOCHS,
            batch_size=Settings.LSTM_BATCH_SIZE,
            callbacks=self.lstm_model.get_callbacks(),
            verbose=1,
        )

        self.lstm_model.save()
        logger.success("LSTM training complete ✅")

        return {"lstm": self.lstm_model}

    def train_classical(self, train_df: pd.DataFrame) -> dict:
        """Train all classical ML models."""
        logger.info("Training classical ML models...")

        X_train, X_val, y_train, y_val = self.prepare_classical_data(train_df)

        self.classical_models.build_all()

        for name in self.classical_models.models:
            self.classical_models.train(name, X_train, y_train)

        self.classical_models.save_models()
        logger.success("Classical model training complete ✅")

        return self.classical_models.models

    def train_all(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame = None,
        rul_df: pd.DataFrame = None,
    ) -> dict:
        """Train all models (LSTM + classical)."""
        trained = {}

        trained.update(self.train_lstm(train_df))
        trained.update(self.train_classical(train_df))

        logger.success(f"All models trained: {list(trained.keys())}")
        return trained
