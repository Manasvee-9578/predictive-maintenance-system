"""
╔══════════════════════════════════════════════════════════════╗
║   LSTM Model — Deep Learning RUL Prediction                 ║
╚══════════════════════════════════════════════════════════════╝

Implements LSTM and Bidirectional LSTM architectures for
sequence-based Remaining Useful Life prediction.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path

from configs.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LSTMModel:
    """LSTM-based deep learning model for RUL prediction."""

    def __init__(self, n_features: int, sequence_length: int = None):
        self.n_features = n_features
        self.sequence_length = sequence_length or Settings.SEQUENCE_LENGTH
        self.model = None

    def build_lstm(self) -> keras.Model:
        """Build a standard LSTM architecture."""
        logger.info(
            f"Building LSTM model: seq_len={self.sequence_length}, "
            f"features={self.n_features}"
        )

        model = Sequential(
            [
                LSTM(
                    units=Settings.LSTM_UNITS,
                    input_shape=(self.sequence_length, self.n_features),
                    return_sequences=True,
                ),
                Dropout(Settings.LSTM_DROPOUT),
                LSTM(units=Settings.LSTM_UNITS // 2, return_sequences=False),
                Dropout(Settings.LSTM_DROPOUT),
                Dense(64, activation="relu"),
                BatchNormalization(),
                Dropout(Settings.LSTM_DROPOUT),
                Dense(32, activation="relu"),
                Dense(1, activation="linear"),  # Regression output
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=Settings.LSTM_LEARNING_RATE),
            loss="mse",
            metrics=["mae"],
        )

        self.model = model
        model.summary(print_fn=logger.info)
        return model

    def build_bilstm(self) -> keras.Model:
        """Build a Bidirectional LSTM architecture."""
        logger.info(
            f"Building BiLSTM model: seq_len={self.sequence_length}, "
            f"features={self.n_features}"
        )

        model = Sequential(
            [
                Bidirectional(
                    LSTM(units=Settings.LSTM_UNITS, return_sequences=True),
                    input_shape=(self.sequence_length, self.n_features),
                ),
                Dropout(Settings.LSTM_DROPOUT),
                Bidirectional(
                    LSTM(units=Settings.LSTM_UNITS // 2, return_sequences=False)
                ),
                Dropout(Settings.LSTM_DROPOUT),
                Dense(64, activation="relu"),
                BatchNormalization(),
                Dropout(Settings.LSTM_DROPOUT),
                Dense(32, activation="relu"),
                Dense(1, activation="linear"),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=Settings.LSTM_LEARNING_RATE),
            loss="mse",
            metrics=["mae"],
        )

        self.model = model
        model.summary(print_fn=logger.info)
        return model

    def get_callbacks(self) -> list:
        """Return training callbacks for early stopping, LR scheduling, and checkpointing."""
        model_path = Settings.MODEL_DIR / "best_lstm_model.keras"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        return [
            EarlyStopping(
                monitor="val_loss",
                patience=Settings.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=Settings.REDUCE_LR_PATIENCE,
                min_lr=1e-6,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(model_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

    @staticmethod
    def create_sequences(
        data: np.ndarray, labels: np.ndarray, sequence_length: int = None
    ) -> tuple:
        """
        Create sliding window sequences for LSTM input.

        Args:
            data: Feature array (n_samples, n_features)
            labels: RUL labels array
            sequence_length: Window size

        Returns:
            tuple: (X_sequences, y_labels)
        """
        seq_len = sequence_length or Settings.SEQUENCE_LENGTH
        X, y = [], []

        for i in range(len(data) - seq_len + 1):
            X.append(data[i : i + seq_len])
            y.append(labels[i + seq_len - 1])

        return np.array(X), np.array(y)

    def save(self, filepath: Path = None):
        """Save the trained Keras model."""
        filepath = filepath or Settings.MODEL_DIR / "lstm_model.keras"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: Path = None):
        """Load a previously saved Keras model."""
        filepath = filepath or Settings.MODEL_DIR / "lstm_model.keras"
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self.model
