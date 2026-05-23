"""
╔══════════════════════════════════════════════════════════════╗
║   Global Settings & Hyperparameters                         ║
╚══════════════════════════════════════════════════════════════╝

Centralized configuration for the entire application.
All settings can be overridden via environment variables or .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings:
    """Application-wide configuration settings."""

    # ── Application ──────────────────────────────────────────
    APP_NAME: str = os.getenv("APP_NAME", "Predictive Maintenance Platform")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # ── Data Paths ───────────────────────────────────────────
    DATA_DIR: Path = PROJECT_ROOT / os.getenv("DATA_DIR", "data/nasa")
    PROCESSED_DIR: Path = PROJECT_ROOT / os.getenv("PROCESSED_DIR", "data/processed")
    MODEL_DIR: Path = PROJECT_ROOT / os.getenv("MODEL_DIR", "models")
    OUTPUT_DIR: Path = PROJECT_ROOT / os.getenv("OUTPUT_DIR", "outputs")

    # ── Dataset Files ────────────────────────────────────────
    TRAIN_FILE: str = "train_FD001.txt"
    TEST_FILE: str = "test_FD001.txt"
    RUL_FILE: str = "RUL_FD001.txt"

    # ── Feature Configuration ────────────────────────────────
    OPERATIONAL_SETTINGS: list = ["op_setting_1", "op_setting_2", "op_setting_3"]
    SENSOR_COLUMNS: list = [f"sensor_{i}" for i in range(1, 22)]
    INDEX_COLUMNS: list = ["engine_id", "cycle"]
    ALL_COLUMNS: list = INDEX_COLUMNS + OPERATIONAL_SETTINGS + SENSOR_COLUMNS

    # Sensors with near-zero variance (to drop)
    DROP_SENSORS: list = [
        "sensor_1",
        "sensor_5",
        "sensor_6",
        "sensor_10",
        "sensor_16",
        "sensor_18",
        "sensor_19",
    ]

    # ── RUL Configuration ────────────────────────────────────
    MAX_RUL: int = int(os.getenv("MAX_RUL", "125"))
    SEQUENCE_LENGTH: int = int(os.getenv("SEQUENCE_LENGTH", "30"))

    # ── LSTM Hyperparameters ─────────────────────────────────
    LSTM_UNITS: int = 64
    LSTM_DROPOUT: float = 0.2
    LSTM_EPOCHS: int = int(os.getenv("LSTM_EPOCHS", "50"))
    LSTM_BATCH_SIZE: int = int(os.getenv("LSTM_BATCH_SIZE", "64"))
    LSTM_LEARNING_RATE: float = float(os.getenv("LSTM_LEARNING_RATE", "0.001"))
    EARLY_STOPPING_PATIENCE: int = 10
    REDUCE_LR_PATIENCE: int = 5

    # ── Classical ML Hyperparameters ─────────────────────────
    RF_N_ESTIMATORS: int = 100
    RF_MAX_DEPTH: int = 15
    GB_N_ESTIMATORS: int = 200
    GB_LEARNING_RATE: float = 0.1
    GB_MAX_DEPTH: int = 5

    # ── Anomaly Detection ────────────────────────────────────
    ANOMALY_CONTAMINATION: float = 0.05
    ZSCORE_THRESHOLD: float = 3.0
    IQR_MULTIPLIER: float = 1.5

    # ── Dashboard ────────────────────────────────────────────
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    STREAMLIT_THEME: str = os.getenv("STREAMLIT_THEME", "dark")

    # ── Logging ──────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "outputs/logs/app.log")

    @classmethod
    def ensure_directories(cls):
        """Create all required output directories if they don't exist."""
        dirs = [
            cls.PROCESSED_DIR,
            cls.MODEL_DIR,
            cls.OUTPUT_DIR / "logs",
            cls.OUTPUT_DIR / "reports",
            cls.OUTPUT_DIR / "figures",
            cls.OUTPUT_DIR / "processed",
            cls.OUTPUT_DIR / "predictions",
            cls.OUTPUT_DIR / "plots",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def display(cls):
        """Print current configuration for debugging."""
        print("\n" + "=" * 60)
        print("  Current Configuration")
        print("=" * 60)
        for attr in sorted(dir(cls)):
            if attr.isupper():
                print(f"  {attr:<30} = {getattr(cls, attr)}")
        print("=" * 60 + "\n")
