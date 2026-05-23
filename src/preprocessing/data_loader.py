"""
╔══════════════════════════════════════════════════════════════╗
║   Data Loader — NASA C-MAPSS Dataset                        ║
╚══════════════════════════════════════════════════════════════╝

Loads and parses raw C-MAPSS turbofan engine degradation data
from whitespace-delimited text files.
"""

import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import pandas as pd
import numpy as np

from configs.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """Load raw NASA C-MAPSS dataset files."""

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Settings.DATA_DIR

    def load_train(self) -> pd.DataFrame:
        """Load training data (run-to-failure trajectories)."""
        filepath = self.data_dir / Settings.TRAIN_FILE
        logger.info(f"Loading training data from {filepath}")

        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            names=Settings.ALL_COLUMNS,
            engine="python",
        )
        logger.info(
            f"Training data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns"
        )
        return df

    def load_test(self) -> pd.DataFrame:
        """Load test data (partial engine trajectories)."""
        filepath = self.data_dir / Settings.TEST_FILE
        logger.info(f"Loading test data from {filepath}")

        df = pd.read_csv(
            filepath,
            sep=r"\s+",
            header=None,
            names=Settings.ALL_COLUMNS,
            engine="python",
        )
        logger.info(f"Test data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df

    def load_rul(self) -> pd.DataFrame:
        """Load ground truth RUL values for test engines."""
        filepath = self.data_dir / Settings.RUL_FILE
        logger.info(f"Loading RUL labels from {filepath}")

        df = pd.read_csv(filepath, sep=r"\s+", header=None, names=["rul"])
        df.index += 1  # Engine IDs are 1-indexed
        df.index.name = "engine_id"
        logger.info(f"RUL labels loaded: {len(df)} engines")
        return df

    def load_all(self) -> tuple:
        """Load all dataset files at once."""
        train_df = self.load_train()
        test_df = self.load_test()
        rul_df = self.load_rul()
        return train_df, test_df, rul_df

    def validate_data(self, df: pd.DataFrame, name: str = "dataset") -> bool:
        """Run basic validation checks on loaded data."""
        checks_passed = True

        if df.empty:
            logger.error(f"[{name}] DataFrame is empty!")
            checks_passed = False

        null_count = df.isnull().sum().sum()
        if null_count > 0:
            logger.warning(f"[{name}] Found {null_count} null values")

        n_engines = df["engine_id"].nunique() if "engine_id" in df.columns else "N/A"
        logger.info(f"[{name}] Engines: {n_engines} | Shape: {df.shape}")

        return checks_passed


if __name__ == "__main__":
    logger.info("Starting data loading pipeline...")

    loader = DataLoader()

    train_df, test_df, rul_df = loader.load_all()

    loader.validate_data(train_df, "train")
    loader.validate_data(test_df, "test")

    print("\n========== DATASET SUMMARY ==========")
    print(f"Train Shape: {train_df.shape}")
    print(f"Test Shape: {test_df.shape}")
    print(f"RUL Shape: {rul_df.shape}")
    print("=====================================\n")

    print(train_df.head())
