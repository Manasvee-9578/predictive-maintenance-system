"""
╔══════════════════════════════════════════════════════════════╗
║   Validators — Input Validation Utilities                   ║
╚══════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def validate_dataframe(
    df: pd.DataFrame, required_columns: list = None, name: str = "DataFrame"
) -> bool:
    """
    Validate a DataFrame meets minimum requirements.

    Args:
        df: DataFrame to validate
        required_columns: List of columns that must exist
        name: Name for logging context

    Returns:
        True if validation passes
    """
    if df is None or df.empty:
        logger.error(f"[{name}] DataFrame is None or empty")
        return False

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            logger.error(f"[{name}] Missing required columns: {missing}")
            return False

    # Check for excessive nulls
    null_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    if null_pct > 50:
        logger.warning(f"[{name}] {null_pct:.1f}% null values detected")

    logger.info(
        f"[{name}] Validation passed ✅ ({df.shape[0]:,} rows × {df.shape[1]} cols)"
    )
    return True


def validate_file_exists(filepath: str | Path) -> bool:
    """Check that a file exists and is not empty."""
    path = Path(filepath)
    if not path.exists():
        logger.error(f"File not found: {path}")
        return False
    if path.stat().st_size == 0:
        logger.error(f"File is empty: {path}")
        return False
    return True


def validate_predictions(
    y_pred: np.ndarray, min_val: float = 0, max_val: float = 200
) -> bool:
    """Validate that predictions are within expected bounds."""
    if y_pred is None or len(y_pred) == 0:
        logger.error("Predictions array is empty")
        return False

    out_of_range = ((y_pred < min_val) | (y_pred > max_val)).sum()
    if out_of_range > 0:
        logger.warning(
            f"{out_of_range} predictions out of range [{min_val}, {max_val}]"
        )

    if np.isnan(y_pred).any():
        logger.error("Predictions contain NaN values")
        return False

    return True
