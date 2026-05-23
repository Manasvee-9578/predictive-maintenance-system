"""
╔══════════════════════════════════════════════════════════════╗
║   Helpers — Miscellaneous Utility Functions                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import time
import functools
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def timer(func):
    """Decorator to measure and log function execution time."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        if elapsed < 60:
            logger.info(f"⏱ {func.__name__} completed in {elapsed:.2f}s")
        else:
            minutes, seconds = divmod(elapsed, 60)
            logger.info(
                f"⏱ {func.__name__} completed in {int(minutes)}m {seconds:.1f}s"
            )

        return result

    return wrapper


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return the path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_number(n: float, precision: int = 2) -> str:
    """Format large numbers with commas and specified precision."""
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.{precision}f}M"
    elif abs(n) >= 1_000:
        return f"{n / 1_000:.{precision}f}K"
    else:
        return f"{n:.{precision}f}"


def get_engine_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary table of engine statistics.

    Args:
        df: DataFrame with engine_id and cycle columns

    Returns:
        Summary DataFrame with stats per engine
    """
    summary = (
        df.groupby("engine_id")
        .agg(
            total_cycles=("cycle", "max"),
            n_records=("cycle", "count"),
        )
        .reset_index()
    )

    summary["avg_cycle"] = summary["total_cycles"].mean()
    return summary


def clip_predictions(
    predictions: np.ndarray, min_val: float = 0, max_val: float = None
) -> np.ndarray:
    """Clip RUL predictions to valid range."""
    from configs.settings import Settings

    max_val = max_val or Settings.MAX_RUL
    return np.clip(predictions, min_val, max_val)
