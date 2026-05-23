"""
Preprocessing package — Data loading, feature engineering, and pipeline orchestration.

Reusable preprocessing functions are available at the package level:

    >>> from src.preprocessing import compute_rul, compute_rolling_statistics
    >>> from src.preprocessing import FeatureEngineer, DataPipeline
"""

from src.preprocessing.data_loader import DataLoader
from src.preprocessing.feature_engineering import (
    FeatureEngineer,
    # Reusable standalone functions
    compute_rul,
    compute_rolling_statistics,
    compute_lag_features,
    compute_health_degradation_indicators,
    compute_sensor_trend_slopes,
    compute_exponential_moving_averages,
    normalize_by_operating_condition,
    compute_engine_grouped_features,
    compute_cycle_degradation_features,
    drop_low_variance_sensors,
    normalize_sensors,
)
from src.preprocessing.data_pipeline import DataPipeline

__all__ = [
    # Classes
    "DataLoader",
    "FeatureEngineer",
    "DataPipeline",
    # Reusable functions
    "compute_rul",
    "compute_rolling_statistics",
    "compute_lag_features",
    "compute_health_degradation_indicators",
    "compute_sensor_trend_slopes",
    "compute_exponential_moving_averages",
    "normalize_by_operating_condition",
    "compute_engine_grouped_features",
    "compute_cycle_degradation_features",
    "drop_low_variance_sensors",
    "normalize_sensors",
]
