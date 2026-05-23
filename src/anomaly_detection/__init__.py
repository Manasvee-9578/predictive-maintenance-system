"""
Anomaly Detection package — Isolation Forest training, real-time
prediction, health scoring, and industrial monitoring visualization.

Core classes:
    - ``AnomalyTrainer``   — train & export anomaly model
    - ``AnomalyPredictor`` — batch & real-time inference
    - ``StatisticalDetector`` — Z-Score & IQR baselines
    - ``MLAnomalyDetector``  — legacy Isolation Forest + SVM
    - ``DetectorPipeline``   — orchestrate all detectors

Reusable utilities:
    - ``anomaly_utils``    — scoring, severity, health, visualization
"""

from src.anomaly_detection.statistical import StatisticalDetector
from src.anomaly_detection.ml_detector import MLAnomalyDetector
from src.anomaly_detection.detector_pipeline import DetectorPipeline
from src.anomaly_detection.train_anomaly_model import AnomalyTrainer
from src.anomaly_detection.predict_anomalies import AnomalyPredictor
from src.anomaly_detection.anomaly_utils import (
    normalize_anomaly_scores,
    classify_severity,
    compute_health_score,
    get_sensor_feature_columns,
)

__all__ = [
    "StatisticalDetector",
    "MLAnomalyDetector",
    "DetectorPipeline",
    "AnomalyTrainer",
    "AnomalyPredictor",
    "normalize_anomaly_scores",
    "classify_severity",
    "compute_health_score",
    "get_sensor_feature_columns",
]
