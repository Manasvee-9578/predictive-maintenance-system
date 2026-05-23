"""Remaining Useful Life prediction package.

The package exposes the main training, inference, and sequence-preparation
classes without eagerly importing TensorFlow-heavy modules.
"""

__all__ = [
    "LSTMModel",
    "ClassicalModels",
    "ModelTrainer",
    "ModelEvaluator",
    "SequencePreparer",
    "RULTrainer",
    "RULPredictor",
    "classify_risk",
    "predict_maintenance_urgency",
    "estimate_confidence",
    "compute_metrics",
]


def __getattr__(name):
    if name == "LSTMModel":
        from src.rul_prediction.lstm_model import LSTMModel

        return LSTMModel
    if name == "ClassicalModels":
        from src.rul_prediction.classical_models import ClassicalModels

        return ClassicalModels
    if name == "ModelTrainer":
        from src.rul_prediction.model_trainer import ModelTrainer

        return ModelTrainer
    if name == "ModelEvaluator":
        from src.rul_prediction.model_evaluator import ModelEvaluator

        return ModelEvaluator
    if name == "SequencePreparer":
        from src.rul_prediction.prepare_sequences import SequencePreparer

        return SequencePreparer
    if name == "RULTrainer":
        from src.rul_prediction.train_rul_model import RULTrainer

        return RULTrainer
    if name == "RULPredictor":
        from src.rul_prediction.predict_rul import RULPredictor

        return RULPredictor
    if name in {
        "classify_risk",
        "predict_maintenance_urgency",
        "estimate_confidence",
        "compute_metrics",
    }:
        from src.rul_prediction import rul_utils

        return getattr(rul_utils, name)
    raise AttributeError(f"module 'src.rul_prediction' has no attribute {name!r}")
