"""Unit tests for model evaluator."""

import pytest
import numpy as np
from src.rul_prediction.model_evaluator import ModelEvaluator


class TestModelEvaluator:
    def setup_method(self):
        self.evaluator = ModelEvaluator()
        self.y_true = np.array([100, 80, 60, 40, 20, 10, 5])
        self.y_pred = np.array([95, 85, 55, 42, 18, 12, 3])

    def test_rmse(self):
        rmse = self.evaluator.rmse(self.y_true, self.y_pred)
        assert rmse >= 0
        assert isinstance(rmse, float)

    def test_mae(self):
        mae = self.evaluator.mae(self.y_true, self.y_pred)
        assert mae >= 0

    def test_r2(self):
        r2 = self.evaluator.r2(self.y_true, self.y_pred)
        assert r2 <= 1.0

    def test_scoring_function(self):
        score = self.evaluator.scoring_function(self.y_true, self.y_pred)
        assert isinstance(score, float)

    def test_perfect_prediction(self):
        rmse = self.evaluator.rmse(self.y_true, self.y_true)
        assert rmse == 0.0
