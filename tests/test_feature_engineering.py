"""Unit tests for FeatureEngineer."""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.feature_engineering import FeatureEngineer
from configs.settings import Settings


class TestFeatureEngineer:
    def setup_method(self):
        self.loader = DataLoader()
        self.engineer = FeatureEngineer()
        self.train_df = self.loader.load_train()

    def test_add_rul_labels(self):
        df = self.engineer.add_rul_labels(self.train_df.copy())
        assert "rul" in df.columns
        assert df["rul"].min() >= 0
        assert df["rul"].max() <= Settings.MAX_RUL

    def test_normalize_sensors(self):
        df = self.engineer.normalize_sensors(self.train_df.copy(), fit=True)
        for col in self.engineer.feature_columns:
            if col in df.columns:
                assert df[col].min() >= -0.01  # Allow small float errors
                assert df[col].max() <= 1.01

    def test_drop_low_variance(self):
        df = self.engineer.drop_low_variance_sensors(self.train_df.copy())
        for sensor in Settings.DROP_SENSORS:
            assert sensor not in df.columns
