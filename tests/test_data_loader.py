"""Unit tests for DataLoader."""

import pytest
import pandas as pd
from src.preprocessing.data_loader import DataLoader
from configs.settings import Settings


class TestDataLoader:
    def setup_method(self):
        self.loader = DataLoader()

    def test_load_train_returns_dataframe(self):
        df = self.loader.load_train()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_train_has_expected_columns(self):
        df = self.loader.load_train()
        assert "engine_id" in df.columns
        assert "cycle" in df.columns
        assert len(df.columns) == len(Settings.ALL_COLUMNS)

    def test_load_test_returns_dataframe(self):
        df = self.loader.load_test()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_load_rul_returns_dataframe(self):
        rul = self.loader.load_rul()
        assert isinstance(rul, pd.DataFrame)
        assert "rul" in rul.columns

    def test_validate_data_passes(self):
        df = self.loader.load_train()
        assert self.loader.validate_data(df, "train") is True
