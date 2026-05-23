"""
Sliding-window sequence preparation for Remaining Useful Life forecasting.

This module converts processed NASA C-MAPSS tabular data into 3-D tensors
that can be consumed by LSTM models:

    (samples, sequence_length, features)

Windows are built independently per engine so that one engine trajectory never
bleeds into another. Training and validation splits are also done at engine
level, which is important for realistic model evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from configs.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

INDEX_COLUMNS = {"engine_id", "cycle"}
TARGET_COLUMNS = {
    "rul",
    "remaining_life_frac",
    "late_life_flag",
    "health_index",
}


def load_processed_csv(path: Path) -> pd.DataFrame:
    """Load a processed CSV with clear error reporting."""
    try:
        if not path.exists():
            raise FileNotFoundError(f"Processed data file not found: {path}")
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Processed data file is empty: {path}")
        return df
    except Exception as exc:
        logger.exception(f"Failed to load processed data from {path}")
        raise exc


def get_feature_columns(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> list[str]:
    """
    Return numeric model features from a processed C-MAPSS DataFrame.

    The preprocessing output contains engineered features that are useful for
    forecasting, but it also contains target-derived helper columns. Those are
    excluded to avoid label leakage.
    """
    if feature_columns is not None:
        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing[:10]}")
        return list(feature_columns)

    excluded = INDEX_COLUMNS | TARGET_COLUMNS
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [
        col
        for col in numeric_cols
        if col not in excluded and not col.lower().endswith("_rul")
    ]

    if not features:
        raise ValueError("No numeric feature columns were found for RUL training.")

    return features


def _clean_feature_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Keep LSTM inputs finite and deterministic."""
    features = df[feature_columns].copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.fillna(0.0)


def build_sequences_for_engine(
    features: np.ndarray,
    labels: np.ndarray | None,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Build sliding-window sequences for a single engine.

    Engines shorter than the requested window are left-padded with their first
    observed row. The label for each window is the RUL at the final cycle in
    that window.
    """
    X, y, _ = build_sequences_with_cycles(
        features=features,
        labels=labels,
        cycles=None,
        sequence_length=sequence_length,
        last_only=False,
    )
    return X, y


def build_sequences_with_cycles(
    features: np.ndarray,
    labels: np.ndarray | None,
    cycles: np.ndarray | None,
    sequence_length: int,
    last_only: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Build windows and optionally return the cycle aligned to each window."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer.")
    if len(features) == 0:
        raise ValueError("Cannot build sequences for an empty engine trajectory.")

    features = np.asarray(features, dtype=np.float32)
    labels_array = None if labels is None else np.asarray(labels, dtype=np.float32)
    cycles_array = None if cycles is None else np.asarray(cycles)

    if len(features) < sequence_length:
        pad_count = sequence_length - len(features)
        feature_pad = np.repeat(features[:1], pad_count, axis=0)
        features = np.vstack([feature_pad, features])

        if labels_array is not None:
            label_pad = np.repeat(labels_array[:1], pad_count)
            labels_array = np.concatenate([label_pad, labels_array])

        if cycles_array is not None:
            cycle_pad = np.repeat(cycles_array[:1], pad_count)
            cycles_array = np.concatenate([cycle_pad, cycles_array])

    start_indices = range(len(features) - sequence_length + 1)
    if last_only:
        start_indices = [len(features) - sequence_length]

    X = []
    y = [] if labels_array is not None else None
    out_cycles = [] if cycles_array is not None else None

    for start in start_indices:
        end = start + sequence_length
        X.append(features[start:end])
        if y is not None:
            y.append(labels_array[end - 1])
        if out_cycles is not None:
            out_cycles.append(cycles_array[end - 1])

    X_array = np.asarray(X, dtype=np.float32)
    y_array = None if y is None else np.asarray(y, dtype=np.float32)
    cycles_out = None if out_cycles is None else np.asarray(out_cycles)
    return X_array, y_array, cycles_out


class SequencePreparer:
    """Reusable sequence builder for RUL training, evaluation, and inference."""

    def __init__(
        self,
        sequence_length: int | None = None,
        val_split: float = 0.2,
        random_state: int = 42,
        feature_columns: list[str] | None = None,
    ) -> None:
        self.sequence_length = int(sequence_length or Settings.SEQUENCE_LENGTH)
        self.val_split = val_split
        self.random_state = random_state
        self.feature_columns = list(feature_columns or [])
        self.n_features = len(self.feature_columns)
        self.train_engine_ids: np.ndarray | None = None
        self.val_engine_ids: np.ndarray | None = None

    def prepare_train_sequences(
        self,
        train_path: Path | None = None,
        df: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create engine-level train and validation sequences."""
        df = self._load_or_use_df(
            df=df,
            path=train_path
            or Settings.OUTPUT_DIR / "processed" / "train_processed.csv",
            name="training",
        )
        self._validate_required_columns(df, require_rul=True)
        self._set_features(df)

        engine_ids = np.asarray(sorted(df["engine_id"].unique()))
        if len(engine_ids) < 2:
            raise ValueError(
                "At least two engines are required for train/validation split."
            )

        train_ids, val_ids = train_test_split(
            engine_ids,
            test_size=self.val_split,
            random_state=self.random_state,
            shuffle=True,
        )
        self.train_engine_ids = np.asarray(sorted(train_ids))
        self.val_engine_ids = np.asarray(sorted(val_ids))

        X_train, y_train = self._build_split(
            df, self.train_engine_ids, require_labels=True
        )
        X_val, y_val = self._build_split(df, self.val_engine_ids, require_labels=True)

        logger.info(
            f"Prepared RUL sequences: train={X_train.shape}, val={X_val.shape}, "
            f"features={self.n_features}, window={self.sequence_length}"
        )
        return X_train, X_val, y_train, y_val

    def prepare_test_sequences(
        self,
        test_path: Path | None = None,
        rul_path: Path | None = None,
        df_test: pd.DataFrame | None = None,
        df_rul: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create one last-window sequence per test engine."""
        df_test = self._load_or_use_df(
            df=df_test,
            path=test_path or Settings.OUTPUT_DIR / "processed" / "test_processed.csv",
            name="test",
        )
        self._validate_required_columns(df_test, require_rul=False)
        self._set_features(df_test)

        if df_rul is None:
            label_path = (
                rul_path or Settings.OUTPUT_DIR / "processed" / "rul_processed.csv"
            )
            df_rul = load_processed_csv(label_path)

        rul_map = self._build_rul_map(df_rul, sorted(df_test["engine_id"].unique()))
        X_list, y_list, id_list = [], [], []

        for engine_id in sorted(df_test["engine_id"].unique()):
            engine_df = df_test[df_test["engine_id"] == engine_id].sort_values("cycle")
            features = _clean_feature_frame(engine_df, self.feature_columns).to_numpy()
            X, _, _ = build_sequences_with_cycles(
                features=features,
                labels=None,
                cycles=None,
                sequence_length=self.sequence_length,
                last_only=True,
            )
            X_list.append(X[0])
            y_list.append(float(rul_map.get(engine_id, np.nan)))
            id_list.append(engine_id)

        X_test = np.asarray(X_list, dtype=np.float32)
        y_test = np.asarray(y_list, dtype=np.float32)
        engine_ids = np.asarray(id_list)
        logger.info(f"Prepared test RUL sequences: {X_test.shape}")
        return X_test, y_test, engine_ids

    def prepare_full_trajectories(
        self,
        df: pd.DataFrame,
    ) -> dict[int, tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
        """Create all sliding windows for each engine for trend visualizations."""
        self._validate_required_columns(df, require_rul=False)
        self._set_features(df)
        has_rul = "rul" in df.columns
        trajectories: dict[int, tuple[np.ndarray, np.ndarray | None, np.ndarray]] = {}

        for engine_id in sorted(df["engine_id"].unique()):
            engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")
            features = _clean_feature_frame(engine_df, self.feature_columns).to_numpy()
            labels = engine_df["rul"].to_numpy() if has_rul else None
            cycles = engine_df["cycle"].to_numpy()

            X, y, aligned_cycles = build_sequences_with_cycles(
                features=features,
                labels=labels,
                cycles=cycles,
                sequence_length=self.sequence_length,
                last_only=False,
            )
            trajectories[int(engine_id)] = (X, y, aligned_cycles)

        return trajectories

    def summary(self) -> dict:
        """Return sequence preparation metadata for model persistence."""
        return {
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "feature_columns": self.feature_columns,
            "val_split": self.val_split,
            "train_engine_ids": (
                None
                if self.train_engine_ids is None
                else self.train_engine_ids.tolist()
            ),
            "val_engine_ids": (
                None if self.val_engine_ids is None else self.val_engine_ids.tolist()
            ),
        }

    def _load_or_use_df(
        self, df: pd.DataFrame | None, path: Path, name: str
    ) -> pd.DataFrame:
        if df is not None:
            return df.copy()
        logger.info(f"Loading {name} data from {path}")
        return load_processed_csv(path)

    def _validate_required_columns(self, df: pd.DataFrame, require_rul: bool) -> None:
        required = ["engine_id", "cycle"]
        if require_rul:
            required.append("rul")
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _set_features(self, df: pd.DataFrame) -> None:
        self.feature_columns = get_feature_columns(
            df,
            self.feature_columns if self.feature_columns else None,
        )
        self.n_features = len(self.feature_columns)

    def _build_split(
        self,
        df: pd.DataFrame,
        engine_ids: np.ndarray,
        require_labels: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_parts, y_parts = [], []

        for engine_id in engine_ids:
            engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")
            features = _clean_feature_frame(engine_df, self.feature_columns).to_numpy()
            labels = engine_df["rul"].to_numpy() if require_labels else None
            X_engine, y_engine = build_sequences_for_engine(
                features=features,
                labels=labels,
                sequence_length=self.sequence_length,
            )
            X_parts.append(X_engine)
            if y_engine is not None:
                y_parts.append(y_engine)

        if not X_parts:
            raise ValueError("No engine sequences were generated.")

        X = np.concatenate(X_parts, axis=0).astype(np.float32)
        y = np.concatenate(y_parts, axis=0).astype(np.float32)
        return X, y

    @staticmethod
    def _build_rul_map(df_rul: pd.DataFrame, engine_ids: list[int]) -> dict[int, float]:
        if "rul" not in df_rul.columns:
            raise ValueError("RUL label file must contain a 'rul' column.")

        if "engine_id" in df_rul.columns:
            return {
                int(row["engine_id"]): float(row["rul"]) for _, row in df_rul.iterrows()
            }

        rul_values = df_rul["rul"].to_numpy(dtype=np.float32)
        return {
            int(engine_id): float(rul_values[index])
            for index, engine_id in enumerate(engine_ids)
            if index < len(rul_values)
        }


if __name__ == "__main__":
    Settings.ensure_directories()
    preparer = SequencePreparer()
    X_train, X_val, y_train, y_val = preparer.prepare_train_sequences()
    X_test, y_test, engine_ids = preparer.prepare_test_sequences()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Features: {preparer.n_features}, Engines: {len(engine_ids)}")
