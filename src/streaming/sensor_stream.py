"""
╔══════════════════════════════════════════════════════════════╗
║   Sensor Stream — Sequential CMAPSS Data Replay Engine       ║
╚══════════════════════════════════════════════════════════════╝

Reads the processed training data and yields sensor readings
one cycle at a time, simulating real-time industrial telemetry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Generator, List, Optional
import numpy as np
import pandas as pd

from src.streaming.stream_utils import SENSOR_COLUMNS, StreamConfig

# ── Project paths ────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


class SensorStream:
    """Replays CMAPSS sensor data cycle-by-cycle for selected engines.

    Usage::

        stream = SensorStream(engine_ids=[1, 24, 52])
        for cycle_data in stream:
            print(cycle_data)  # dict per engine
    """

    def __init__(
        self,
        engine_ids: Optional[List[int]] = None,
        config: Optional[StreamConfig] = None,
        data_path: Optional[Path] = None,
    ) -> None:
        self.config = config or StreamConfig()
        self._path = data_path or (_PROCESSED_DIR / "train_processed.csv")

        # Load the full dataset once
        self._df = self._load_data()

        # Determine engine IDs
        available = sorted(self._df["engine_id"].unique().tolist())
        if engine_ids:
            self.engine_ids = [e for e in engine_ids if e in available]
        else:
            self.engine_ids = available[: self.config.n_engines]

        # Pre-slice per-engine DataFrames, sorted by cycle
        self._engine_data: Dict[int, pd.DataFrame] = {}
        self._engine_cycle_idx: Dict[int, int] = {}
        for eid in self.engine_ids:
            edf = (
                self._df[self._df["engine_id"] == eid]
                .sort_values("cycle")
                .reset_index(drop=True)
            )
            self._engine_data[eid] = edf
            self._engine_cycle_idx[eid] = 0

    # ── Data loading ─────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        """Load processed CSV with only needed columns."""
        cols_to_keep = (
            ["engine_id", "cycle", "rul"]
            + SENSOR_COLUMNS
            + [
                "op_setting_1",
                "op_setting_2",
                "op_setting_3",
            ]
        )
        if self._path.exists():
            df = pd.read_csv(
                self._path,
                usecols=lambda c: c in cols_to_keep,
            )
            return df
        raise FileNotFoundError(f"Processed data not found: {self._path}")

    # ── Public API ───────────────────────────────────────────

    @property
    def total_engines(self) -> int:
        return len(self.engine_ids)

    def max_cycles(self, engine_id: int) -> int:
        return len(self._engine_data.get(engine_id, pd.DataFrame()))

    def current_cycle_index(self, engine_id: int) -> int:
        return self._engine_cycle_idx.get(engine_id, 0)

    def is_exhausted(self, engine_id: int) -> bool:
        return self.current_cycle_index(engine_id) >= self.max_cycles(engine_id)

    def all_exhausted(self) -> bool:
        return all(self.is_exhausted(eid) for eid in self.engine_ids)

    def reset(self, engine_id: Optional[int] = None) -> None:
        if engine_id:
            self._engine_cycle_idx[engine_id] = 0
        else:
            for eid in self.engine_ids:
                self._engine_cycle_idx[eid] = 0

    def next_reading(self, engine_id: int) -> Optional[Dict]:
        """Get the next sensor reading for a specific engine.

        Returns:
            Dictionary with sensor values, cycle, RUL, or None if exhausted.
        """
        edf = self._engine_data.get(engine_id)
        idx = self._engine_cycle_idx.get(engine_id, 0)
        if edf is None or idx >= len(edf):
            return None

        row = edf.iloc[idx]
        self._engine_cycle_idx[engine_id] = idx + 1

        reading = {
            "engine_id": int(engine_id),
            "cycle": int(row.get("cycle", idx + 1)),
            "rul": float(row.get("rul", 125)),
        }
        # Add sensor values
        for s in SENSOR_COLUMNS:
            if s in row.index:
                reading[s] = float(row[s])
        # Add op settings
        for op in ["op_setting_1", "op_setting_2", "op_setting_3"]:
            if op in row.index:
                reading[op] = float(row[op])

        return reading

    def next_batch(self) -> Dict[int, Optional[Dict]]:
        """Get next reading for all engines simultaneously.

        Returns:
            Dict mapping engine_id → reading dict (or None if exhausted)
        """
        batch = {}
        for eid in self.engine_ids:
            batch[eid] = self.next_reading(eid)
        return batch

    def stream(self) -> Generator[Dict[int, Optional[Dict]], None, None]:
        """Infinite generator yielding batches cycle-by-cycle.

        Engines that reach end-of-life automatically reset to cycle 1
        to simulate continuous operation.
        """
        while True:
            # Auto-reset exhausted engines
            for eid in self.engine_ids:
                if self.is_exhausted(eid):
                    self.reset(eid)
            yield self.next_batch()
