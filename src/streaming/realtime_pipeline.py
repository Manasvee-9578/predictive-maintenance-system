"""
╔══════════════════════════════════════════════════════════════╗
║   Real-Time Pipeline — Streaming Predictive Analytics        ║
╚══════════════════════════════════════════════════════════════╝

Orchestrates the streaming loop: reads sensor data cycle-by-cycle,
injects anomalies, computes health/risk/RUL scores, and produces
dashboard-ready state snapshots.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.streaming.sensor_stream import SensorStream
from src.streaming.anomaly_injector import AnomalyInjector, AnomalyEvent
from src.streaming.stream_utils import (
    StreamConfig,
    SENSOR_COLUMNS,
    compute_health_score,
    compute_failure_risk,
    classify_priority,
    recommend_action,
    compute_urgency,
    rolling_degradation_rate,
)


@dataclass
class EngineState:
    """Live state for a single engine being monitored."""

    engine_id: int
    cycle: int = 0
    rul: float = 125.0

    # Rolling sensor history (list of dicts)
    sensor_history: List[Dict] = field(default_factory=list)

    # Anomaly tracking
    anomaly_scores: List[float] = field(default_factory=list)
    anomaly_events: List[AnomalyEvent] = field(default_factory=list)
    is_anomaly: bool = False
    latest_anomaly_score: float = 0.0

    # Computed metrics
    health_score: float = 100.0
    failure_risk: float = 0.0
    degradation_rate: float = 0.0
    priority: str = "Low"
    recommended_action: str = "Continue Monitoring"
    urgency: float = 0.0

    # Status
    status: str = "Normal"  # Normal, Warning, Critical, Anomaly


@dataclass
class PipelineSnapshot:
    """Complete state snapshot at one point in time, for the dashboard."""

    tick: int
    engine_states: Dict[int, EngineState]
    new_events: List[AnomalyEvent]
    fleet_health: float
    fleet_risk: float
    total_anomalies: int


class RealtimePipeline:
    """Streaming analytics pipeline for multi-engine monitoring.

    Usage (in Streamlit session_state)::

        pipeline = RealtimePipeline(engine_ids=[1, 24, 52, 78])
        snapshot = pipeline.tick()  # advance one cycle
    """

    def __init__(
        self,
        engine_ids: Optional[List[int]] = None,
        config: Optional[StreamConfig] = None,
    ) -> None:
        self.config = config or StreamConfig()
        self._stream = SensorStream(engine_ids=engine_ids, config=self.config)
        self._injector = AnomalyInjector(config=self.config)

        self.engine_ids = self._stream.engine_ids
        self._tick_count = 0

        # Initialize per-engine state
        self._states: Dict[int, EngineState] = {
            eid: EngineState(engine_id=eid) for eid in self.engine_ids
        }

        # Global event log
        self._all_events: List[AnomalyEvent] = []

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def states(self) -> Dict[int, EngineState]:
        return self._states

    def get_state(self, engine_id: int) -> Optional[EngineState]:
        return self._states.get(engine_id)

    def get_sensor_dataframe(self, engine_id: int) -> pd.DataFrame:
        """Get sensor history as a DataFrame for charting."""
        state = self._states.get(engine_id)
        if state is None or not state.sensor_history:
            return pd.DataFrame()
        return pd.DataFrame(state.sensor_history)

    def get_events(
        self, engine_id: Optional[int] = None, last_n: int = 20
    ) -> List[AnomalyEvent]:
        """Get recent anomaly events, optionally filtered by engine."""
        events = self._all_events
        if engine_id is not None:
            events = [e for e in events if e.engine_id == engine_id]
        return events[-last_n:]

    # ── Main tick ────────────────────────────────────────────

    def tick(self) -> PipelineSnapshot:
        """Advance all engines by one cycle and return updated snapshot.

        Steps per engine:
            1. Read next sensor cycle from CMAPSS data
            2. Inject potential anomalies
            3. Add sensor noise for realism
            4. Compute anomaly score (analytical heuristic)
            5. Update health, risk, priority, action
            6. Append to rolling history

        Returns:
            PipelineSnapshot with current state of all engines.
        """
        self._tick_count += 1
        batch = self._stream.next_batch()
        new_events: List[AnomalyEvent] = []

        for eid in self.engine_ids:
            reading = batch.get(eid)
            state = self._states[eid]

            if reading is None:
                # Engine exhausted — reset for continuous streaming
                self._stream.reset(eid)
                self._states[eid] = EngineState(engine_id=eid)
                continue

            cycle = reading.get("cycle", state.cycle + 1)
            rul = reading.get("rul", 125.0)

            # ── Step 2: Anomaly injection ────────────────────
            reading, event = self._injector.maybe_inject(
                reading,
                cycle=cycle,
                engine_id=eid,
                rul=rul,
            )
            if event:
                new_events.append(event)
                self._all_events.append(event)
                state.anomaly_events.append(event)

            # ── Step 3: Sensor noise ─────────────────────────
            reading = self._injector.add_sensor_noise(reading)

            # ── Step 4: Anomaly score (analytical) ───────────
            anomaly_score = self._compute_anomaly_score(reading, state)
            state.latest_anomaly_score = anomaly_score
            state.anomaly_scores.append(anomaly_score)
            state.is_anomaly = anomaly_score > 0.5 or event is not None

            # ── Step 5: Update metrics ───────────────────────
            state.cycle = cycle
            state.rul = rul
            state.health_score = compute_health_score(np.array(state.anomaly_scores))
            state.degradation_rate = rolling_degradation_rate(
                np.array(state.anomaly_scores)
            )
            state.failure_risk = compute_failure_risk(
                rul,
                anomaly_score,
                state.health_score,
                state.degradation_rate,
                self.config.max_rul,
            )
            state.priority = classify_priority(state.failure_risk)
            state.recommended_action = recommend_action(state.priority)
            state.urgency = compute_urgency(state.failure_risk)

            # Status
            if state.priority == "Critical":
                state.status = "Critical"
            elif state.is_anomaly:
                state.status = "Anomaly"
            elif state.priority in ("High", "Medium"):
                state.status = "Warning"
            else:
                state.status = "Normal"

            # ── Step 6: Rolling history ──────────────────────
            sensor_record = {
                "cycle": cycle,
                "rul": rul,
                "anomaly_score": anomaly_score,
                "health_score": state.health_score,
                "failure_risk": state.failure_risk,
            }
            for s in SENSOR_COLUMNS:
                if s in reading:
                    sensor_record[s] = reading[s]
            state.sensor_history.append(sensor_record)

            # Trim to window size
            if len(state.sensor_history) > self.config.window_size:
                state.sensor_history = state.sensor_history[-self.config.window_size :]
            if len(state.anomaly_scores) > self.config.window_size:
                state.anomaly_scores = state.anomaly_scores[-self.config.window_size :]

        # ── Fleet-level metrics ──────────────────────────────
        active_states = [s for s in self._states.values() if s.cycle > 0]
        fleet_health = (
            float(np.mean([s.health_score for s in active_states]))
            if active_states
            else 100.0
        )
        fleet_risk = (
            float(np.mean([s.failure_risk for s in active_states]))
            if active_states
            else 0.0
        )
        total_anomalies = sum(len(s.anomaly_events) for s in active_states)

        return PipelineSnapshot(
            tick=self._tick_count,
            engine_states=dict(self._states),
            new_events=new_events,
            fleet_health=fleet_health,
            fleet_risk=fleet_risk,
            total_anomalies=total_anomalies,
        )

    # ── Anomaly scoring (analytical heuristic) ───────────────

    def _compute_anomaly_score(self, reading: Dict, state: EngineState) -> float:
        """Compute anomaly score from sensor deviation analysis.

        Uses Z-score-like deviation from rolling mean of recent readings.
        """
        if len(state.sensor_history) < 5:
            return 0.05  # insufficient history

        recent = pd.DataFrame(state.sensor_history[-30:])
        deviations = []

        for sensor in SENSOR_COLUMNS:
            if sensor not in reading or sensor not in recent.columns:
                continue
            current = reading[sensor]
            mean = recent[sensor].mean()
            std = recent[sensor].std()
            if std < 1e-10:
                continue
            z = abs(current - mean) / std
            deviations.append(min(z / 4.0, 1.0))  # normalize to 0-1

        if not deviations:
            return 0.05

        # Combine: use 80th percentile of deviations (robust to noise)
        score = float(np.percentile(deviations, 80))

        # Boost score for end-of-life proximity
        rul = reading.get("rul", 125)
        rul_factor = max(0.0, (1.0 - rul / self.config.max_rul) * 0.15)
        score = min(score + rul_factor, 1.0)

        return round(score, 4)
