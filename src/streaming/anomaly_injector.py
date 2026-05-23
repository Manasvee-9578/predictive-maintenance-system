"""
╔══════════════════════════════════════════════════════════════╗
║   Anomaly Injector — Realistic Industrial Fault Simulation   ║
╚══════════════════════════════════════════════════════════════╝

Injects configurable anomalies into live sensor readings to
simulate real-world industrial failure modes: temperature spikes,
vibration anomalies, pressure drops, and degradation acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple
import numpy as np

from src.streaming.stream_utils import StreamConfig


class AnomalyType(Enum):
    """Industrial failure modes."""

    TEMPERATURE_SPIKE = "temperature_spike"
    VIBRATION_ANOMALY = "vibration_anomaly"
    PRESSURE_DROP = "pressure_drop"
    EFFICIENCY_LOSS = "efficiency_loss"
    DEGRADATION_BURST = "degradation_burst"


# Which sensors are affected by each anomaly type
ANOMALY_SENSOR_MAP: Dict[AnomalyType, list] = {
    AnomalyType.TEMPERATURE_SPIKE: ["sensor_2", "sensor_3", "sensor_4"],
    AnomalyType.VIBRATION_ANOMALY: ["sensor_8", "sensor_9", "sensor_13", "sensor_14"],
    AnomalyType.PRESSURE_DROP: ["sensor_7", "sensor_11"],
    AnomalyType.EFFICIENCY_LOSS: ["sensor_12", "sensor_15", "sensor_17"],
    AnomalyType.DEGRADATION_BURST: ["sensor_20", "sensor_21"],
}

# Severity multipliers
SEVERITY_MULTIPLIER = {
    "low": 1.5,
    "medium": 3.0,
    "high": 5.0,
    "critical": 8.0,
}


@dataclass
class AnomalyEvent:
    """Record of an injected anomaly."""

    engine_id: int
    cycle: int
    anomaly_type: AnomalyType
    severity: str
    affected_sensors: list
    magnitude: float
    description: str


class AnomalyInjector:
    """Injects realistic anomalies into live sensor readings.

    Usage::

        injector = AnomalyInjector()
        reading, event = injector.maybe_inject(reading, cycle=42, engine_id=1)
    """

    def __init__(self, config: Optional[StreamConfig] = None) -> None:
        self.config = config or StreamConfig()
        self._rng = np.random.default_rng(seed=None)  # non-deterministic

        # Track recent anomalies to avoid consecutive injection
        self._cooldown: Dict[int, int] = {}  # engine_id → last_anomaly_cycle
        self._cooldown_min = 5  # minimum cycles between anomalies

    def maybe_inject(
        self,
        reading: Dict,
        cycle: int,
        engine_id: int,
        rul: float = 125.0,
    ) -> Tuple[Dict, Optional[AnomalyEvent]]:
        """Potentially inject an anomaly into a sensor reading.

        The probability of injection increases as RUL decreases
        (more failures near end-of-life).

        Args:
            reading:    Mutable dict of sensor values
            cycle:      Current cycle number
            engine_id:  Engine identifier
            rul:        Current estimated RUL

        Returns:
            (modified_reading, AnomalyEvent or None)
        """
        # Cooldown check
        last = self._cooldown.get(engine_id, -100)
        if cycle - last < self._cooldown_min:
            return reading, None

        # Adaptive probability: higher near end-of-life
        base_prob = self.config.anomaly_probability
        rul_factor = max(0.5, 1.0 + (1.0 - rul / self.config.max_rul) * 2.0)
        prob = min(base_prob * rul_factor, 0.35)

        if self._rng.random() > prob:
            return reading, None

        # ── Select anomaly type ──────────────────────────────
        anomaly_type = self._rng.choice(list(AnomalyType))
        affected_sensors = ANOMALY_SENSOR_MAP[anomaly_type]

        # ── Select severity ──────────────────────────────────
        if rul < 20:
            severity = self._rng.choice(
                ["medium", "high", "critical"], p=[0.3, 0.4, 0.3]
            )
        elif rul < 50:
            severity = self._rng.choice(["low", "medium", "high"], p=[0.3, 0.5, 0.2])
        else:
            severity = self._rng.choice(["low", "medium"], p=[0.7, 0.3])

        multiplier = SEVERITY_MULTIPLIER[severity]

        # ── Apply perturbation ───────────────────────────────
        modified = reading.copy()
        actual_affected = []

        for sensor in affected_sensors:
            if sensor not in modified:
                continue

            base_val = modified[sensor]
            noise_std = abs(base_val) * 0.01 * multiplier

            if anomaly_type == AnomalyType.TEMPERATURE_SPIKE:
                delta = abs(self._rng.normal(0, noise_std)) * multiplier
                modified[sensor] = base_val + delta
            elif anomaly_type == AnomalyType.PRESSURE_DROP:
                delta = abs(self._rng.normal(0, noise_std)) * multiplier * 0.7
                modified[sensor] = base_val - delta
            elif anomaly_type == AnomalyType.VIBRATION_ANOMALY:
                delta = self._rng.normal(0, noise_std) * multiplier * 1.2
                modified[sensor] = base_val + delta
            elif anomaly_type == AnomalyType.EFFICIENCY_LOSS:
                delta = abs(self._rng.normal(0, noise_std)) * multiplier * 0.5
                modified[sensor] = base_val - delta
            else:  # DEGRADATION_BURST
                delta = abs(self._rng.normal(0, noise_std)) * multiplier * 0.8
                modified[sensor] = base_val + delta

            actual_affected.append(sensor)

        if not actual_affected:
            return reading, None

        magnitude = float(multiplier * self._rng.uniform(0.5, 1.5))

        # Record cooldown
        self._cooldown[engine_id] = cycle

        # Build description
        descriptions = {
            AnomalyType.TEMPERATURE_SPIKE: f"🌡️ Temperature spike detected — {severity.upper()} severity on {', '.join(actual_affected)}",
            AnomalyType.VIBRATION_ANOMALY: f"📳 Vibration anomaly — {severity.upper()} severity affecting fan/core speed",
            AnomalyType.PRESSURE_DROP: f"⬇️ Pressure fluctuation — {severity.upper()} drop in HPC outlet",
            AnomalyType.EFFICIENCY_LOSS: f"⚡ Efficiency degradation — {severity.upper()} loss in fuel/bypass ratio",
            AnomalyType.DEGRADATION_BURST: f"🔥 Degradation burst — {severity.upper()} coolant bleed anomaly detected",
        }

        event = AnomalyEvent(
            engine_id=engine_id,
            cycle=cycle,
            anomaly_type=anomaly_type,
            severity=severity,
            affected_sensors=actual_affected,
            magnitude=magnitude,
            description=descriptions[anomaly_type],
        )

        return modified, event

    def add_sensor_noise(self, reading: Dict) -> Dict:
        """Add subtle Gaussian noise to all sensors for realism."""
        modified = reading.copy()
        for sensor in ANOMALY_SENSOR_MAP.get(AnomalyType.TEMPERATURE_SPIKE, []):
            if sensor in modified:
                val = modified[sensor]
                modified[sensor] = val + self._rng.normal(
                    0, abs(val) * self.config.noise_level
                )
        return modified
