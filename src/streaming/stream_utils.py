"""
╔══════════════════════════════════════════════════════════════╗
║   Stream Utilities — Shared helpers for real-time pipeline   ║
╚══════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# ── Streaming configuration ─────────────────────────────────


@dataclass
class StreamConfig:
    """Tunable parameters for the real-time simulation."""

    # Replay speed: seconds between streamed cycles
    tick_interval: float = 1.0

    # How many past cycles to keep in the rolling window
    window_size: int = 80

    # Anomaly injection probability per cycle (0.0 – 1.0)
    anomaly_probability: float = 0.06

    # Sensor noise standard deviation (fraction of signal range)
    noise_level: float = 0.005

    # Number of engines to monitor simultaneously
    n_engines: int = 4

    # Maximum RUL cap (matches project Settings.MAX_RUL)
    max_rul: int = 125


# ── Sensor metadata ─────────────────────────────────────────

SENSOR_COLUMNS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]

SENSOR_LABELS = {
    "sensor_2": "Fan Inlet Temp",
    "sensor_3": "LPC Outlet Temp",
    "sensor_4": "HPC Outlet Temp",
    "sensor_7": "HPC Outlet Pres",
    "sensor_8": "Phys Fan Speed",
    "sensor_9": "Phys Core Speed",
    "sensor_11": "HPC Static Pres",
    "sensor_12": "Fuel/Air Ratio",
    "sensor_13": "Corr Fan Speed",
    "sensor_14": "Corr Core Speed",
    "sensor_15": "Bypass Ratio",
    "sensor_17": "Bleed Enthalpy",
    "sensor_20": "HPT Coolant",
    "sensor_21": "LPT Coolant",
}


# ── Health & risk functions (lightweight, no heavy imports) ──


def compute_health_score(anomaly_scores: np.ndarray) -> float:
    """Compute health score (0-100) from recent anomaly scores."""
    if len(anomaly_scores) == 0:
        return 100.0
    avg = float(np.mean(anomaly_scores[-20:]))
    return round(max(0.0, min(100.0, (1.0 - avg) * 100)), 1)


def compute_failure_risk(
    rul: float,
    anomaly_score: float,
    health: float,
    degradation: float,
    max_rul: int = 125,
) -> float:
    """Quick failure risk estimate (0-100%)."""
    rul_risk = max(0.0, 1.0 - rul / max_rul) * 100
    anomaly_risk = min(anomaly_score, 1.0) * 100
    health_risk = max(0.0, 100.0 - health)
    deg_risk = min(degradation / 0.05, 1.0) * 100

    risk = rul_risk * 0.35 + anomaly_risk * 0.25 + health_risk * 0.25 + deg_risk * 0.15
    return round(float(np.clip(risk, 0, 100)), 1)


def classify_priority(risk: float) -> str:
    if risk >= 70:
        return "Critical"
    if risk >= 50:
        return "High"
    if risk >= 30:
        return "Medium"
    return "Low"


def recommend_action(priority: str) -> str:
    return {
        "Critical": "Immediate Inspection Required",
        "High": "Replace Component Soon",
        "Medium": "Schedule Maintenance",
        "Low": "Continue Monitoring",
    }.get(priority, "Continue Monitoring")


def compute_urgency(risk: float) -> float:
    """Map risk (0-100) to urgency (0-10)."""
    return round(risk / 10.0, 2)


# ── Rolling statistics ──────────────────────────────────────


def rolling_degradation_rate(anomaly_scores: np.ndarray, window: int = 10) -> float:
    """Estimate degradation rate from recent anomaly score trend."""
    if len(anomaly_scores) < 3:
        return 0.0
    recent = anomaly_scores[-min(window, len(anomaly_scores)) :]
    if len(recent) < 2:
        return 0.0
    diffs = np.diff(recent)
    return float(np.clip(np.mean(diffs), 0, 0.1))
