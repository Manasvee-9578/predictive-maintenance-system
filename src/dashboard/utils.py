"""
╔══════════════════════════════════════════════════════════════╗
║   Dashboard Utilities — Data Loading & Helper Functions      ║
╚══════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import streamlit as st
import pandas as pd
import numpy as np

# ── Project paths ────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"


# ── Cached data loaders ─────────────────────────────────────


@st.cache_data(ttl=300)
def load_recommendations() -> pd.DataFrame:
    path = PREDICTIONS_DIR / "maintenance_recommendations.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_risk_summary() -> pd.DataFrame:
    path = PREDICTIONS_DIR / "risk_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_health_scores() -> pd.DataFrame:
    path = PREDICTIONS_DIR / "health_scores_prediction.csv"
    if not path.exists():
        path = PREDICTIONS_DIR / "health_scores.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_rul_predictions() -> pd.DataFrame:
    path = PREDICTIONS_DIR / "rul_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_rul_trends() -> pd.DataFrame:
    path = PREDICTIONS_DIR / "rul_training_trends.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_anomaly_results() -> pd.DataFrame:
    path = PREDICTIONS_DIR / "anomaly_results.csv"
    if path.exists():
        df = pd.read_csv(
            path,
            usecols=lambda c: c
            in [
                "engine_id",
                "cycle",
                "rul",
                "anomaly_score_norm",
                "anomaly_pred",
                "is_anomaly",
                "severity",
                "health_index",
                "degradation_rate",
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
            ],
        )
        return df
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_training_history() -> pd.DataFrame:
    path = PREDICTIONS_DIR / "rul_training_history.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_training_metrics() -> pd.DataFrame:
    path = PREDICTIONS_DIR / "rul_training_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_processed_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "train_processed.csv"
    if path.exists():
        df = pd.read_csv(
            path,
            usecols=lambda c: c
            in [
                "engine_id",
                "cycle",
                "rul",
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
                "op_setting_1",
                "op_setting_2",
                "op_setting_3",
            ],
        )
        return df
    return pd.DataFrame()


# ── Sensor columns ───────────────────────────────────────────

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
    "sensor_2": "Total Temperature (Fan Inlet)",
    "sensor_3": "Total Temperature (LPC Outlet)",
    "sensor_4": "Total Temperature (HPC Outlet)",
    "sensor_7": "Total Pressure (HPC Outlet)",
    "sensor_8": "Physical Fan Speed",
    "sensor_9": "Physical Core Speed",
    "sensor_11": "Static Pressure (HPC Outlet)",
    "sensor_12": "Fuel/Air Ratio",
    "sensor_13": "Corrected Fan Speed",
    "sensor_14": "Corrected Core Speed",
    "sensor_15": "Bypass Ratio",
    "sensor_17": "Bleed Enthalpy",
    "sensor_20": "HPT Coolant Bleed",
    "sensor_21": "LPT Coolant Bleed",
}


# ── Helper functions ─────────────────────────────────────────


def fmt_number(n: float, decimals: int = 1) -> str:
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.{decimals}f}M"
    if abs(n) >= 1_000:
        return f"{n/1_000:.{decimals}f}K"
    return f"{n:.{decimals}f}"


def get_priority_color(priority: str) -> str:
    colors = {
        "Low": "#10b981",
        "Medium": "#f59e0b",
        "High": "#f97316",
        "Critical": "#ef4444",
    }
    return colors.get(priority, "#94a3b8")


def get_health_color(status: str) -> str:
    colors = {"Normal": "#10b981", "Warning": "#f59e0b", "Critical": "#ef4444"}
    return colors.get(status, "#94a3b8")


def get_risk_color(risk_pct: float) -> str:
    if risk_pct >= 75:
        return "#ef4444"
    if risk_pct >= 50:
        return "#f97316"
    if risk_pct >= 25:
        return "#f59e0b"
    return "#10b981"
