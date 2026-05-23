"""
╔══════════════════════════════════════════════════════════════╗
║   Plotly Chart Builders — Reusable Chart Components         ║
╚══════════════════════════════════════════════════════════════╝
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_sensor_timeline(df: pd.DataFrame, sensor: str, engine_id: int) -> go.Figure:
    """Create an interactive sensor reading timeline for a specific engine."""
    engine_data = df[df["engine_id"] == engine_id].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=engine_data["cycle"],
            y=engine_data[sensor],
            mode="lines",
            name=sensor,
            line=dict(color="#636EFA", width=2),
        )
    )

    fig.update_layout(
        title=f"{sensor} — Engine {engine_id}",
        xaxis_title="Cycle",
        yaxis_title="Sensor Reading",
        template="plotly_dark",
        height=400,
        hovermode="x unified",
    )
    return fig


def create_rul_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"
) -> go.Figure:
    """Create predicted vs actual RUL scatter plot."""
    fig = go.Figure()

    # Perfect prediction line
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color="white", dash="dash", width=1),
        )
    )

    # Actual scatter
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name=model_name,
            marker=dict(color="#636EFA", size=6, opacity=0.6),
        )
    )

    fig.update_layout(
        title=f"Predicted vs Actual RUL — {model_name}",
        xaxis_title="Actual RUL",
        yaxis_title="Predicted RUL",
        template="plotly_dark",
        height=500,
    )
    return fig


def create_health_gauge(health_pct: float, engine_id: int) -> go.Figure:
    """Create a gauge chart showing engine health percentage."""
    color = (
        "#2ecc71" if health_pct > 60 else "#f39c12" if health_pct > 30 else "#e74c3c"
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=health_pct,
            title={"text": f"Engine {engine_id} Health"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "rgba(231,76,60,0.2)"},
                    {"range": [30, 60], "color": "rgba(243,156,18,0.2)"},
                    {"range": [60, 100], "color": "rgba(46,204,113,0.2)"},
                ],
            },
            number={"suffix": "%"},
        )
    )

    fig.update_layout(template="plotly_dark", height=250)
    return fig
