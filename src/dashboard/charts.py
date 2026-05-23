"""
╔══════════════════════════════════════════════════════════════╗
║   Charts — Advanced Plotly Visualizations                    ║
╚══════════════════════════════════════════════════════════════╝

All chart builders use the dark industrial theme from theme.py.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.dashboard.theme import PLOTLY_LAYOUT, COLORS, PRIORITY_COLORS, HEALTH_COLORS


def _apply_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
        height=height,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  Health & Gauges
# ═══════════════════════════════════════════════════════════════


def health_gauge(
    value: float, title: str = "Health", max_val: float = 100
) -> go.Figure:
    if value > 80:
        bar_color, ref = "#10b981", "good"
    elif value > 50:
        bar_color, ref = "#f59e0b", "warning"
    else:
        bar_color, ref = "#ef4444", "critical"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number=dict(suffix="%", font=dict(size=32, color="#f1f5f9")),
            title=dict(text=title, font=dict(size=13, color="#94a3b8")),
            gauge=dict(
                axis=dict(range=[0, max_val], tickcolor="#475569", dtick=20),
                bar=dict(color=bar_color, thickness=0.75),
                bgcolor="rgba(30,41,59,0.5)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 40], color="rgba(239,68,68,0.08)"),
                    dict(range=[40, 70], color="rgba(245,158,11,0.08)"),
                    dict(range=[70, 100], color="rgba(16,185,129,0.08)"),
                ],
                threshold=dict(
                    line=dict(color="#f1f5f9", width=2), thickness=0.8, value=value
                ),
            ),
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        height=220,
        margin=dict(l=30, r=30, t=50, b=10),
    )
    return fig


def risk_gauge(value: float, title: str = "Failure Risk") -> go.Figure:
    if value < 25:
        bar_color = "#10b981"
    elif value < 50:
        bar_color = "#f59e0b"
    elif value < 75:
        bar_color = "#f97316"
    else:
        bar_color = "#ef4444"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number=dict(suffix="%", font=dict(size=32, color="#f1f5f9")),
            title=dict(text=title, font=dict(size=13, color="#94a3b8")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#475569"),
                bar=dict(color=bar_color, thickness=0.75),
                bgcolor="rgba(30,41,59,0.5)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 25], color="rgba(16,185,129,0.08)"),
                    dict(range=[25, 50], color="rgba(245,158,11,0.08)"),
                    dict(range=[50, 75], color="rgba(249,115,22,0.08)"),
                    dict(range=[75, 100], color="rgba(239,68,68,0.08)"),
                ],
            ),
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        height=220,
        margin=dict(l=30, r=30, t=50, b=10),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  Fleet Overview
# ═══════════════════════════════════════════════════════════════


def fleet_health_bar(health_df: pd.DataFrame) -> go.Figure:
    df = health_df.sort_values("health_score").copy()
    colors = df["health_status"].map(HEALTH_COLORS).fillna("#94a3b8")

    fig = go.Figure(
        go.Bar(
            x=df["engine_id"].astype(str),
            y=df["health_score"],
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="Engine %{x}<br>Health: %{y:.1f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=80,
        line_dash="dot",
        line_color="rgba(16,185,129,0.4)",
        annotation_text="Normal",
        annotation_position="top right",
        annotation_font_color="#64748b",
    )
    fig.add_hline(
        y=50,
        line_dash="dot",
        line_color="rgba(239,68,68,0.4)",
        annotation_text="Critical",
        annotation_position="top right",
        annotation_font_color="#64748b",
    )
    _apply_layout(fig, "Fleet Health Status", 380)
    fig.update_xaxes(title="Engine ID", type="category", tickfont=dict(size=9))
    fig.update_yaxes(title="Health Score", range=[0, 100])
    return fig


def priority_donut(rec_df: pd.DataFrame) -> go.Figure:
    counts = rec_df["maintenance_priority"].value_counts()
    order = ["Critical", "High", "Medium", "Low"]
    labels = [p for p in order if p in counts.index]
    values = [counts[p] for p in labels]
    colors = [PRIORITY_COLORS[p] for p in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(colors=colors, line=dict(color="#0a0e1a", width=2)),
            textinfo="label+value",
            textfont=dict(size=12, color="#e2e8f0"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        )
    )
    total = sum(values)
    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:10px;color:#94a3b8'>Total</span>",
        x=0.5,
        y=0.5,
        font=dict(size=20, color="#f1f5f9"),
        showarrow=False,
    )
    _apply_layout(fig, "Maintenance Priority Distribution", 350)
    return fig


def risk_distribution_histogram(rec_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=rec_df["failure_risk_pct"],
            nbinsx=20,
            marker=dict(color="#3b82f6", line=dict(width=0.5, color="#1e3a5f")),
            hovertemplate="Risk: %{x:.1f}%<br>Count: %{y}<extra></extra>",
        )
    )
    fig.add_vline(x=50, line_dash="dash", line_color="rgba(245,158,11,0.5)")
    fig.add_vline(x=75, line_dash="dash", line_color="rgba(239,68,68,0.5)")
    _apply_layout(fig, "Failure Risk Distribution", 350)
    fig.update_xaxes(title="Failure Risk (%)")
    fig.update_yaxes(title="Count")
    return fig


def action_bar_chart(rec_df: pd.DataFrame) -> go.Figure:
    counts = rec_df["recommended_action"].value_counts()
    action_colors = {
        "Continue Monitoring": "#10b981",
        "Schedule Maintenance": "#3b82f6",
        "Immediate Inspection Required": "#f97316",
        "Replace Component Soon": "#ef4444",
    }
    colors = [action_colors.get(a, "#94a3b8") for a in counts.index]

    fig = go.Figure(
        go.Bar(
            y=counts.index,
            x=counts.values,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v} ({v/len(rec_df)*100:.0f}%)" for v in counts.values],
            textposition="outside",
            textfont=dict(color="#e2e8f0"),
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    _apply_layout(fig, "Recommended Actions", 320)
    fig.update_xaxes(title="Count")
    return fig


# ═══════════════════════════════════════════════════════════════
#  RUL Forecasting
# ═══════════════════════════════════════════════════════════════


def rul_scatter(rul_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    max_val = max(rul_df["actual_rul"].max(), rul_df["predicted_rul"].max()) * 1.1
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="Perfect",
            line=dict(color="rgba(148,163,184,0.3)", dash="dash", width=1),
            showlegend=True,
        )
    )

    risk_colors = (
        rul_df["risk_category"]
        .map({"Low Risk": "#10b981", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"})
        .fillna("#94a3b8")
    )

    fig.add_trace(
        go.Scatter(
            x=rul_df["actual_rul"],
            y=rul_df["predicted_rul"],
            mode="markers",
            name="Engines",
            marker=dict(
                color=risk_colors,
                size=9,
                opacity=0.85,
                line=dict(width=1, color="rgba(0,0,0,0.3)"),
            ),
            text=rul_df["engine_id"].astype(str),
            hovertemplate="Engine %{text}<br>Actual: %{x:.0f}<br>Predicted: %{y:.0f}<extra></extra>",
        )
    )
    _apply_layout(fig, "Predicted vs Actual RUL", 450)
    fig.update_xaxes(title="Actual RUL (cycles)")
    fig.update_yaxes(title="Predicted RUL (cycles)")
    return fig


def rul_error_distribution(rul_df: pd.DataFrame) -> go.Figure:
    errors = rul_df["predicted_rul"] - rul_df["actual_rul"]
    fig = go.Figure(
        go.Histogram(
            x=errors,
            nbinsx=30,
            marker=dict(color="#8b5cf6", line=dict(width=0.5, color="#4c1d95")),
            hovertemplate="Error: %{x:.1f}<br>Count: %{y}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(148,163,184,0.4)")
    _apply_layout(fig, "Prediction Error Distribution", 350)
    fig.update_xaxes(title="Prediction Error (cycles)")
    fig.update_yaxes(title="Count")
    return fig


def engine_degradation_trend(
    trend_df: pd.DataFrame, engine_ids: list[int]
) -> go.Figure:
    fig = go.Figure()
    palette = [
        "#3b82f6",
        "#06b6d4",
        "#8b5cf6",
        "#f59e0b",
        "#ef4444",
        "#10b981",
        "#f97316",
        "#ec4899",
    ]

    for i, eid in enumerate(engine_ids):
        edf = trend_df[trend_df["engine_id"] == eid].sort_values("cycle")
        if edf.empty:
            continue
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=edf["cycle"],
                y=edf["predicted_rul"],
                mode="lines",
                name=f"Engine {eid}",
                line=dict(color=color, width=2),
                hovertemplate=f"Engine {eid}<br>Cycle: %{{x}}<br>RUL: %{{y:.1f}}<extra></extra>",
            )
        )
        if "actual_rul" in edf.columns:
            fig.add_trace(
                go.Scatter(
                    x=edf["cycle"],
                    y=edf["actual_rul"],
                    mode="lines",
                    name=f"Engine {eid} (actual)",
                    line=dict(color=color, width=1, dash="dot"),
                    opacity=0.4,
                    showlegend=False,
                )
            )

    fig.add_hrect(y0=0, y1=15, fillcolor="rgba(239,68,68,0.06)", line_width=0)
    fig.add_hrect(y0=15, y1=30, fillcolor="rgba(249,115,22,0.04)", line_width=0)
    _apply_layout(fig, "Engine Degradation Trends", 450)
    fig.update_xaxes(title="Cycle")
    fig.update_yaxes(title="Predicted RUL (cycles)")
    return fig


# ═══════════════════════════════════════════════════════════════
#  Anomaly Detection
# ═══════════════════════════════════════════════════════════════


def anomaly_timeline(anom_df: pd.DataFrame, engine_id: int) -> go.Figure:
    edf = anom_df[anom_df["engine_id"] == engine_id].sort_values("cycle")
    if edf.empty:
        fig = go.Figure()
        _apply_layout(fig, f"Engine {engine_id} — No data", 350)
        return fig

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edf["cycle"],
            y=edf["anomaly_score_norm"],
            mode="lines",
            name="Anomaly Score",
            line=dict(color="#3b82f6", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.08)",
        )
    )

    anomalies = (
        edf[edf["is_anomaly"] == 1] if "is_anomaly" in edf.columns else pd.DataFrame()
    )
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["cycle"],
                y=anomalies["anomaly_score_norm"],
                mode="markers",
                name="Anomaly",
                marker=dict(
                    color="#ef4444",
                    size=6,
                    symbol="diamond",
                    line=dict(width=1, color="rgba(239,68,68,0.5)"),
                ),
            )
        )

    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(245,158,11,0.4)")
    _apply_layout(fig, f"Engine {engine_id} — Anomaly Score Timeline", 380)
    fig.update_xaxes(title="Cycle")
    fig.update_yaxes(title="Anomaly Score", range=[0, 1.05])
    return fig


def anomaly_severity_pie(anom_df: pd.DataFrame) -> go.Figure:
    if "severity" not in anom_df.columns:
        return go.Figure()

    counts = anom_df.groupby("engine_id")["severity"].last().value_counts()
    sev_colors = {"Normal": "#10b981", "Warning": "#f59e0b", "Critical": "#ef4444"}
    labels = list(counts.index)
    colors = [sev_colors.get(s, "#94a3b8") for s in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=counts.values,
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="#0a0e1a", width=2)),
            textinfo="label+percent",
            textfont=dict(size=12, color="#e2e8f0"),
        )
    )
    _apply_layout(fig, "Anomaly Severity Distribution", 340)
    return fig


def sensor_panel(
    anom_df: pd.DataFrame, engine_id: int, sensors: list[str]
) -> go.Figure:
    edf = anom_df[anom_df["engine_id"] == engine_id].sort_values("cycle")
    rows = len(sensors) + 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=["Anomaly Score"] + sensors,
    )

    fig.add_trace(
        go.Scatter(
            x=edf["cycle"],
            y=edf.get("anomaly_score_norm", pd.Series(dtype=float)),
            mode="lines",
            name="Anomaly Score",
            line=dict(color="#ef4444", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.06)",
        ),
        row=1,
        col=1,
    )

    palette = [
        "#3b82f6",
        "#06b6d4",
        "#8b5cf6",
        "#10b981",
        "#f59e0b",
        "#ec4899",
        "#f97316",
    ]
    for i, sensor in enumerate(sensors):
        if sensor in edf.columns:
            fig.add_trace(
                go.Scatter(
                    x=edf["cycle"],
                    y=edf[sensor],
                    mode="lines",
                    name=sensor,
                    line=dict(color=palette[i % len(palette)], width=1.5),
                ),
                row=i + 2,
                col=1,
            )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(400, rows * 140),
        showlegend=False,
        title=dict(
            text=f"Engine {engine_id} — Sensor Panel",
            font=dict(size=15, color="#e2e8f0"),
        ),
    )
    fig.update_xaxes(title="Cycle", row=rows, col=1)
    for i in range(1, rows + 1):
        fig.update_yaxes(gridcolor="rgba(148,163,184,0.04)", row=i, col=1)
        fig.update_xaxes(gridcolor="rgba(148,163,184,0.04)", row=i, col=1)
    return fig


# ═══════════════════════════════════════════════════════════════
#  Sensor Analytics
# ═══════════════════════════════════════════════════════════════


def sensor_timeline(df: pd.DataFrame, engine_id: int, sensor: str) -> go.Figure:
    edf = df[df["engine_id"] == engine_id].sort_values("cycle")
    fig = go.Figure(
        go.Scatter(
            x=edf["cycle"],
            y=edf[sensor],
            mode="lines",
            line=dict(color="#06b6d4", width=2),
            fill="tozeroy",
            fillcolor="rgba(6,182,212,0.06)",
            hovertemplate="Cycle: %{x}<br>Value: %{y:.4f}<extra></extra>",
        )
    )
    _apply_layout(fig, f"{sensor} — Engine {engine_id}", 350)
    fig.update_xaxes(title="Cycle")
    fig.update_yaxes(title="Sensor Reading")
    return fig


def sensor_correlation_heatmap(df: pd.DataFrame, sensors: list[str]) -> go.Figure:
    available = [s for s in sensors if s in df.columns]
    if len(available) < 2:
        return go.Figure()

    corr = df[available].corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=available,
            y=available,
            colorscale=[[0, "#1e3a5f"], [0.5, "#0f172a"], [1, "#3b82f6"]],
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=9, color="#e2e8f0"),
            hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
        )
    )
    _apply_layout(fig, "Sensor Correlation Matrix", 500)
    return fig


def multi_sensor_comparison(
    df: pd.DataFrame, engine_id: int, sensors: list[str]
) -> go.Figure:
    edf = df[df["engine_id"] == engine_id].sort_values("cycle")
    fig = go.Figure()
    palette = [
        "#3b82f6",
        "#06b6d4",
        "#8b5cf6",
        "#10b981",
        "#f59e0b",
        "#ec4899",
        "#f97316",
    ]
    for i, s in enumerate(sensors):
        if s in edf.columns:
            fig.add_trace(
                go.Scatter(
                    x=edf["cycle"],
                    y=edf[s],
                    mode="lines",
                    name=s,
                    line=dict(color=palette[i % len(palette)], width=1.8),
                )
            )
    _apply_layout(fig, f"Engine {engine_id} — Sensor Comparison", 420)
    fig.update_xaxes(title="Cycle")
    fig.update_yaxes(title="Normalized Reading")
    return fig


# ═══════════════════════════════════════════════════════════════
#  Business Impact
# ═══════════════════════════════════════════════════════════════


def business_kpi_indicators(impact: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=4, specs=[[{"type": "indicator"}] * 4])
    items = [
        (
            impact.get("estimated_cost_savings_usd", 0),
            "Cost Savings",
            "$",
            "",
            "#10b981",
        ),
        (
            impact.get("downtime_reduction_pct", 0),
            "Downtime Reduction",
            "",
            "%",
            "#3b82f6",
        ),
        (
            impact.get("fleet_reliability_score", 0),
            "Fleet Reliability",
            "",
            "%",
            "#8b5cf6",
        ),
        (
            impact.get("failure_prevention_rate", 0),
            "Failure Prevention",
            "",
            "%",
            "#06b6d4",
        ),
    ]
    for i, (val, title, pref, suf, color) in enumerate(items, 1):
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=val,
                title=dict(text=title, font=dict(size=12, color="#94a3b8")),
                number=dict(
                    prefix=pref,
                    suffix=suf,
                    font=dict(size=28, color=color),
                    valueformat=",.0f" if val >= 100 else ".1f",
                ),
            ),
            row=1,
            col=i,
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def risk_vs_rul_scatter(rec_df: pd.DataFrame) -> go.Figure:
    colors = rec_df["maintenance_priority"].map(PRIORITY_COLORS).fillna("#94a3b8")
    fig = go.Figure(
        go.Scatter(
            x=rec_df["predicted_rul"],
            y=rec_df["failure_risk_pct"],
            mode="markers",
            marker=dict(
                color=colors,
                size=rec_df["urgency_score"] * 2 + 6,
                opacity=0.85,
                line=dict(width=1, color="rgba(0,0,0,0.3)"),
            ),
            text=rec_df["engine_id"].astype(str),
            hovertemplate="Engine %{text}<br>RUL: %{x:.0f}<br>Risk: %{y:.1f}%<extra></extra>",
        )
    )
    _apply_layout(fig, "Failure Risk vs Predicted RUL", 420)
    fig.update_xaxes(title="Predicted RUL (cycles)")
    fig.update_yaxes(title="Failure Risk (%)")
    return fig


def reliability_vs_health(rec_df: pd.DataFrame) -> go.Figure:
    colors = rec_df["maintenance_priority"].map(PRIORITY_COLORS).fillna("#94a3b8")
    fig = go.Figure(
        go.Scatter(
            x=rec_df["health_score"],
            y=rec_df["equipment_reliability"],
            mode="markers",
            marker=dict(
                color=colors,
                size=10,
                opacity=0.85,
                line=dict(width=1, color="rgba(0,0,0,0.3)"),
            ),
            text=rec_df["engine_id"].astype(str),
            hovertemplate="Engine %{text}<br>Health: %{x:.1f}<br>Reliability: %{y:.1f}%<extra></extra>",
        )
    )
    _apply_layout(fig, "Health vs Reliability", 400)
    fig.update_xaxes(title="Health Score")
    fig.update_yaxes(title="Reliability Score (%)")
    return fig


def urgency_heatmap(rec_df: pd.DataFrame) -> go.Figure:
    df = rec_df.sort_values("urgency_score", ascending=False)
    z = [df["failure_risk_pct"].values, df["urgency_score"].values * 10]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=df["engine_id"].astype(str),
            y=["Failure Risk (%)", "Urgency (×10)"],
            colorscale=[
                [0, "#0f2b0f"],
                [0.3, "#10b981"],
                [0.5, "#f59e0b"],
                [0.7, "#f97316"],
                [1, "#ef4444"],
            ],
            text=np.round(np.array(z), 1),
            texttemplate="%{text}",
            textfont=dict(size=9),
            colorbar=dict(title="Score"),
        )
    )
    _apply_layout(fig, "Equipment Urgency & Risk Heatmap", 300)
    fig.update_xaxes(title="Engine ID", type="category", tickfont=dict(size=8))
    return fig


def training_loss_chart(history_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "loss" in history_df.columns:
        fig.add_trace(
            go.Scatter(
                y=history_df["loss"],
                mode="lines",
                name="Train Loss",
                line=dict(color="#3b82f6", width=2),
            )
        )
    if "val_loss" in history_df.columns:
        fig.add_trace(
            go.Scatter(
                y=history_df["val_loss"],
                mode="lines",
                name="Val Loss",
                line=dict(color="#06b6d4", width=2),
            )
        )
    if "mae" in history_df.columns:
        fig.add_trace(
            go.Scatter(
                y=history_df["mae"],
                mode="lines",
                name="Train MAE",
                line=dict(color="#8b5cf6", width=1.5, dash="dot"),
            ),
            secondary_y=True,
        )
    if "val_mae" in history_df.columns:
        fig.add_trace(
            go.Scatter(
                y=history_df["val_mae"],
                mode="lines",
                name="Val MAE",
                line=dict(color="#ec4899", width=1.5, dash="dot"),
            ),
            secondary_y=True,
        )
    _apply_layout(fig, "LSTM Training History", 400)
    fig.update_xaxes(title="Epoch")
    fig.update_yaxes(title="Loss", secondary_y=False)
    fig.update_yaxes(title="MAE", secondary_y=True)
    return fig
