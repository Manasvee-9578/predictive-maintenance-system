"""
╔══════════════════════════════════════════════════════════════╗
║   Maintenance Utilities — I/O, Formatting & Alerts          ║
╚══════════════════════════════════════════════════════════════╝

Shared helper functions for the maintenance module:

    • Data loading and merging
    • Recommendation card formatting
    • Alert message generation
    • CSV export
    • Visualization builders
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from configs.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

PLOT_TEMPLATE = "plotly_white"

# ── Color palettes ───────────────────────────────────────────────

PRIORITY_COLORS = {
    "Low": "#2E7D32",
    "Medium": "#F9A825",
    "High": "#EF6C00",
    "Critical": "#C62828",
}

ACTION_COLORS = {
    "Continue Monitoring": "#2E7D32",
    "Schedule Maintenance": "#1565C0",
    "Immediate Inspection Required": "#EF6C00",
    "Replace Component Soon": "#C62828",
}

RISK_TIER_COLORS = {
    "Low": "#4CAF50",
    "Moderate": "#FFC107",
    "Elevated": "#FF9800",
    "Critical": "#F44336",
}


# ═══════════════════════════════════════════════════════════════
#  Data Loading & Merging
# ═══════════════════════════════════════════════════════════════


def load_prediction_data(
    predictions_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Load all prediction outputs needed for maintenance analysis.

    Reads:
        - ``rul_predictions.csv``      — per-engine RUL forecasts
        - ``health_scores.csv``        — training health scores
        - ``health_scores_prediction.csv`` — prediction health scores
        - ``anomaly_predictions_batch.csv`` — anomaly detection on test set

    Returns:
        Dictionary with keys: ``rul``, ``health_train``,
        ``health_pred``, ``anomaly_batch``
    """
    pred_dir = predictions_dir or (Settings.OUTPUT_DIR / "predictions")
    data: Dict[str, pd.DataFrame] = {}

    files = {
        "rul": "rul_predictions.csv",
        "health_train": "health_scores.csv",
        "health_pred": "health_scores_prediction.csv",
        "anomaly_batch": "anomaly_predictions_batch.csv",
    }

    for key, filename in files.items():
        path = pred_dir / filename
        if path.exists():
            data[key] = pd.read_csv(path)
            logger.info(f"Loaded {filename} — {len(data[key])} rows")
        else:
            logger.warning(f"File not found: {path}")
            data[key] = pd.DataFrame()

    return data


def merge_equipment_data(
    rul_df: pd.DataFrame,
    health_df: pd.DataFrame,
    anomaly_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge RUL predictions, health scores, and anomaly summaries.

    Creates a single per-engine DataFrame suitable for
    recommendation generation.

    Args:
        rul_df:     RUL predictions (engine_id, predicted_rul, ...)
        health_df:  Health scores (engine_id, health_score, ...)
        anomaly_df: Optional per-engine anomaly summary

    Returns:
        Merged DataFrame indexed by ``engine_id``
    """
    if rul_df.empty:
        logger.error("RUL DataFrame is empty — cannot merge")
        return pd.DataFrame()

    merged = rul_df.copy()

    # Merge health scores
    if not health_df.empty and "engine_id" in health_df.columns:
        health_cols = [
            "engine_id",
            "health_score",
            "health_status",
            "mean_anomaly_score",
            "max_anomaly_score",
            "anomaly_count",
        ]
        available_cols = [c for c in health_cols if c in health_df.columns]
        merged = merged.merge(
            health_df[available_cols],
            on="engine_id",
            how="left",
            suffixes=("", "_health"),
        )

    # Compute degradation rate from anomaly batch if available
    if anomaly_df is not None and not anomaly_df.empty:
        if "degradation_rate" in anomaly_df.columns:
            deg_summary = (
                anomaly_df.groupby("engine_id")["degradation_rate"]
                .agg(["mean", "max", "last"])
                .rename(
                    columns={
                        "mean": "degradation_rate",
                        "max": "max_degradation_rate",
                        "last": "latest_degradation_rate",
                    }
                )
                .reset_index()
            )
            merged = merged.merge(deg_summary, on="engine_id", how="left")

    # Fill missing values with safe defaults
    merged["health_score"] = merged.get("health_score", pd.Series(75.0)).fillna(75.0)
    merged["mean_anomaly_score"] = merged.get(
        "mean_anomaly_score", pd.Series(0.0)
    ).fillna(0.0)
    merged["degradation_rate"] = merged.get("degradation_rate", pd.Series(0.01)).fillna(
        0.01
    )

    logger.info(f"Merged equipment data — {len(merged)} engines")
    return merged


# ═══════════════════════════════════════════════════════════════
#  Recommendation Card Formatting
# ═══════════════════════════════════════════════════════════════


def format_recommendation_card(row: pd.Series) -> Dict[str, Any]:
    """Format a single equipment record into a dashboard-ready
    recommendation card.

    Args:
        row: Series with recommendation columns

    Returns:
        Dictionary suitable for JSON serialization or dashboard rendering
    """
    engine_id = int(row.get("engine_id", 0))
    priority = row.get("maintenance_priority", "Low")
    action = row.get("recommended_action", "Continue Monitoring")
    risk = float(row.get("failure_risk_pct", 0))
    urgency = float(row.get("urgency_score", 0))
    rul = float(row.get("predicted_rul", 125))
    health = float(row.get("health_score", 100))
    reliability = float(row.get("equipment_reliability", 0))

    # Generate human-readable description
    if priority == "Critical":
        description = (
            f"Engine {engine_id} requires IMMEDIATE attention. "
            f"Predicted remaining life is {rul:.0f} cycles with "
            f"{risk:.1f}% failure risk. Health score is critically low at {health:.1f}."
        )
    elif priority == "High":
        description = (
            f"Engine {engine_id} shows significant degradation. "
            f"Schedule maintenance within the next {max(1, int(rul * 0.3))} cycles. "
            f"Current failure risk: {risk:.1f}%."
        )
    elif priority == "Medium":
        description = (
            f"Engine {engine_id} is showing early signs of wear. "
            f"Plan maintenance during next scheduled downtime. "
            f"Estimated {rul:.0f} cycles remaining."
        )
    else:
        description = (
            f"Engine {engine_id} is operating within normal parameters. "
            f"Continue standard monitoring. Reliability score: {reliability:.1f}%."
        )

    # Estimated time window
    if rul <= 15:
        time_window = "Immediate (0-2 days)"
    elif rul <= 30:
        time_window = "Urgent (3-7 days)"
    elif rul <= 60:
        time_window = "Short-term (1-3 weeks)"
    elif rul <= 100:
        time_window = "Medium-term (1-2 months)"
    else:
        time_window = "Long-term (3+ months)"

    card = {
        "engine_id": engine_id,
        "priority": priority,
        "priority_color": PRIORITY_COLORS.get(priority, "#616161"),
        "recommended_action": action,
        "action_color": ACTION_COLORS.get(action, "#616161"),
        "failure_risk_pct": round(risk, 1),
        "urgency_score": round(urgency, 1),
        "predicted_rul_cycles": round(rul, 1),
        "health_score": round(health, 1),
        "equipment_reliability": round(reliability, 1),
        "time_window": time_window,
        "description": description,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    return card


def generate_machine_summary(
    recommendations_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Generate a list of machine summary cards for all equipment.

    Args:
        recommendations_df: Full recommendations DataFrame

    Returns:
        List of card dictionaries
    """
    cards = []
    for _, row in recommendations_df.iterrows():
        cards.append(format_recommendation_card(row))

    logger.info(f"Generated {len(cards)} machine summary cards")
    return cards


# ═══════════════════════════════════════════════════════════════
#  Alert Message Generation
# ═══════════════════════════════════════════════════════════════


def generate_alert_message(
    row: pd.Series,
    include_timestamp: bool = True,
) -> str:
    """Generate a human-readable alert message for an equipment unit.

    Args:
        row:               Series with recommendation columns
        include_timestamp: Whether to include generation timestamp

    Returns:
        Formatted alert string
    """
    engine_id = int(row.get("engine_id", 0))
    priority = row.get("maintenance_priority", "Low")
    action = row.get("recommended_action", "Continue Monitoring")
    risk = float(row.get("failure_risk_pct", 0))
    rul = float(row.get("predicted_rul", 125))

    severity_emoji = {
        "Critical": "🔴",
        "High": "🟠",
        "Medium": "🟡",
        "Low": "🟢",
    }

    emoji = severity_emoji.get(priority, "ℹ️")

    lines = [
        f"{emoji} [{priority.upper()}] Engine {engine_id}",
        f"   Action: {action}",
        f"   Risk: {risk:.1f}% | RUL: {rul:.0f} cycles",
    ]

    if include_timestamp:
        lines.append(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    return "\n".join(lines)


def generate_alert_report(
    recommendations_df: pd.DataFrame,
    priority_filter: Optional[List[str]] = None,
) -> str:
    """Generate a full alert report for equipment needing attention.

    Args:
        recommendations_df: Full recommendations DataFrame
        priority_filter:    Only include these priority levels
                           (default: High, Critical)

    Returns:
        Multi-line alert report string
    """
    if priority_filter is None:
        priority_filter = ["High", "Critical"]

    filtered = recommendations_df[
        recommendations_df["maintenance_priority"].isin(priority_filter)
    ].sort_values("urgency_score", ascending=False)

    if filtered.empty:
        return "✅ No equipment requires immediate attention."

    header = (
        "╔══════════════════════════════════════════════════════════╗\n"
        "║   MAINTENANCE ALERT REPORT                              ║\n"
        "╚══════════════════════════════════════════════════════════╝\n"
        f"\n  {len(filtered)} unit(s) require attention\n"
        f"  Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "─" * 58
    )

    alerts = [header]
    for _, row in filtered.iterrows():
        alerts.append(generate_alert_message(row))

    alerts.append("─" * 58)
    report = "\n\n".join(alerts)

    logger.info(f"Alert report generated — {len(filtered)} alerts")
    return report


# ═══════════════════════════════════════════════════════════════
#  CSV Export
# ═══════════════════════════════════════════════════════════════


def save_recommendations_csv(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    filename: str = "maintenance_recommendations.csv",
) -> Path:
    """Save maintenance recommendations to CSV.

    Args:
        df:         Recommendations DataFrame
        output_dir: Output directory (default: outputs/predictions)
        filename:   Output filename

    Returns:
        Path to saved file
    """
    out = output_dir or (Settings.OUTPUT_DIR / "predictions")
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    df.to_csv(path, index=False)
    logger.info(f"Saved recommendations to {path} ({len(df)} rows)")
    return path


def save_risk_summary_csv(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    filename: str = "risk_summary.csv",
) -> Path:
    """Save risk summary to CSV.

    Args:
        df:         Risk summary DataFrame
        output_dir: Output directory (default: outputs/predictions)
        filename:   Output filename

    Returns:
        Path to saved file
    """
    out = output_dir or (Settings.OUTPUT_DIR / "predictions")
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    df.to_csv(path, index=False)
    logger.info(f"Saved risk summary to {path} ({len(df)} rows)")
    return path


# ═══════════════════════════════════════════════════════════════
#  Visualization Builders
# ═══════════════════════════════════════════════════════════════


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert a hex color string to rgba() format for Plotly."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _base_layout(fig: go.Figure, title: str) -> go.Figure:
    """Apply consistent chart styling."""
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=dict(text=title, font=dict(size=18, color="#1A1A2E")),
        font=dict(family="Segoe UI, Arial, sans-serif", size=12),
        margin=dict(l=60, r=30, t=70, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_maintenance_priority_distribution(
    recommendations_df: pd.DataFrame,
    title: str = "Maintenance Priority Distribution",
) -> go.Figure:
    """Donut chart showing maintenance priority breakdown.

    Args:
        recommendations_df: DataFrame with ``maintenance_priority``
        title: Chart title

    Returns:
        Plotly Figure
    """
    counts = recommendations_df["maintenance_priority"].value_counts()

    # Ensure consistent ordering
    order = ["Critical", "High", "Medium", "Low"]
    labels = [p for p in order if p in counts.index]
    values = [counts[p] for p in labels]
    colors = [PRIORITY_COLORS[p] for p in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            marker=dict(colors=colors, line=dict(color="white", width=2)),
            textinfo="label+percent+value",
            textfont_size=13,
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Count: %{value}<br>"
                "Percentage: %{percent}<extra></extra>"
            ),
        )
    )

    # Add center annotation
    total = sum(values)
    fig.add_annotation(
        text=f"<b>{total}</b><br>Total",
        x=0.5,
        y=0.5,
        font_size=16,
        showarrow=False,
    )

    return _base_layout(fig, title)


def plot_risk_distribution(
    recommendations_df: pd.DataFrame,
    title: str = "Failure Risk Distribution",
) -> go.Figure:
    """Histogram of failure risk percentages across the fleet.

    Args:
        recommendations_df: DataFrame with ``failure_risk_pct``
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    risk_values = recommendations_df["failure_risk_pct"]

    fig.add_trace(
        go.Histogram(
            x=risk_values,
            nbinsx=25,
            marker=dict(
                color=risk_values,
                colorscale=[
                    [0, "#4CAF50"],
                    [0.33, "#FFC107"],
                    [0.66, "#FF9800"],
                    [1.0, "#F44336"],
                ],
                colorbar=dict(title="Risk %"),
                line=dict(width=0.5, color="white"),
            ),
            hovertemplate="Risk: %{x:.1f}%<br>Count: %{y}<extra></extra>",
        )
    )

    # Add threshold lines
    fig.add_vline(
        x=25,
        line_dash="dash",
        line_color="#4CAF50",
        annotation_text="Low",
        annotation_position="top",
    )
    fig.add_vline(
        x=50,
        line_dash="dash",
        line_color="#FFC107",
        annotation_text="Moderate",
        annotation_position="top",
    )
    fig.add_vline(
        x=75,
        line_dash="dash",
        line_color="#F44336",
        annotation_text="Critical",
        annotation_position="top",
    )

    _base_layout(fig, title)
    fig.update_xaxes(title="Failure Risk (%)", range=[0, 100])
    fig.update_yaxes(title="Number of Equipment Units")
    return fig


def plot_equipment_criticality_dashboard(
    recommendations_df: pd.DataFrame,
    title: str = "Equipment Criticality Dashboard",
) -> go.Figure:
    """Multi-panel dashboard showing equipment criticality metrics.

    Creates a 2×2 subplot with:
        1. Risk vs RUL scatter
        2. Urgency by priority bar
        3. Health vs reliability scatter
        4. Top critical equipment table

    Args:
        recommendations_df: Full recommendations DataFrame
        title: Dashboard title

    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Failure Risk vs Predicted RUL",
            "Urgency Score by Priority",
            "Health Score vs Reliability",
            "Top 10 Critical Equipment",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "table"}],
        ],
    )

    df = recommendations_df.copy()

    # ── Panel 1: Risk vs RUL ─────────────────────────────────
    colors_mapped = df["maintenance_priority"].map(PRIORITY_COLORS)
    fig.add_trace(
        go.Scatter(
            x=df["predicted_rul"],
            y=df["failure_risk_pct"],
            mode="markers",
            marker=dict(
                color=colors_mapped,
                size=df["urgency_score"] * 2 + 5,
                opacity=0.8,
                line=dict(width=0.5, color="white"),
            ),
            text=df["engine_id"].astype(str),
            hovertemplate=(
                "Engine %{text}<br>"
                "RUL: %{x:.0f} cycles<br>"
                "Risk: %{y:.1f}%<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Predicted RUL (cycles)", row=1, col=1)
    fig.update_yaxes(title_text="Failure Risk (%)", row=1, col=1)

    # ── Panel 2: Urgency by Priority ─────────────────────────
    priority_order = ["Low", "Medium", "High", "Critical"]
    priority_urgency = (
        df.groupby("maintenance_priority")["urgency_score"]
        .mean()
        .reindex(priority_order)
        .fillna(0)
    )
    fig.add_trace(
        go.Bar(
            x=priority_urgency.index,
            y=priority_urgency.values,
            marker_color=[
                PRIORITY_COLORS.get(p, "#616161") for p in priority_urgency.index
            ],
            text=[f"{v:.1f}" for v in priority_urgency.values],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Priority", row=1, col=2)
    fig.update_yaxes(title_text="Avg Urgency Score", row=1, col=2)

    # ── Panel 3: Health vs Reliability ───────────────────────
    if "equipment_reliability" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["health_score"],
                y=df["equipment_reliability"],
                mode="markers",
                marker=dict(
                    color=colors_mapped,
                    size=8,
                    opacity=0.8,
                    line=dict(width=0.5, color="white"),
                ),
                text=df["engine_id"].astype(str),
                hovertemplate=(
                    "Engine %{text}<br>"
                    "Health: %{x:.1f}<br>"
                    "Reliability: %{y:.1f}%<extra></extra>"
                ),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text="Health Score", row=2, col=1)
        fig.update_yaxes(title_text="Reliability Score", row=2, col=1)

    # ── Panel 4: Top Critical Equipment ──────────────────────
    top_critical = df.nlargest(10, "failure_risk_pct")
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Engine", "RUL", "Risk %", "Priority"],
                fill_color="#1A1A2E",
                font=dict(color="white", size=12),
                align="center",
            ),
            cells=dict(
                values=[
                    top_critical["engine_id"].astype(str),
                    top_critical["predicted_rul"].round(0).astype(int).astype(str),
                    top_critical["failure_risk_pct"].round(1).astype(str) + "%",
                    top_critical["maintenance_priority"],
                ],
                fill_color=[
                    ["white"] * len(top_critical),
                    ["white"] * len(top_critical),
                    ["white"] * len(top_critical),
                    [
                        _hex_to_rgba(PRIORITY_COLORS.get(p, "#ffffff"), 0.2)
                        for p in top_critical["maintenance_priority"]
                    ],
                ],
                align="center",
                font=dict(size=11),
                height=25,
            ),
        ),
        row=2,
        col=2,
    )

    _base_layout(fig, title)
    fig.update_layout(height=750, showlegend=False)
    return fig


def plot_recommendation_action_chart(
    recommendations_df: pd.DataFrame,
    title: str = "Recommended Actions Distribution",
) -> go.Figure:
    """Horizontal bar chart of recommended maintenance actions.

    Args:
        recommendations_df: DataFrame with ``recommended_action``
        title: Chart title

    Returns:
        Plotly Figure
    """
    counts = recommendations_df["recommended_action"].value_counts()

    fig = go.Figure(
        go.Bar(
            y=counts.index,
            x=counts.values,
            orientation="h",
            marker_color=[ACTION_COLORS.get(act, "#616161") for act in counts.index],
            text=[f"{v} ({v/len(recommendations_df)*100:.0f}%)" for v in counts.values],
            textposition="outside",
            hovertemplate="%{y}: %{x} units<extra></extra>",
        )
    )

    _base_layout(fig, title)
    fig.update_xaxes(title="Number of Equipment Units")
    fig.update_yaxes(title="")
    fig.update_layout(height=400)
    return fig


def plot_urgency_heatmap(
    recommendations_df: pd.DataFrame,
    title: str = "Equipment Urgency & Risk Heatmap",
) -> go.Figure:
    """Heatmap showing urgency and risk across all equipment.

    Args:
        recommendations_df: DataFrame with engine-level metrics
        title: Chart title

    Returns:
        Plotly Figure
    """
    df = recommendations_df.sort_values("urgency_score", ascending=False)

    metrics = ["failure_risk_pct", "urgency_score"]
    metric_labels = ["Failure Risk (%)", "Urgency (0-10)"]

    # Normalize urgency to 0-100 for consistent heatmap scale
    z_data = [
        df["failure_risk_pct"].values,
        df["urgency_score"].values * 10,  # scale to 0-100
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z_data,
            x=df["engine_id"].astype(str),
            y=metric_labels,
            colorscale=[
                [0, "#1B5E20"],
                [0.3, "#4CAF50"],
                [0.5, "#FFC107"],
                [0.7, "#FF9800"],
                [1.0, "#D32F2F"],
            ],
            text=np.round(np.array(z_data), 1),
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorbar=dict(title="Score"),
            hovertemplate=("Engine %{x}<br>" "%{y}: %{z:.1f}<extra></extra>"),
        )
    )

    _base_layout(fig, title)
    fig.update_xaxes(title="Engine ID", type="category")
    fig.update_layout(height=350)
    return fig


def plot_business_impact_summary(
    impact_metrics: Dict[str, float],
    title: str = "Business Impact Summary",
) -> go.Figure:
    """KPI indicator cards for business impact metrics.

    Args:
        impact_metrics: Dictionary from ``RiskScorer.compute_business_impact``
        title: Chart title

    Returns:
        Plotly Figure with indicator gauges
    """
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "indicator"}] * 3,
            [{"type": "indicator"}] * 3,
        ],
        vertical_spacing=0.25,
        horizontal_spacing=0.08,
    )

    indicators = [
        {
            "value": impact_metrics.get("estimated_cost_savings_usd", 0),
            "title": "Cost Savings",
            "prefix": "$",
            "suffix": "",
            "color": "#2E7D32",
            "row": 1,
            "col": 1,
        },
        {
            "value": impact_metrics.get("downtime_reduction_pct", 0),
            "title": "Downtime Reduction",
            "prefix": "",
            "suffix": "%",
            "color": "#1565C0",
            "row": 1,
            "col": 2,
        },
        {
            "value": impact_metrics.get("fleet_reliability_score", 0),
            "title": "Fleet Reliability",
            "prefix": "",
            "suffix": "%",
            "color": "#6A1B9A",
            "row": 1,
            "col": 3,
        },
        {
            "value": impact_metrics.get("failure_prevention_rate", 0),
            "title": "Failure Prevention",
            "prefix": "",
            "suffix": "%",
            "color": "#00695C",
            "row": 2,
            "col": 1,
        },
        {
            "value": impact_metrics.get("critical_equipment_count", 0),
            "title": "Critical Equipment",
            "prefix": "",
            "suffix": " units",
            "color": "#C62828",
            "row": 2,
            "col": 2,
        },
        {
            "value": impact_metrics.get("fleet_size", 0),
            "title": "Fleet Size",
            "prefix": "",
            "suffix": " units",
            "color": "#37474F",
            "row": 2,
            "col": 3,
        },
    ]

    for ind in indicators:
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=ind["value"],
                title=dict(text=ind["title"], font=dict(size=14)),
                number=dict(
                    prefix=ind["prefix"],
                    suffix=ind["suffix"],
                    font=dict(size=28, color=ind["color"]),
                    valueformat=",.0f" if ind["value"] >= 100 else ",.1f",
                ),
            ),
            row=ind["row"],
            col=ind["col"],
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#1A1A2E")),
        font=dict(family="Segoe UI, Arial, sans-serif"),
        height=400,
        margin=dict(l=30, r=30, t=70, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def save_figure(
    fig: go.Figure,
    output_dir: Optional[Path] = None,
    filename: str = "chart.html",
) -> Path:
    """Save a Plotly figure as dashboard-ready HTML.

    Args:
        fig:        Plotly Figure object
        output_dir: Target directory (default: outputs/plots)
        filename:   Output filename

    Returns:
        Path to saved file
    """
    out = output_dir or (Settings.OUTPUT_DIR / "plots")
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    fig.write_html(str(path), include_plotlyjs="cdn")
    logger.info(f"Saved chart to {path}")
    return path
