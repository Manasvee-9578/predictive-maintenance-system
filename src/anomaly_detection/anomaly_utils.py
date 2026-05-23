"""
╔══════════════════════════════════════════════════════════════╗
║   Anomaly Detection Utilities                                ║
║   Reusable helpers for scoring, classification & viz         ║
╚══════════════════════════════════════════════════════════════╝

Production-grade utility functions for the anomaly detection
subsystem.  Every function is stateless and composable:

    ▸ Severity classification  (Normal / Warning / Critical)
    ▸ Equipment health scoring (0 – 100 scale)
    ▸ Feature selector for sensor columns
    ▸ Plotly visualization builders (industrial-monitoring style)
    ▸ Result export helpers
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

from configs.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

SEVERITY_NORMAL = "Normal"
SEVERITY_WARNING = "Warning"
SEVERITY_CRITICAL = "Critical"

# Industrial color palette
COLOR_NORMAL = "#00C853"  # vibrant green
COLOR_WARNING = "#FFD600"  # amber-yellow
COLOR_CRITICAL = "#FF1744"  # bright red
COLOR_BG_DARK = "#0D1117"  # dark panel background
COLOR_GRID = "#21262D"  # subtle grid
COLOR_TEXT = "#C9D1D9"  # muted white text
COLOR_ACCENT = "#58A6FF"  # blue accent


# ═══════════════════════════════════════════════════════════════
#  Feature Selection
# ═══════════════════════════════════════════════════════════════


def get_sensor_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of sensor columns present in *df* that are usable
    for anomaly detection (excludes low-variance and meta columns).

    Falls back to any column matching ``sensor_*`` if the configured
    list does not intersect the DataFrame.
    """
    candidates = [
        c
        for c in Settings.SENSOR_COLUMNS
        if c in df.columns and c not in Settings.DROP_SENSORS
    ]
    if not candidates:
        candidates = [c for c in df.columns if c.startswith("sensor_")]
    return candidates


# ═══════════════════════════════════════════════════════════════
#  Anomaly Score Computation
# ═══════════════════════════════════════════════════════════════


def normalize_anomaly_scores(raw_scores: np.ndarray) -> np.ndarray:
    """
    Normalise Isolation Forest ``decision_function`` output to [0, 1].

    The raw score is negative for anomalies and positive for inliers.
    After normalisation: **0 = perfectly normal**, **1 = extreme anomaly**.
    """
    # Negate so anomalies become positive / large
    inverted = -raw_scores
    mn, mx = inverted.min(), inverted.max()
    if mx - mn == 0:
        return np.zeros_like(inverted)
    return (inverted - mn) / (mx - mn)


# ═══════════════════════════════════════════════════════════════
#  Severity Classification
# ═══════════════════════════════════════════════════════════════


def classify_severity(
    anomaly_scores: np.ndarray,
    warning_threshold: float = 0.55,
    critical_threshold: float = 0.80,
) -> np.ndarray:
    """
    Map normalised anomaly scores to severity labels.

    Args:
        anomaly_scores: Array of normalised scores in [0, 1].
        warning_threshold: Score above which the reading is a Warning.
        critical_threshold: Score above which the reading is Critical.

    Returns:
        Array of string labels: ``Normal`` / ``Warning`` / ``Critical``.
    """
    labels = np.full(len(anomaly_scores), SEVERITY_NORMAL, dtype=object)
    labels[anomaly_scores >= warning_threshold] = SEVERITY_WARNING
    labels[anomaly_scores >= critical_threshold] = SEVERITY_CRITICAL
    return labels


def severity_color(severity: str) -> str:
    """Return the hex colour for a given severity label."""
    return {
        SEVERITY_NORMAL: COLOR_NORMAL,
        SEVERITY_WARNING: COLOR_WARNING,
        SEVERITY_CRITICAL: COLOR_CRITICAL,
    }.get(severity, COLOR_TEXT)


# ═══════════════════════════════════════════════════════════════
#  Equipment Health Scoring
# ═══════════════════════════════════════════════════════════════


def compute_health_score(
    anomaly_scores: np.ndarray,
    engine_ids: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """
    Compute a **per-engine health score** on a 0 – 100 scale.

    The score is defined as:

        ``health = 100 × (1 − mean_anomaly_score_for_engine)``

    A perfectly healthy engine scores 100; an engine with uniformly
    maximal anomaly scores scores 0.

    Returns:
        DataFrame with columns ``engine_id``, ``health_score``,
        ``health_status``, ``mean_anomaly_score``, ``max_anomaly_score``,
        ``anomaly_count``.
    """
    df = pd.DataFrame(
        {
            "engine_id": engine_ids,
            "anomaly_score": anomaly_scores,
        }
    )

    agg = (
        df.groupby("engine_id")["anomaly_score"]
        .agg(
            mean_anomaly_score="mean",
            max_anomaly_score="max",
            std_anomaly_score="std",
            anomaly_count=lambda x: (x >= 0.55).sum(),
        )
        .reset_index()
    )

    agg["health_score"] = (100 * (1 - agg["mean_anomaly_score"])).round(1).clip(0, 100)

    # Health status buckets
    conditions = [
        agg["health_score"] >= 80,
        agg["health_score"] >= 50,
    ]
    choices = [SEVERITY_NORMAL, SEVERITY_WARNING]
    agg["health_status"] = np.select(conditions, choices, default=SEVERITY_CRITICAL)

    return agg.sort_values("health_score")


# ═══════════════════════════════════════════════════════════════
#  Visualization — Industrial Monitoring Theme
# ═══════════════════════════════════════════════════════════════

_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=COLOR_BG_DARK,
    plot_bgcolor=COLOR_BG_DARK,
    font=dict(family="Inter, Segoe UI, sans-serif", color=COLOR_TEXT, size=12),
    title_font=dict(size=18, color="#FFFFFF"),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    xaxis=dict(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID),
    yaxis=dict(gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID),
    margin=dict(l=60, r=30, t=70, b=50),
)


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply the industrial dark-theme to any Plotly figure."""
    fig.update_layout(**_LAYOUT_DEFAULTS)
    return fig


def plot_anomaly_scores_distribution(
    anomaly_scores: np.ndarray,
    severities: np.ndarray,
    title: str = "Anomaly Score Distribution",
) -> go.Figure:
    """
    Histogram of anomaly scores colour-coded by severity.

    Ideal for a dashboard overview panel.
    """
    df = pd.DataFrame({"score": anomaly_scores, "severity": severities})
    color_map = {
        SEVERITY_NORMAL: COLOR_NORMAL,
        SEVERITY_WARNING: COLOR_WARNING,
        SEVERITY_CRITICAL: COLOR_CRITICAL,
    }
    fig = px.histogram(
        df,
        x="score",
        color="severity",
        color_discrete_map=color_map,
        nbins=60,
        barmode="overlay",
        opacity=0.75,
        labels={"score": "Anomaly Score", "severity": "Severity"},
        title=title,
    )
    fig = _apply_theme(fig)
    fig.update_layout(
        xaxis_title="Anomaly Score (0 = Normal, 1 = Extreme)",
        yaxis_title="Number of Observations",
    )
    return fig


def plot_sensor_with_anomalies(
    df: pd.DataFrame,
    sensor_col: str,
    engine_id: int,
    score_col: str = "anomaly_score_norm",
    severity_col: str = "severity",
    title: str | None = None,
) -> go.Figure:
    """
    Line chart of a single sensor for one engine, with anomalies
    highlighted as coloured markers.

    Shows the Normal / Warning / Critical colour coding directly
    on the time-series.
    """
    eng = df[df["engine_id"] == engine_id].copy().sort_values("cycle")
    title = title or f"Engine {engine_id} — {sensor_col} with Anomaly Highlights"

    fig = go.Figure()

    # Baseline trace
    fig.add_trace(
        go.Scatter(
            x=eng["cycle"],
            y=eng[sensor_col],
            mode="lines",
            name=sensor_col,
            line=dict(color=COLOR_ACCENT, width=1.5),
        )
    )

    # Overlay anomaly markers
    for sev, colour, symbol in [
        (SEVERITY_WARNING, COLOR_WARNING, "diamond"),
        (SEVERITY_CRITICAL, COLOR_CRITICAL, "x"),
    ]:
        subset = eng[eng[severity_col] == sev]
        if len(subset) == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["cycle"],
                y=subset[sensor_col],
                mode="markers",
                name=sev,
                marker=dict(
                    color=colour,
                    size=8,
                    symbol=symbol,
                    line=dict(width=1, color="#FFFFFF"),
                ),
            )
        )

    fig = _apply_theme(fig)
    fig.update_layout(
        title=title,
        xaxis_title="Cycle",
        yaxis_title=sensor_col,
    )
    return fig


def plot_health_score_gauge(
    health_score: float,
    engine_id: int | str = "",
) -> go.Figure:
    """
    Single-engine health gauge (0 – 100) with red / yellow / green
    banding.  Dashboard-ready.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            title=dict(
                text=f"Engine {engine_id} Health", font=dict(size=16, color="#FFF")
            ),
            number=dict(font=dict(size=40, color="#FFF"), suffix="%"),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="#555"),
                bar=dict(color=COLOR_ACCENT, thickness=0.35),
                bgcolor=COLOR_BG_DARK,
                borderwidth=2,
                bordercolor="#333",
                steps=[
                    dict(range=[0, 50], color=COLOR_CRITICAL),
                    dict(range=[50, 80], color=COLOR_WARNING),
                    dict(range=[80, 100], color=COLOR_NORMAL),
                ],
                threshold=dict(
                    line=dict(color="#FFF", width=3),
                    thickness=0.8,
                    value=health_score,
                ),
            ),
        )
    )
    fig = _apply_theme(fig)
    fig.update_layout(height=320)
    return fig


def plot_health_score_heatmap(health_df: pd.DataFrame) -> go.Figure:
    """
    Heatmap / bar chart of all engine health scores sorted ascending.

    Green bars = healthy,  yellow = degrading,  red = critical.
    """
    df = health_df.sort_values("health_score", ascending=True).copy()
    colours = (
        df["health_status"]
        .map(
            {
                SEVERITY_NORMAL: COLOR_NORMAL,
                SEVERITY_WARNING: COLOR_WARNING,
                SEVERITY_CRITICAL: COLOR_CRITICAL,
            }
        )
        .tolist()
    )

    fig = go.Figure(
        go.Bar(
            x=df["health_score"],
            y=df["engine_id"].astype(str),
            orientation="h",
            marker=dict(color=colours, line=dict(width=0.5, color="#222")),
            text=df["health_score"].apply(lambda v: f"{v:.0f}%"),
            textposition="outside",
            textfont=dict(size=10, color=COLOR_TEXT),
        )
    )
    fig = _apply_theme(fig)
    fig.update_layout(
        title="Equipment Health Scores — All Engines",
        xaxis_title="Health Score (0–100)",
        yaxis_title="Engine ID",
        height=max(400, len(df) * 22),
        yaxis=dict(type="category"),
    )
    return fig


def plot_anomaly_timeline(
    df: pd.DataFrame,
    engine_id: int,
    score_col: str = "anomaly_score_norm",
    severity_col: str = "severity",
) -> go.Figure:
    """
    Anomaly score time-line for one engine with coloured severity
    bands and a threshold reference line.
    """
    eng = df[df["engine_id"] == engine_id].sort_values("cycle")
    fig = go.Figure()

    # Score line
    fig.add_trace(
        go.Scatter(
            x=eng["cycle"],
            y=eng[score_col],
            mode="lines+markers",
            name="Anomaly Score",
            line=dict(color=COLOR_ACCENT, width=2),
            marker=dict(size=4, color=COLOR_ACCENT),
        )
    )

    # Warning threshold
    fig.add_hline(
        y=0.55,
        line_dash="dash",
        line_color=COLOR_WARNING,
        annotation_text="Warning",
        annotation_position="top left",
    )
    fig.add_hline(
        y=0.80,
        line_dash="dash",
        line_color=COLOR_CRITICAL,
        annotation_text="Critical",
        annotation_position="top left",
    )

    # Background severity bands
    fig.add_hrect(
        y0=0, y1=0.55, fillcolor=COLOR_NORMAL, opacity=0.07, layer="below", line_width=0
    )
    fig.add_hrect(
        y0=0.55,
        y1=0.80,
        fillcolor=COLOR_WARNING,
        opacity=0.07,
        layer="below",
        line_width=0,
    )
    fig.add_hrect(
        y0=0.80,
        y1=1.0,
        fillcolor=COLOR_CRITICAL,
        opacity=0.10,
        layer="below",
        line_width=0,
    )

    fig = _apply_theme(fig)
    fig.update_layout(
        title=f"Engine {engine_id} — Anomaly Score Timeline",
        xaxis_title="Cycle",
        yaxis_title="Anomaly Score",
        yaxis_range=[-0.02, 1.05],
    )
    return fig


def plot_multi_sensor_anomaly_panel(
    df: pd.DataFrame,
    engine_id: int,
    sensors: list[str],
    severity_col: str = "severity",
) -> go.Figure:
    """
    Multi-row subplot panel: one row per sensor, anomalies highlighted.

    Creates a unified dashboard view for a single engine across
    multiple sensor channels.
    """
    eng = df[df["engine_id"] == engine_id].sort_values("cycle")
    n = len(sensors)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[s.replace("_", " ").title() for s in sensors],
        vertical_spacing=0.03,
    )

    for i, sensor in enumerate(sensors, 1):
        if sensor not in eng.columns:
            continue
        # Normal line
        fig.add_trace(
            go.Scatter(
                x=eng["cycle"],
                y=eng[sensor],
                mode="lines",
                name=sensor,
                showlegend=(i == 1),
                line=dict(color=COLOR_ACCENT, width=1),
            ),
            row=i,
            col=1,
        )

        # Anomaly markers
        for sev, colour, sym in [
            (SEVERITY_WARNING, COLOR_WARNING, "diamond"),
            (SEVERITY_CRITICAL, COLOR_CRITICAL, "x"),
        ]:
            sub = eng[eng[severity_col] == sev]
            if len(sub) == 0 or sensor not in sub.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["cycle"],
                    y=sub[sensor],
                    mode="markers",
                    name=f"{sev}" if i == 1 else None,
                    showlegend=(i == 1),
                    marker=dict(
                        color=colour,
                        size=6,
                        symbol=sym,
                        line=dict(width=0.5, color="#FFF"),
                    ),
                ),
                row=i,
                col=1,
            )

    fig = _apply_theme(fig)
    fig.update_layout(
        title=f"Engine {engine_id} — Multi-Sensor Anomaly Panel",
        height=220 * n,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  Export Helpers
# ═══════════════════════════════════════════════════════════════


def save_anomaly_results(
    df: pd.DataFrame,
    output_path: Path,
    filename: str = "anomaly_results.csv",
) -> Path:
    """Save anomaly-annotated DataFrame to CSV."""
    output_path.mkdir(parents=True, exist_ok=True)
    fpath = output_path / filename
    df.to_csv(fpath, index=False)
    logger.info(f"Anomaly results saved → {fpath}")
    return fpath


def save_health_report(
    health_df: pd.DataFrame,
    output_path: Path,
    filename: str = "health_scores.csv",
) -> Path:
    """Save engine health score report to CSV."""
    output_path.mkdir(parents=True, exist_ok=True)
    fpath = output_path / filename
    health_df.to_csv(fpath, index=False)
    logger.info(f"Health report saved → {fpath}")
    return fpath


def save_plotly_figure(
    fig: go.Figure,
    output_path: Path,
    filename: str = "chart.html",
) -> Path:
    """Save a Plotly figure as an interactive HTML file."""
    output_path.mkdir(parents=True, exist_ok=True)
    fpath = output_path / filename
    fig.write_html(str(fpath), include_plotlyjs="cdn")
    logger.info(f"Chart saved → {fpath}")
    return fpath
