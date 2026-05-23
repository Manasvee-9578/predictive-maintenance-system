"""Reusable utilities for RUL forecasting, evaluation, and dashboards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from configs.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

RISK_LOW = "Low Risk"
RISK_MEDIUM = "Medium Risk"
RISK_HIGH = "High Risk"

URGENCY_ROUTINE = "Routine"
URGENCY_SCHEDULED = "Scheduled"
URGENCY_PRIORITY = "Priority"
URGENCY_IMMEDIATE = "Immediate"

RISK_COLORS = {
    RISK_LOW: "#2E7D32",
    RISK_MEDIUM: "#F9A825",
    RISK_HIGH: "#C62828",
}

URGENCY_COLORS = {
    URGENCY_ROUTINE: "#2E7D32",
    URGENCY_SCHEDULED: "#1565C0",
    URGENCY_PRIORITY: "#EF6C00",
    URGENCY_IMMEDIATE: "#B71C1C",
}

PLOT_TEMPLATE = "plotly_white"


def classify_risk(
    predicted_rul: np.ndarray,
    high_risk_threshold: float = 15.0,
    medium_risk_threshold: float = 60.0,
) -> np.ndarray:
    """Convert predicted RUL values into practical risk categories."""
    values = np.asarray(predicted_rul, dtype=float)
    labels = np.full(values.shape, RISK_LOW, dtype=object)
    labels[values <= medium_risk_threshold] = RISK_MEDIUM
    labels[values <= high_risk_threshold] = RISK_HIGH
    return labels


def predict_maintenance_urgency(
    predicted_rul: np.ndarray,
    confidence: np.ndarray | None = None,
) -> np.ndarray:
    """Map RUL forecasts to maintenance action levels."""
    values = np.asarray(predicted_rul, dtype=float)
    urgency = np.full(values.shape, URGENCY_ROUTINE, dtype=object)
    urgency[values <= 80] = URGENCY_SCHEDULED
    urgency[values <= 30] = URGENCY_PRIORITY
    urgency[values <= 10] = URGENCY_IMMEDIATE

    if confidence is not None:
        conf = np.asarray(confidence, dtype=float)
        low_confidence = conf < 0.45
        urgency[(urgency == URGENCY_ROUTINE) & low_confidence] = URGENCY_SCHEDULED
        urgency[(urgency == URGENCY_SCHEDULED) & low_confidence] = URGENCY_PRIORITY

    return urgency


def estimate_confidence(predicted_rul: np.ndarray) -> np.ndarray:
    """
    Create a dashboard confidence proxy.

    This is not statistical uncertainty. It is a simple heuristic that lowers
    confidence near operational decision boundaries, where a few cycles can
    change the displayed risk class.
    """
    values = np.asarray(predicted_rul, dtype=float)
    thresholds = np.asarray([10.0, 15.0, 30.0, 60.0, 80.0])
    distances = np.min(np.abs(values[:, None] - thresholds[None, :]), axis=1)
    scaled_distance = np.clip(distances / 30.0, 0.0, 1.0)
    confidence = 0.55 + 0.40 * scaled_distance
    clipped_at_max = values >= Settings.MAX_RUL - 1
    confidence[clipped_at_max] = np.minimum(confidence[clipped_at_max], 0.70)
    return np.round(np.clip(confidence, 0.35, 0.95), 4)


def nasa_scoring(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA asymmetric RUL scoring function."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    diff = pred - true
    score = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    return float(np.sum(score))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "RUL LSTM",
) -> dict:
    """Compute MAE, RMSE, R2, and NASA score."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)

    if mask.sum() == 0:
        raise ValueError(
            "Cannot compute metrics because no finite labels are available."
        )

    true = true[mask]
    pred = pred[mask]
    mae = float(mean_absolute_error(true, pred))
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    r2 = float(r2_score(true, pred)) if len(true) > 1 else float("nan")
    score = nasa_scoring(true, pred)

    metrics = {
        "model": model_name,
        "samples": int(len(true)),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4) if np.isfinite(r2) else np.nan,
        "nasa_score": round(score, 4),
    }
    logger.info(
        f"{model_name}: MAE={metrics['mae']}, RMSE={metrics['rmse']}, "
        f"R2={metrics['r2']}, NASA={metrics['nasa_score']}"
    )
    return metrics


def add_prediction_columns(
    df: pd.DataFrame, actual_col: str = "actual_rul"
) -> pd.DataFrame:
    """Add dashboard risk, confidence, urgency, and error columns."""
    output = df.copy()
    output["predicted_rul"] = output["predicted_rul"].clip(
        lower=0, upper=Settings.MAX_RUL
    )
    output["confidence"] = estimate_confidence(output["predicted_rul"].to_numpy())
    output["risk_category"] = classify_risk(output["predicted_rul"].to_numpy())
    output["maintenance_urgency"] = predict_maintenance_urgency(
        output["predicted_rul"].to_numpy(),
        output["confidence"].to_numpy(),
    )
    if actual_col in output.columns:
        output["error"] = output["predicted_rul"] - output[actual_col]
        output["abs_error"] = output["error"].abs()
    return output


def _base_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title=title,
        font=dict(family="Segoe UI, Arial, sans-serif", size=12),
        margin=dict(l=60, r=30, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Actual RUL",
) -> go.Figure:
    """Scatter plot for regression quality."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=true,
            y=pred,
            mode="markers",
            name="Engines",
            marker=dict(color="#1565C0", size=7, opacity=0.75),
        )
    )
    min_value = float(min(true.min(), pred.min()))
    max_value = float(max(true.max(), pred.max()))
    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="Perfect prediction",
            line=dict(color="#424242", dash="dash"),
        )
    )
    _base_layout(fig, title)
    fig.update_xaxes(title="Actual RUL (cycles)")
    fig.update_yaxes(title="Predicted RUL (cycles)")
    return fig


def plot_training_history(
    history: dict, title: str = "RUL LSTM Training History"
) -> go.Figure:
    """Plot training and validation loss/MAE curves."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "loss" in history:
        fig.add_trace(go.Scatter(y=history["loss"], mode="lines", name="Train loss"))
    if "val_loss" in history:
        fig.add_trace(go.Scatter(y=history["val_loss"], mode="lines", name="Val loss"))
    if "mae" in history:
        fig.add_trace(
            go.Scatter(y=history["mae"], mode="lines", name="Train MAE"),
            secondary_y=True,
        )
    if "val_mae" in history:
        fig.add_trace(
            go.Scatter(y=history["val_mae"], mode="lines", name="Val MAE"),
            secondary_y=True,
        )
    _base_layout(fig, title)
    fig.update_xaxes(title="Epoch")
    fig.update_yaxes(title="Loss", secondary_y=False)
    fig.update_yaxes(title="MAE", secondary_y=True)
    return fig


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "RUL Prediction Error Distribution",
) -> go.Figure:
    """Histogram of prediction error."""
    errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    errors = errors[np.isfinite(errors)]
    fig = go.Figure(go.Histogram(x=errors, nbinsx=40, marker_color="#1565C0"))
    fig.add_vline(x=0, line_dash="dash", line_color="#424242")
    _base_layout(fig, title)
    fig.update_xaxes(title="Prediction error (predicted - actual)")
    fig.update_yaxes(title="Count")
    return fig


def plot_engine_degradation_trends(
    trend_df: pd.DataFrame,
    title: str = "Engine Degradation Trends",
) -> go.Figure:
    """Plot actual and predicted RUL across selected engine life cycles."""
    fig = go.Figure()
    for engine_id, engine_df in trend_df.groupby("engine_id"):
        engine_df = engine_df.sort_values("cycle")
        fig.add_trace(
            go.Scatter(
                x=engine_df["cycle"],
                y=engine_df["predicted_rul"],
                mode="lines",
                name=f"Engine {engine_id} predicted",
            )
        )
        if "actual_rul" in engine_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=engine_df["cycle"],
                    y=engine_df["actual_rul"],
                    mode="lines",
                    name=f"Engine {engine_id} actual",
                    line=dict(dash="dot"),
                    opacity=0.55,
                )
            )

    fig.add_hrect(y0=0, y1=15, fillcolor="#C62828", opacity=0.10, line_width=0)
    fig.add_hrect(y0=15, y1=60, fillcolor="#F9A825", opacity=0.08, line_width=0)
    _base_layout(fig, title)
    fig.update_xaxes(title="Cycle")
    fig.update_yaxes(title="RUL (cycles)")
    return fig


def plot_degradation_curve(
    df: pd.DataFrame,
    engine_id: int,
    sensor_cols: list[str],
    pred_col: str = "predicted_rul",
) -> go.Figure:
    """Create a per-engine RUL plus sensor trend panel."""
    engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")
    rows = max(1, len(sensor_cols) + 1)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04)
    fig.add_trace(
        go.Scatter(
            x=engine_df["cycle"],
            y=engine_df[pred_col],
            mode="lines",
            name="Predicted RUL",
        ),
        row=1,
        col=1,
    )

    palette = px.colors.qualitative.Safe
    for idx, sensor in enumerate(sensor_cols, start=2):
        if sensor not in engine_df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=engine_df["cycle"],
                y=engine_df[sensor],
                mode="lines",
                name=sensor,
                line=dict(color=palette[(idx - 2) % len(palette)]),
            ),
            row=idx,
            col=1,
        )

    _base_layout(fig, f"Engine {engine_id} Degradation Detail")
    fig.update_layout(height=max(420, rows * 180))
    fig.update_xaxes(title="Cycle", row=rows, col=1)
    return fig


def plot_failure_risk(
    prediction_df: pd.DataFrame,
    title: str = "Failure Risk Visualization",
) -> go.Figure:
    """Dashboard-ready bar chart of engines ranked by predicted RUL."""
    df = prediction_df.sort_values("predicted_rul").copy()
    colors = df["risk_category"].map(RISK_COLORS).fillna("#616161")
    fig = go.Figure(
        go.Bar(
            x=df["engine_id"].astype(str),
            y=df["predicted_rul"],
            marker_color=colors,
            text=df["risk_category"],
            textposition="outside",
            customdata=np.stack(
                [
                    df.get("confidence", pd.Series(np.nan, index=df.index)),
                    df.get("maintenance_urgency", ""),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Engine %{x}<br>Predicted RUL=%{y:.1f}<br>"
                "Confidence=%{customdata[0]:.0%}<br>Urgency=%{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=15, line_dash="dash", line_color=RISK_COLORS[RISK_HIGH])
    fig.add_hline(y=60, line_dash="dot", line_color=RISK_COLORS[RISK_MEDIUM])
    _base_layout(fig, title)
    fig.update_xaxes(title="Engine ID", type="category")
    fig.update_yaxes(title="Predicted RUL (cycles)")
    return fig


def plot_risk_distribution(
    risk_labels: np.ndarray,
    title: str = "RUL Risk Distribution",
) -> go.Figure:
    """Pie chart of low/medium/high risk engines."""
    counts = pd.Series(risk_labels).value_counts()
    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.45,
            marker=dict(
                colors=[RISK_COLORS.get(label, "#616161") for label in counts.index]
            ),
        )
    )
    return _base_layout(fig, title)


def plot_confidence_heatmap(
    engine_ids: np.ndarray,
    predicted_rul: np.ndarray,
    confidence: np.ndarray,
    risk_labels: np.ndarray,
) -> go.Figure:
    """Compact heatmap of engine RUL and confidence."""
    df = pd.DataFrame(
        {
            "engine_id": engine_ids,
            "predicted_rul": predicted_rul,
            "confidence": confidence,
            "risk": risk_labels,
        }
    ).sort_values("predicted_rul")
    fig = go.Figure(
        go.Heatmap(
            z=[df["predicted_rul"], df["confidence"] * 100],
            x=df["engine_id"].astype(str),
            y=["Predicted RUL", "Confidence %"],
            colorscale="RdYlGn",
            text=np.round(np.vstack([df["predicted_rul"], df["confidence"] * 100]), 1),
            texttemplate="%{text}",
        )
    )
    _base_layout(fig, "Engine RUL Confidence Heatmap")
    fig.update_xaxes(title="Engine ID")
    return fig


def plot_urgency_summary(urgency_labels: np.ndarray) -> go.Figure:
    """Pie chart of maintenance urgency categories."""
    counts = pd.Series(urgency_labels).value_counts()
    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.45,
            marker=dict(
                colors=[URGENCY_COLORS.get(label, "#616161") for label in counts.index]
            ),
        )
    )
    return _base_layout(fig, "Maintenance Urgency Summary")


def plot_rul_trend(
    df: pd.DataFrame,
    engine_id: int,
    pred_col: str = "predicted_rul",
    actual_col: str = "actual_rul",
    title: str | None = None,
) -> go.Figure:
    """Single-engine actual versus predicted RUL trend."""
    engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")
    fig = go.Figure()
    if actual_col in engine_df.columns:
        fig.add_trace(
            go.Scatter(
                x=engine_df["cycle"],
                y=engine_df[actual_col],
                mode="lines",
                name="Actual",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=engine_df["cycle"], y=engine_df[pred_col], mode="lines", name="Predicted"
        )
    )
    _base_layout(fig, title or f"Engine {engine_id} RUL Trend")
    fig.update_xaxes(title="Cycle")
    fig.update_yaxes(title="RUL (cycles)")
    return fig


def save_predictions_csv(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "rul_predictions.csv",
) -> Path:
    """Save prediction output for dashboards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df.to_csv(path, index=False)
    logger.info(f"Saved predictions to {path}")
    return path


def save_metrics_csv(
    metrics_list: list[dict],
    output_dir: Path,
    filename: str = "rul_metrics.csv",
) -> Path:
    """Save evaluation metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    pd.DataFrame(metrics_list).to_csv(path, index=False)
    logger.info(f"Saved metrics to {path}")
    return path


def save_figure(fig: go.Figure, output_dir: Path, filename: str) -> Path:
    """Save a Plotly figure as dashboard-ready HTML."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.write_html(str(path), include_plotlyjs="cdn")
    logger.info(f"Saved chart to {path}")
    return path
