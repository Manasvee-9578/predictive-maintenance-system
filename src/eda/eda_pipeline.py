"""
Industrial-Grade EDA Pipeline for NASA C-MAPSS FD001 Dataset.
Generates comprehensive visualizations saved to outputs/eda/.
"""

import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path

# ── Configuration ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nasa"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

SENSOR_DESCRIPTIONS = {
    "sensor_1": "Fan inlet temp (°R)",
    "sensor_2": "LPC outlet temp (°R)",
    "sensor_3": "HPC outlet temp (°R)",
    "sensor_4": "LPT outlet temp (°R)",
    "sensor_5": "Fan inlet press (psia)",
    "sensor_6": "Bypass-duct press (psia)",
    "sensor_7": "HPC outlet press (psia)",
    "sensor_8": "Phys fan speed (rpm)",
    "sensor_9": "Phys core speed (rpm)",
    "sensor_10": "Engine press ratio",
    "sensor_11": "HPC outlet static press (psia)",
    "sensor_12": "Fuel/air ratio",
    "sensor_13": "Corrected fan speed (rpm)",
    "sensor_14": "Corrected core speed (rpm)",
    "sensor_15": "Bypass ratio",
    "sensor_16": "Burner fuel-air ratio",
    "sensor_17": "Bleed enthalpy",
    "sensor_18": "Demanded fan speed (rpm)",
    "sensor_19": "Demanded corrected fan speed (rpm)",
    "sensor_20": "HPT coolant bleed",
    "sensor_21": "LPT coolant bleed",
}

# ── Premium Plot Theme ───────────────────────────────────────
DARK_BG = "#0d1117"
CARD_BG = "#161b22"
ACCENT = "#58a6ff"
ACCENT2 = "#f78166"
ACCENT3 = "#3fb950"
ACCENT4 = "#d2a8ff"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#21262d"

PALETTE = [
    "#58a6ff",
    "#f78166",
    "#3fb950",
    "#d2a8ff",
    "#f0883e",
    "#79c0ff",
    "#ffa657",
    "#7ee787",
    "#bc8cff",
    "#ff7b72",
]

plt.rcParams.update(
    {
        "figure.facecolor": DARK_BG,
        "axes.facecolor": CARD_BG,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    }
)

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": DARK_BG,
        "plot_bgcolor": CARD_BG,
        "font": {"color": TEXT_COLOR, "family": "Inter, sans-serif"},
        "xaxis": {"gridcolor": GRID_COLOR, "zerolinecolor": GRID_COLOR},
        "yaxis": {"gridcolor": GRID_COLOR, "zerolinecolor": GRID_COLOR},
        "colorway": PALETTE,
    }
}
pio.templates["industrial_dark"] = go.layout.Template(PLOTLY_TEMPLATE)
pio.templates.default = "industrial_dark"


def save_plotly(fig, name):
    fig.write_html(str(OUTPUT_DIR / f"{name}.html"), include_plotlyjs="cdn")
    try:
        fig.write_image(
            str(OUTPUT_DIR / f"{name}.png"), width=1400, height=800, scale=2
        )
    except Exception:
        pass  # kaleido not available, HTML is enough
    print(f"  >> Saved: {name}")


def save_matplotlib(fig, name):
    fig.savefig(
        str(OUTPUT_DIR / f"{name}.png"),
        dpi=200,
        bbox_inches="tight",
        facecolor=DARK_BG,
        edgecolor="none",
    )
    plt.close(fig)
    print(f"  ✅ Saved: {name}")


# ═══════════════════════════════════════════════════════════════
# STEP 1: DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════
def load_and_clean():
    print("\n" + "═" * 60)
    print("  STEP 1: DATA LOADING & CLEANING")
    print("═" * 60)

    train = pd.read_csv(
        DATA_DIR / "train_FD001.txt",
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python",
    )
    test = pd.read_csv(
        DATA_DIR / "test_FD001.txt",
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python",
    )
    rul = pd.read_csv(
        DATA_DIR / "RUL_FD001.txt", sep=r"\s+", header=None, names=["rul"]
    )
    rul.index += 1
    rul.index.name = "engine_id"

    print(
        f"  Train: {train.shape[0]:,} rows × {train.shape[1]} cols | {train['engine_id'].nunique()} engines"
    )
    print(
        f"  Test:  {test.shape[0]:,} rows × {test.shape[1]} cols | {test['engine_id'].nunique()} engines"
    )
    print(f"  RUL:   {len(rul)} engines")

    # Missing values
    missing_train = train.isnull().sum()
    missing_test = test.isnull().sum()
    print(
        f"\n  Missing values — Train: {missing_train.sum()} | Test: {missing_test.sum()}"
    )

    # Drop constant / near-zero variance columns
    std_vals = train.iloc[:, 2:].std()
    low_var_cols = std_vals[std_vals < 0.0001].index.tolist()
    print(f"  Near-zero variance columns ({len(low_var_cols)}): {low_var_cols}")

    # Add RUL to train
    max_cycles = train.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    train = train.merge(max_cycles, on="engine_id")
    train["rul"] = train["max_cycle"] - train["cycle"]
    train["rul_capped"] = train["rul"].clip(upper=125)
    train.drop("max_cycle", axis=1, inplace=True)

    # Useful sensors (excluding low variance)
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    useful_sensors = [c for c in sensor_cols if c not in low_var_cols]

    print(f"  Useful sensors ({len(useful_sensors)}): {useful_sensors}")

    # Save summary report
    summary = []
    summary.append("NASA C-MAPSS FD001 — DATA SUMMARY REPORT")
    summary.append("=" * 50)
    summary.append(f"Training samples:   {len(train):,}")
    summary.append(f"Test samples:       {len(test):,}")
    summary.append(f"Training engines:   {train['engine_id'].nunique()}")
    summary.append(f"Test engines:       {test['engine_id'].nunique()}")
    summary.append(f"Features:           {len(COLUMN_NAMES)}")
    summary.append(f"Missing values:     {missing_train.sum() + missing_test.sum()}")
    summary.append(f"Low-var columns:    {low_var_cols}")
    summary.append(f"Useful sensors:     {useful_sensors}")
    summary.append("\nDESCRIPTIVE STATISTICS")
    summary.append(train[useful_sensors].describe().to_string())

    with open(OUTPUT_DIR / "data_summary_report.txt", "w") as f:
        f.write("\n".join(summary))
    print("  ✅ Saved: data_summary_report.txt")

    return train, test, rul, useful_sensors, low_var_cols


# ═══════════════════════════════════════════════════════════════
# STEP 2: CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════
def plot_correlation_heatmap(train, useful_sensors):
    print("\n" + "═" * 60)
    print("  STEP 2: CORRELATION HEATMAP")
    print("═" * 60)

    corr = train[useful_sensors + ["rul_capped"]].corr()

    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(250, 15, s=75, l=40, center="dark", as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        linecolor=GRID_COLOR,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title(
        "Sensor Correlation Matrix with RUL", fontsize=18, fontweight="bold", pad=20
    )
    save_matplotlib(fig, "01_correlation_heatmap")

    # RUL correlation bar chart
    rul_corr = corr["rul_capped"].drop("rul_capped").sort_values()
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    colors = [ACCENT3 if v > 0 else ACCENT2 for v in rul_corr.values]
    bars = ax2.barh(
        rul_corr.index, rul_corr.values, color=colors, edgecolor="none", height=0.7
    )
    ax2.set_xlabel("Correlation with RUL", fontsize=13)
    ax2.set_title(
        "Sensor Correlation with Remaining Useful Life",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax2.axvline(x=0, color=TEXT_COLOR, linewidth=0.8, alpha=0.3)
    for bar, val in zip(bars, rul_corr.values):
        ax2.text(
            val + (0.01 if val > 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left" if val > 0 else "right",
            fontsize=9,
            color=TEXT_COLOR,
        )
    ax2.grid(axis="x", alpha=0.2)
    save_matplotlib(fig2, "02_rul_correlation_ranking")


# ═══════════════════════════════════════════════════════════════
# STEP 3: ENGINE LIFECYCLE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════
def plot_lifecycle_distribution(train, rul):
    print("\n" + "═" * 60)
    print("  STEP 3: ENGINE LIFECYCLE DISTRIBUTION")
    print("═" * 60)

    max_cycles = train.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "total_cycles"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Engine Lifespan Distribution (Train)",
            "Ground Truth RUL Distribution (Test)",
        ),
    )

    fig.add_trace(
        go.Histogram(
            x=max_cycles["total_cycles"],
            nbinsx=30,
            marker=dict(color=ACCENT, line=dict(color=DARK_BG, width=1)),
            name="Train Lifespan",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=rul["rul"],
            nbinsx=25,
            marker=dict(color=ACCENT2, line=dict(color=DARK_BG, width=1)),
            name="Test RUL",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=500,
        title_text="Engine Lifecycle Analysis",
        title_font_size=20,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Total Cycles", row=1, col=1)
    fig.update_xaxes(title_text="Remaining Useful Life", row=1, col=2)
    save_plotly(fig, "03_lifecycle_distribution")


# ═══════════════════════════════════════════════════════════════
# STEP 4: SENSOR TREND ANALYSIS
# ═══════════════════════════════════════════════════════════════
def plot_sensor_trends(train, useful_sensors):
    print("\n" + "═" * 60)
    print("  STEP 4: SENSOR TREND ANALYSIS")
    print("═" * 60)

    sample_engines = sorted(train["engine_id"].unique())[:6]
    n_sensors = min(len(useful_sensors), 14)
    cols = 2
    rows = (n_sensors + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3.2))
    axes = axes.flatten()

    for idx, sensor in enumerate(useful_sensors[:n_sensors]):
        ax = axes[idx]
        for eid in sample_engines:
            edata = train[train["engine_id"] == eid]
            ax.plot(edata["cycle"], edata[sensor], alpha=0.7, linewidth=1.0)
        desc = SENSOR_DESCRIPTIONS.get(sensor, sensor)
        ax.set_title(f"{sensor}: {desc}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Cycle", fontsize=9)
        ax.grid(True, alpha=0.15)

    for idx in range(n_sensors, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Sensor Readings Over Engine Lifecycle (6 Sample Engines)",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_matplotlib(fig, "04_sensor_trends")


# ═══════════════════════════════════════════════════════════════
# STEP 5: ENGINE DEGRADATION VISUALIZATION
# ═══════════════════════════════════════════════════════════════
def plot_degradation(train, useful_sensors):
    print("\n" + "═" * 60)
    print("  STEP 5: ENGINE DEGRADATION VISUALIZATION")
    print("═" * 60)

    # Normalize cycle to percentage of life
    train_copy = train.copy()
    max_c = train_copy.groupby("engine_id")["cycle"].transform("max")
    train_copy["life_pct"] = train_copy["cycle"] / max_c * 100

    top_sensors = [
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_7",
        "sensor_11",
        "sensor_12",
        "sensor_15",
        "sensor_21",
    ]
    top_sensors = [s for s in top_sensors if s in useful_sensors][:6]

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[SENSOR_DESCRIPTIONS.get(s, s) for s in top_sensors],
    )

    engines_sample = sorted(train["engine_id"].unique())[:15]
    for idx, sensor in enumerate(top_sensors):
        r, c = idx // 3 + 1, idx % 3 + 1
        for eid in engines_sample:
            ed = train_copy[train_copy["engine_id"] == eid]
            fig.add_trace(
                go.Scatter(
                    x=ed["life_pct"],
                    y=ed[sensor],
                    mode="lines",
                    line=dict(width=1, color=PALETTE[eid % len(PALETTE)]),
                    opacity=0.5,
                    showlegend=False,
                ),
                row=r,
                col=c,
            )
        fig.update_xaxes(title_text="Life %", row=r, col=c)

    fig.update_layout(
        height=700,
        title_text="Engine Degradation Patterns (% of Life)",
        title_font_size=20,
    )
    save_plotly(fig, "05_engine_degradation")


# ═══════════════════════════════════════════════════════════════
# STEP 6: CYCLE-WISE SENSOR BEHAVIOR
# ═══════════════════════════════════════════════════════════════
def plot_cyclewise_behavior(train, useful_sensors):
    print("\n" + "═" * 60)
    print("  STEP 6: CYCLE-WISE SENSOR BEHAVIOR")
    print("═" * 60)

    key_sensors = [
        s
        for s in [
            "sensor_2",
            "sensor_3",
            "sensor_4",
            "sensor_7",
            "sensor_11",
            "sensor_21",
        ]
        if s in useful_sensors
    ][:4]

    # Mean and std per cycle across all engines
    cycle_stats = train.groupby("cycle")[key_sensors].agg(["mean", "std"]).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, sensor in enumerate(key_sensors):
        ax = axes[idx]
        mean = cycle_stats[(sensor, "mean")]
        std = cycle_stats[(sensor, "std")]
        cycles = cycle_stats["cycle"]

        ax.plot(cycles, mean, color=ACCENT, linewidth=2, label="Fleet Mean")
        ax.fill_between(cycles, mean - std, mean + std, alpha=0.2, color=ACCENT)
        ax.fill_between(
            cycles, mean - 2 * std, mean + 2 * std, alpha=0.08, color=ACCENT
        )
        desc = SENSOR_DESCRIPTIONS.get(sensor, sensor)
        ax.set_title(f"{sensor}: {desc}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Cycle")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)

    fig.suptitle(
        "Cycle-Wise Sensor Behavior (Fleet Mean ± σ/2σ)",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_matplotlib(fig, "06_cyclewise_sensor_behavior")


# ═══════════════════════════════════════════════════════════════
# STEP 7: ANOMALY PATTERN VISUALIZATION
# ═══════════════════════════════════════════════════════════════
def plot_anomaly_patterns(train, useful_sensors):
    print("\n" + "═" * 60)
    print("  STEP 7: ANOMALY PATTERN VISUALIZATION")
    print("═" * 60)

    key = [
        s
        for s in ["sensor_7", "sensor_11", "sensor_4", "sensor_3"]
        if s in useful_sensors
    ][:2]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for idx, sensor in enumerate(key):
        ax = axes[idx]
        mean = train[sensor].mean()
        std = train[sensor].std()
        z_scores = np.abs((train[sensor] - mean) / std)
        anomalies = z_scores > 3

        normal = train[~anomalies]
        abnormal = train[anomalies]

        ax.scatter(
            normal["cycle"],
            normal[sensor],
            s=1,
            alpha=0.15,
            color=ACCENT,
            label="Normal",
        )
        ax.scatter(
            abnormal["cycle"],
            abnormal[sensor],
            s=8,
            alpha=0.9,
            color="#ff4444",
            label=f"Anomaly (Z>3): {anomalies.sum()}",
            zorder=5,
        )
        ax.axhline(
            y=mean + 3 * std, color=ACCENT2, linestyle="--", alpha=0.7, label="+3σ"
        )
        ax.axhline(
            y=mean - 3 * std, color=ACCENT2, linestyle="--", alpha=0.7, label="-3σ"
        )
        ax.set_title(f"Anomaly Detection: {sensor}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Cycle")
        ax.set_ylabel(sensor)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.15)

    fig.suptitle(
        "Statistical Anomaly Detection (Z-Score Method)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_matplotlib(fig, "07_anomaly_patterns")


# ═══════════════════════════════════════════════════════════════
# STEP 8: SENSOR IMPORTANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════
def plot_sensor_importance(train, useful_sensors):
    print("\n" + "═" * 60)
    print("  STEP 8: SENSOR IMPORTANCE ANALYSIS")
    print("═" * 60)

    from sklearn.ensemble import RandomForestRegressor

    X = train[useful_sensors].values
    y = train["rul_capped"].values

    # Sub-sample for speed
    np.random.seed(42)
    idx = np.random.choice(len(X), min(10000, len(X)), replace=False)
    X_sub, y_sub = X[idx], y[idx]

    rf = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_sub, y_sub)

    importances = pd.Series(rf.feature_importances_, index=useful_sensors).sort_values()

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
    bars = ax.barh(
        importances.index,
        importances.values,
        color=colors,
        edgecolor="none",
        height=0.7,
    )
    ax.set_xlabel("Feature Importance", fontsize=13)
    ax.set_title(
        "Sensor Importance for RUL Prediction (Random Forest)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    for bar, val in zip(bars, importances.values):
        ax.text(
            val + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
            color=TEXT_COLOR,
        )
    ax.grid(axis="x", alpha=0.2)
    save_matplotlib(fig, "08_sensor_importance")

    # Save top sensors
    top5 = importances.tail(5).index.tolist()[::-1]
    print(f"  🏆 Top 5 most important sensors: {top5}")
    return top5


# ═══════════════════════════════════════════════════════════════
# STEP 9: RUL DISTRIBUTION BY HEALTH STATE
# ═══════════════════════════════════════════════════════════════
def plot_rul_health_states(train):
    print("\n" + "═" * 60)
    print("  STEP 9: RUL DISTRIBUTION & HEALTH STATES")
    print("═" * 60)

    train_c = train.copy()
    train_c["health"] = pd.cut(
        train_c["rul_capped"],
        bins=[0, 30, 60, 90, 125],
        labels=["Critical", "Warning", "Fair", "Healthy"],
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # RUL distribution
    axes[0].hist(
        train_c["rul_capped"], bins=50, color=ACCENT, edgecolor=DARK_BG, alpha=0.9
    )
    axes[0].set_title(
        "RUL Distribution (Capped at 125)", fontsize=13, fontweight="bold"
    )
    axes[0].set_xlabel("RUL")
    axes[0].set_ylabel("Count")
    axes[0].axvline(
        x=30, color="#ff4444", linestyle="--", alpha=0.7, label="Critical threshold"
    )
    axes[0].legend()

    # Health state pie
    health_counts = train_c["health"].value_counts()
    colors_pie = ["#ff4444", "#ffa657", "#58a6ff", "#3fb950"]
    axes[1].pie(
        health_counts,
        labels=health_counts.index,
        colors=colors_pie,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": TEXT_COLOR, "fontsize": 11},
    )
    axes[1].set_title("Health State Distribution", fontsize=13, fontweight="bold")

    # Engine count per health at final cycle
    final = train_c.groupby("engine_id").last().reset_index()
    fc = (
        final["health"]
        .value_counts()
        .reindex(["Critical", "Warning", "Fair", "Healthy"])
    )
    axes[2].bar(fc.index, fc.values, color=colors_pie, edgecolor="none")
    axes[2].set_title("Final Health State per Engine", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Engine Count")
    for i, v in enumerate(fc.values):
        axes[2].text(i, v + 0.5, str(v), ha="center", fontsize=12, fontweight="bold")

    fig.suptitle("RUL & Engine Health Analysis", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_matplotlib(fig, "09_rul_health_states")


# ═══════════════════════════════════════════════════════════════
# STEP 10: SENSOR DISTRIBUTION BOXPLOTS
# ═══════════════════════════════════════════════════════════════
def plot_sensor_distributions(train, useful_sensors):
    print("\n" + "═" * 60)
    print("  STEP 10: SENSOR DISTRIBUTION ANALYSIS")
    print("═" * 60)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(train[useful_sensors]), columns=useful_sensors
    )

    fig, ax = plt.subplots(figsize=(16, 8))
    bp = ax.boxplot(
        [scaled[c].values for c in useful_sensors],
        labels=useful_sensors,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#ffffff", linewidth=2),
    )
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=PALETTE[i % len(PALETTE)], alpha=0.7)
    ax.set_title(
        "Normalized Sensor Value Distributions", fontsize=16, fontweight="bold", pad=15
    )
    ax.set_ylabel("Normalized Value [0-1]")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2)
    save_matplotlib(fig, "10_sensor_distributions")


# ═══════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 60)
    print("   NASA C-MAPSS FD001 -- Industrial EDA Pipeline")
    print("=" * 60)
    print(f"  Output directory: {OUTPUT_DIR}\n")

    train, test, rul, useful_sensors, low_var = load_and_clean()
    plot_correlation_heatmap(train, useful_sensors)
    plot_lifecycle_distribution(train, rul)
    plot_sensor_trends(train, useful_sensors)
    plot_degradation(train, useful_sensors)
    plot_cyclewise_behavior(train, useful_sensors)
    plot_anomaly_patterns(train, useful_sensors)
    top5 = plot_sensor_importance(train, useful_sensors)
    plot_rul_health_states(train)
    plot_sensor_distributions(train, useful_sensors)

    print("\n" + "=" * 60)
    print("   EDA PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\n  📁 All outputs saved to: {OUTPUT_DIR}")
    print(f"  📊 Total visualizations: 10")
    print(f"  🏆 Top sensors: {top5}\n")


if __name__ == "__main__":
    main()
