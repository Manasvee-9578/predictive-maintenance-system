"""
╔══════════════════════════════════════════════════════════════╗
║   RUL Prediction View                                       ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from configs.settings import Settings


def render_rul_view():
    """Render the RUL prediction results page."""
    st.markdown("# 📈 RUL Prediction Results")
    st.markdown("Remaining Useful Life predictions for test engines.")
    st.markdown("---")

    # ── Load Data ────────────────────────────────────────────
    processed_path = Settings.PROCESSED_DIR / "train_processed.csv"

    if not processed_path.exists():
        st.warning("⚠️ No processed data found. Run the pipeline first.")
        return

    df = pd.read_csv(processed_path)

    # ── RUL Degradation Curves ───────────────────────────────
    st.subheader("📉 Engine Degradation Curves")

    selected_engines = st.multiselect(
        "Select Engines to Compare",
        sorted(df["engine_id"].unique()),
        default=sorted(df["engine_id"].unique())[:5],
    )

    if selected_engines:
        filtered = df[df["engine_id"].isin(selected_engines)]

        fig = px.line(
            filtered,
            x="cycle",
            y="rul",
            color="engine_id",
            title="RUL Degradation Over Time",
            labels={
                "cycle": "Cycle",
                "rul": "Remaining Useful Life",
                "engine_id": "Engine",
            },
        )
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

    # ── Health Score Gauge ───────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Engine Health Scores")

    final_rul = df.groupby("engine_id")["rul"].min().reset_index()
    final_rul["health_pct"] = (final_rul["rul"] / Settings.MAX_RUL * 100).clip(0, 100)

    # Color-coded health categories
    final_rul["status"] = pd.cut(
        final_rul["health_pct"],
        bins=[0, 20, 50, 80, 100],
        labels=["🔴 Critical", "🟠 Warning", "🟡 Fair", "🟢 Healthy"],
    )

    col1, col2, col3, col4 = st.columns(4)
    status_counts = final_rul["status"].value_counts()

    with col1:
        st.metric("🟢 Healthy", status_counts.get("🟢 Healthy", 0))
    with col2:
        st.metric("🟡 Fair", status_counts.get("🟡 Fair", 0))
    with col3:
        st.metric("🟠 Warning", status_counts.get("🟠 Warning", 0))
    with col4:
        st.metric("🔴 Critical", status_counts.get("🔴 Critical", 0))
