"""
╔══════════════════════════════════════════════════════════════╗
║   Overview Page — Fleet Health Dashboard                    ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from configs.settings import Settings


def render_overview():
    """Render the fleet health overview page."""
    st.markdown("# 🏠 Fleet Health Overview")
    st.markdown("Real-time monitoring of turbofan engine fleet status.")
    st.markdown("---")

    # ── Load Data ────────────────────────────────────────────
    processed_path = Settings.PROCESSED_DIR / "train_processed.csv"

    if not processed_path.exists():
        st.warning("⚠️ No processed data found. Run the preprocessing pipeline first:")
        st.code("python main.py --preprocess", language="bash")
        return

    df = pd.read_csv(processed_path)

    # ── KPI Metrics Row ──────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🔢 Total Engines", f"{df['engine_id'].nunique()}")
    with col2:
        st.metric("📊 Total Records", f"{len(df):,}")
    with col3:
        avg_rul = df.groupby("engine_id")["rul"].min().mean()
        st.metric("⏳ Avg Final RUL", f"{avg_rul:.1f} cycles")
    with col4:
        critical = (df.groupby("engine_id")["rul"].min() < 30).sum()
        st.metric("🚨 Critical Engines", f"{critical}")

    st.markdown("---")

    # ── Engine Lifecycle Distribution ────────────────────────
    st.subheader("📊 Engine Lifecycle Distribution")
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "total_cycles"]

    fig = px.histogram(
        max_cycles,
        x="total_cycles",
        nbins=30,
        title="Distribution of Engine Lifespans",
        labels={"total_cycles": "Total Cycles", "count": "Number of Engines"},
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # ── RUL Distribution ─────────────────────────────────────
    st.subheader("⏳ RUL Distribution Across Fleet")
    final_rul = df.groupby("engine_id")["rul"].min().reset_index()

    fig2 = px.histogram(
        final_rul,
        x="rul",
        nbins=25,
        title="Final RUL Distribution (Last Cycle per Engine)",
        labels={"rul": "Remaining Useful Life", "count": "Engines"},
        color_discrete_sequence=["#EF553B"],
    )
    fig2.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig2, use_container_width=True)
