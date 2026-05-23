"""
╔══════════════════════════════════════════════════════════════╗
║   Anomaly Detection View                                    ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from configs.settings import Settings


def render_anomaly_view():
    """Render the anomaly detection visualization page."""
    st.markdown("# 🔍 Anomaly Detection")
    st.markdown("Identify unusual sensor readings across the engine fleet.")
    st.markdown("---")

    # ── Load Data ────────────────────────────────────────────
    processed_path = Settings.PROCESSED_DIR / "train_processed.csv"

    if not processed_path.exists():
        st.warning("⚠️ No processed data found. Run preprocessing first.")
        return

    df = pd.read_csv(processed_path)

    # ── Sensor Selection ─────────────────────────────────────
    sensor_cols = [c for c in df.columns if c.startswith("sensor_") and "roll" not in c]

    col1, col2 = st.columns(2)
    with col1:
        selected_sensor = st.selectbox("Select Sensor", sensor_cols)
    with col2:
        selected_engine = st.selectbox(
            "Select Engine", sorted(df["engine_id"].unique())
        )

    st.markdown("---")

    # ── Sensor Time Series ───────────────────────────────────
    st.subheader(f"📈 {selected_sensor} — Engine {selected_engine}")
    engine_data = df[df["engine_id"] == selected_engine]

    fig = px.line(
        engine_data,
        x="cycle",
        y=selected_sensor,
        title=f"{selected_sensor} Over Time (Engine {selected_engine})",
        labels={"cycle": "Cycle", selected_sensor: "Sensor Value"},
    )
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # ── Sensor Heatmap ───────────────────────────────────────
    st.subheader("🔥 Sensor Correlation Heatmap")
    corr = engine_data[sensor_cols].corr()

    fig2 = px.imshow(
        corr,
        title="Sensor Correlation Matrix",
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    fig2.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig2, use_container_width=True)
