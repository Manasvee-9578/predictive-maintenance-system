"""
╔══════════════════════════════════════════════════════════════╗
║   Streamlit Dashboard — Main Application                    ║
╚══════════════════════════════════════════════════════════════╝

Multi-page interactive dashboard for the Predictive Maintenance
& RUL Forecasting Platform.

Launch:
    streamlit run dashboard/app.py
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from pathlib import Path

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance Platform",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Custom CSS ──────────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    """Main dashboard application."""

    # ── Sidebar Navigation ───────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔧 Predictive Maintenance")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            options=[
                "🏠 Overview",
                "🔍 Anomaly Detection",
                "📈 RUL Prediction",
                "📊 Model Comparison",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #888; font-size: 0.8em;'>"
            "v1.0.0 | Predictive Maintenance Platform"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Page Router ──────────────────────────────────────────
    if "Overview" in page:
        from dashboard.pages.overview import render_overview

        render_overview()

    elif "Anomaly" in page:
        from dashboard.pages.anomaly_view import render_anomaly_view

        render_anomaly_view()

    elif "RUL" in page:
        from dashboard.pages.rul_view import render_rul_view

        render_rul_view()

    elif "Model" in page:
        from dashboard.pages.model_comparison import render_model_comparison

        render_model_comparison()


if __name__ == "__main__":
    main()
