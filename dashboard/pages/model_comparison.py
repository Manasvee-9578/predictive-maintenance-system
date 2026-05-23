"""
╔══════════════════════════════════════════════════════════════╗
║   Model Comparison View                                     ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from configs.settings import Settings


def render_model_comparison():
    """Render the model performance comparison page."""
    st.markdown("# 📊 Model Performance Comparison")
    st.markdown("Compare evaluation metrics across all trained models.")
    st.markdown("---")

    # ── Load Results ─────────────────────────────────────────
    results_path = Settings.OUTPUT_DIR / "reports" / "model_comparison.csv"

    if not results_path.exists():
        st.warning("⚠️ No evaluation results found. Train and evaluate models first:")
        st.code("python main.py", language="bash")

        # Show placeholder comparison
        st.subheader("📋 Expected Model Comparison")
        placeholder = pd.DataFrame(
            {
                "Model": [
                    "LSTM",
                    "BiLSTM",
                    "Random Forest",
                    "Gradient Boosting",
                    "SVR",
                ],
                "RMSE": ["—"] * 5,
                "MAE": ["—"] * 5,
                "R²": ["—"] * 5,
                "Score": ["—"] * 5,
            }
        )
        st.table(placeholder)
        return

    results_df = pd.read_csv(results_path)

    # ── Metrics Table ────────────────────────────────────────
    st.subheader("📋 Evaluation Metrics")
    st.dataframe(
        results_df.style.highlight_min(
            subset=["rmse", "mae"], color="#2ecc71"
        ).highlight_max(subset=["r2"], color="#2ecc71"),
        use_container_width=True,
    )

    # ── Bar Charts ───────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        fig_rmse = px.bar(
            results_df,
            x="model",
            y="rmse",
            title="RMSE Comparison (Lower = Better)",
            color="model",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_rmse.update_layout(template="plotly_dark", showlegend=False, height=400)
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col2:
        fig_r2 = px.bar(
            results_df,
            x="model",
            y="r2",
            title="R² Score Comparison (Higher = Better)",
            color="model",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_r2.update_layout(template="plotly_dark", showlegend=False, height=400)
        st.plotly_chart(fig_r2, use_container_width=True)
