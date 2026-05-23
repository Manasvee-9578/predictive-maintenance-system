"""
Metric Cards — KPI Display Components
"""

import streamlit as st


def render_metric_card(title, value, icon="📊", color="#636EFA"):
    st.markdown(
        f"""
    <div style="background:linear-gradient(135deg,rgba(99,110,250,0.1),rgba(99,110,250,0.05));
    border:1px solid rgba(99,110,250,0.2);border-radius:12px;padding:20px;text-align:center;">
    <div style="font-size:2em;">{icon}</div>
    <div style="font-size:0.85em;color:#aaa;margin:8px 0 4px;">{title}</div>
    <div style="font-size:1.8em;font-weight:700;color:{color};">{value}</div>
    </div>""",
        unsafe_allow_html=True,
    )
