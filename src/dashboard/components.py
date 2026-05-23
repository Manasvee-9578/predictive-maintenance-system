"""
╔══════════════════════════════════════════════════════════════╗
║   Components — Reusable Dashboard UI Elements               ║
╚══════════════════════════════════════════════════════════════╝

Glassmorphism KPI cards, alert panels, recommendation cards,
section headers, and badge components.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
from src.dashboard.utils import (
    get_priority_color,
    get_health_color,
    get_risk_color,
    fmt_number,
)

# ═══════════════════════════════════════════════════════════════
#  KPI Hero Cards
# ═══════════════════════════════════════════════════════════════


def render_kpi_card(
    icon: str, label: str, value: str, color: str = "#3b82f6", delta: str = ""
):
    """Render a single glassmorphism KPI hero card."""
    delta_html = ""
    if delta:
        delta_color = (
            "#10b981" if delta.startswith("+") or delta.startswith("↑") else "#ef4444"
        )
        delta_html = f'<div class="kpi-delta" style="color:{delta_color}">{delta}</div>'

    st.markdown(
        f"""
    <div class="kpi-hero">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="background:linear-gradient(135deg,{color},
            {'#06b6d4' if color == '#3b82f6' else '#f59e0b' if color == '#10b981' else '#fb7185' if color == '#ef4444' else '#c084fc'})
            ;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">{value}</div>
        {delta_html}
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_kpi_row(metrics: list[dict]):
    """Render a row of KPI cards. Each dict has: icon, label, value, color, delta."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            render_kpi_card(
                icon=m.get("icon", "📊"),
                label=m.get("label", ""),
                value=m.get("value", "—"),
                color=m.get("color", "#3b82f6"),
                delta=m.get("delta", ""),
            )


# ═══════════════════════════════════════════════════════════════
#  Section Header
# ═══════════════════════════════════════════════════════════════


def render_section_header(icon: str, title: str, subtitle: str = ""):
    """Render a styled section header with icon."""
    sub_html = (
        f'<span style="color:#64748b;font-size:0.85rem;margin-left:12px">{subtitle}</span>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
    <div class="section-header">
        <span class="section-icon">{icon}</span>
        <span class="section-title">{title}</span>
        {sub_html}
    </div>
    """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
#  Alert Cards
# ═══════════════════════════════════════════════════════════════


def render_alert_card(
    engine_id: int, priority: str, action: str, risk: float, rul: float, health: float
):
    """Render an alert card for a single engine."""
    pcolor = get_priority_color(priority)
    css_class = f"alert-{priority.lower()}"
    badge_class = {
        "Critical": "badge-critical",
        "High": "badge-high",
        "Medium": "badge-warning",
        "Low": "badge-normal",
    }.get(priority, "badge-normal")

    st.markdown(
        f"""
    <div class="alert-card {css_class}">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <div style="font-weight:700;font-size:1.05rem;color:#f1f5f9">
                ⚙️ Engine {engine_id}
            </div>
            <span class="status-badge {badge_class}">{priority}</span>
        </div>
        <div style="color:#94a3b8;font-size:0.88rem;margin-bottom:8px">{action}</div>
        <div style="display:flex;gap:24px;font-size:0.82rem">
            <span style="color:{get_risk_color(risk)}">◉ Risk: {risk:.1f}%</span>
            <span style="color:#94a3b8">⏳ RUL: {rul:.0f} cycles</span>
            <span style="color:{get_health_color('Critical' if health<50 else 'Warning' if health<80 else 'Normal')}">
                ❤️ Health: {health:.1f}
            </span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
#  Recommendation Card
# ═══════════════════════════════════════════════════════════════


def render_recommendation_card(row: pd.Series):
    """Render a detailed recommendation card for an engine."""
    eid = int(row.get("engine_id", 0))
    priority = row.get("maintenance_priority", "Low")
    action = row.get("recommended_action", "Continue Monitoring")
    risk = float(row.get("failure_risk_pct", 0))
    rul = float(row.get("predicted_rul", 125))
    health = float(row.get("health_score", 100))
    reliability = float(row.get("equipment_reliability", 0))
    urgency = float(row.get("urgency_score", 0))

    pcolor = get_priority_color(priority)
    badge_class = {
        "Critical": "badge-critical",
        "High": "badge-high",
        "Medium": "badge-warning",
        "Low": "badge-normal",
    }.get(priority, "badge-normal")

    # Time window
    if rul <= 15:
        tw = "Immediate (0-2 days)"
    elif rul <= 30:
        tw = "Urgent (3-7 days)"
    elif rul <= 60:
        tw = "Short-term (1-3 weeks)"
    elif rul <= 100:
        tw = "Medium-term (1-2 months)"
    else:
        tw = "Long-term (3+ months)"

    st.markdown(
        f"""
    <div class="glass-card" style="border-left:4px solid {pcolor}">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px">
            <div>
                <div style="font-weight:800;font-size:1.15rem;color:#f1f5f9;margin-bottom:4px">
                    ⚙️ Engine {eid}
                </div>
                <div style="font-size:0.82rem;color:#64748b">Maintenance Window: {tw}</div>
            </div>
            <span class="status-badge {badge_class}">{priority}</span>
        </div>

        <div style="background:rgba(59,130,246,0.06);border-radius:10px;padding:14px;margin-bottom:16px;
                    border:1px solid rgba(59,130,246,0.1)">
            <div style="font-weight:600;color:#3b82f6;font-size:0.82rem;margin-bottom:6px">
                🔧 RECOMMENDED ACTION
            </div>
            <div style="color:#e2e8f0;font-size:0.95rem">{action}</div>
        </div>

        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;text-align:center">
            <div>
                <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:0.5px">Risk</div>
                <div style="font-size:1.2rem;font-weight:700;color:{get_risk_color(risk)}">{risk:.1f}%</div>
            </div>
            <div>
                <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:0.5px">RUL</div>
                <div style="font-size:1.2rem;font-weight:700;color:#e2e8f0">{rul:.0f}</div>
            </div>
            <div>
                <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:0.5px">Health</div>
                <div style="font-size:1.2rem;font-weight:700;color:{get_health_color('Critical' if health<50 else 'Warning' if health<80 else 'Normal')}">{health:.1f}</div>
            </div>
            <div>
                <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:0.5px">Reliability</div>
                <div style="font-size:1.2rem;font-weight:700;color:#8b5cf6">{reliability:.1f}%</div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
#  Status Indicator
# ═══════════════════════════════════════════════════════════════


def render_system_status():
    """Render the live system status indicator."""
    st.markdown(
        """
    <div style="display:flex;align-items:center;gap:8px;margin:8px 0 16px 0">
        <span class="pulse-dot pulse-green"></span>
        <span style="font-size:0.82rem;color:#94a3b8;font-weight:500">System Online</span>
        <span style="font-size:0.72rem;color:#475569;margin-left:auto">
            AI Monitoring Active
        </span>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
#  Top Animated Bar
# ═══════════════════════════════════════════════════════════════


def render_top_bar():
    """Render the animated gradient top bar."""
    st.markdown('<div class="top-bar"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  Data Table with styling
# ═══════════════════════════════════════════════════════════════


def render_styled_dataframe(df: pd.DataFrame, height: int = 400):
    """Render a styled DataFrame."""
    st.dataframe(
        df,
        use_container_width=True,
        height=height,
        hide_index=True,
    )
