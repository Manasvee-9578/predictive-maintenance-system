"""
╔══════════════════════════════════════════════════════════════╗
║   Sidebar — Navigation & Global Filters                     ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
from datetime import datetime

PAGES = [
    ("🔴", "Live Monitoring"),
    ("🏠", "Executive Overview"),
    ("❤️", "Machine Health"),
    ("📡", "Sensor Analytics"),
    ("🔍", "Anomaly Detection"),
    ("📈", "RUL Forecasting"),
    ("🔧", "Maintenance"),
    ("💰", "Business Impact"),
]


def render_sidebar() -> str:
    """Render the sidebar navigation and return selected page name."""
    with st.sidebar:
        # ── Logo / Title ─────────────────────────────────────
        st.markdown(
            """
        <div style="text-align:center;padding:16px 0 8px 0">
            <div style="font-size:2.4rem;margin-bottom:4px">⚙️</div>
            <div style="font-size:1.1rem;font-weight:800;letter-spacing:-0.5px;
                        background:linear-gradient(135deg,#3b82f6,#06b6d4);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent">
                PREDICTIVE MAINTENANCE
            </div>
            <div style="font-size:0.7rem;color:#64748b;letter-spacing:2px;
                        text-transform:uppercase;margin-top:2px">
                AI Operations Center
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Live status ──────────────────────────────────────
        st.markdown(
            """
        <div style="display:flex;align-items:center;gap:8px;padding:8px 12px;
                    background:rgba(16,185,129,0.06);border-radius:8px;border:1px solid rgba(16,185,129,0.15);
                    margin-bottom:16px">
            <span class="pulse-dot pulse-green"></span>
            <span style="font-size:0.78rem;color:#10b981;font-weight:600">SYSTEM ONLINE</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # ── Navigation ───────────────────────────────────────
        st.markdown(
            """<div style="font-size:0.72rem;color:#475569;text-transform:uppercase;
                    letter-spacing:1.5px;font-weight:600;margin-bottom:8px;padding-left:4px">
                    Navigation</div>""",
            unsafe_allow_html=True,
        )

        page_options = [f"{icon}  {name}" for icon, name in PAGES]
        selected = st.radio(
            "Navigate",
            options=page_options,
            label_visibility="collapsed",
        )

        # ── Footer ───────────────────────────────────────────
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            f"""
        <div style="text-align:center;padding:8px 0">
            <div style="font-size:0.7rem;color:#475569">
                v2.0 Enterprise Edition
            </div>
            <div style="font-size:0.65rem;color:#334155;margin-top:4px">
                {datetime.now().strftime('%Y-%m-%d %H:%M')}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Extract page name (remove icon prefix)
    return selected.split("  ", 1)[1] if "  " in selected else selected
