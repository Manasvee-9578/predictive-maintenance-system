"""
Sentinel: turbofan engine health platform.

Launch from the project root:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

st.set_page_config(
    page_title="Sentinel | Engine Health Platform",
    page_icon=":material/flight:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard import core, theme  # noqa: E402
from dashboard.views import (  # noqa: E402
    alerts, explorer, home, lab, live, method, models, planner, predict, report,
)

theme.apply()
st.logo(str(Path(__file__).parent / "assets" / "logo.svg"), size="large")

nav = st.navigation(
    {
        "Monitor": [
            st.Page(home.render, title="Home", icon=":material/space_dashboard:", default=True),
            st.Page(explorer.render, title="Fleet explorer", icon=":material/travel_explore:", url_path="explorer"),
            st.Page(report.render, title="Engine report", icon=":material/description:", url_path="report"),
            st.Page(alerts.render, title="Alerts centre", icon=":material/notifications_active:", url_path="alerts"),
        ],
        "Analyse": [
            st.Page(live.render, title="Watch it wear", icon=":material/play_circle:", url_path="replay"),
            st.Page(lab.render, title="What-if simulator", icon=":material/tune:", url_path="whatif"),
            st.Page(planner.render, title="Plan repairs", icon=":material/build:", url_path="planner"),
            st.Page(predict.render, title="Use your data", icon=":material/upload_file:", url_path="upload"),
        ],
        "About": [
            st.Page(models.render, title="Accuracy", icon=":material/verified:", url_path="accuracy"),
            st.Page(method.render, title="How it works", icon=":material/lightbulb:", url_path="method"),
        ],
    },
    position="sidebar",
)

nav.run()
theme.footer()
