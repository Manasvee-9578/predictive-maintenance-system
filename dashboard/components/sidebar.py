"""
╔══════════════════════════════════════════════════════════════╗
║   Sidebar Component — Navigation & Filters                  ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st


def render_sidebar_filters(engines: list, sensors: list) -> dict:
    """
    Render sidebar filter controls and return selected values.

    Args:
        engines: List of available engine IDs
        sensors: List of available sensor column names

    Returns:
        dict with filter selections
    """
    with st.sidebar:
        st.markdown("### 🎛️ Filters")

        selected_engine = st.selectbox("Engine", sorted(engines))

        selected_sensors = st.multiselect(
            "Sensors",
            sensors,
            default=sensors[:3] if len(sensors) >= 3 else sensors,
        )

        cycle_range = st.slider(
            "Cycle Range",
            min_value=1,
            max_value=500,
            value=(1, 300),
        )

        st.markdown("---")
        st.markdown("### ⚙️ Display Options")
        show_anomalies = st.checkbox("Highlight Anomalies", value=True)
        show_rolling = st.checkbox("Show Rolling Average", value=False)

    return {
        "engine": selected_engine,
        "sensors": selected_sensors,
        "cycle_range": cycle_range,
        "show_anomalies": show_anomalies,
        "show_rolling": show_rolling,
    }
