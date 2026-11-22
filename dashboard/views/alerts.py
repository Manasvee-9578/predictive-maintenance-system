"""Alerts centre: a working queue. Acknowledge alerts and assign them to a crew."""

import pandas as pd
import streamlit as st

from dashboard import core, theme

CREWS = ["Unassigned", "Crew A", "Crew B", "Crew C"]


def render():
    b = core.load_bundle()
    theme.header("Alerts centre", "Every engine that needs a human decision. Tick an alert once someone has "
                 "dealt with it and assign it to a crew.", tag="Monitor")

    alerts = core.build_alerts(b.fleet)
    ss = st.session_state
    ss.setdefault("acked", set())
    ss.setdefault("crew", {})

    n = {s: int((alerts["severity"] == s).sum()) for s in core.SEV_ORDER}
    open_n = int((~alerts["engine"].isin(ss["acked"])).sum())
    theme.kpi_cards(
        [
            ("Critical", f"{n['Critical']}", "act immediately", core.SEV_COLOR["Critical"]),
            ("High", f"{n['High']}", "book a workshop slot", core.SEV_COLOR["High"]),
            ("Medium", f"{n['Medium']}", "keep monitoring", core.SEV_COLOR["Medium"]),
            ("Still open", f"{open_n}", f"{len(ss['acked'])} acknowledged", theme.NAVY),
        ]
    )

    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        sev = c1.multiselect("Severity", core.SEV_ORDER, default=core.SEV_ORDER)
        only_open = c2.toggle("Hide acknowledged", value=False)
        d = alerts[alerts["severity"].isin(sev)].copy()
        d["done"] = d["engine"].isin(ss["acked"])
        d["crew"] = d["engine"].map(lambda e: ss["crew"].get(e, "Unassigned"))
        if only_open:
            d = d[~d["done"]]
        view = d[["done", "severity", "engine", "flights_left", "message", "crew"]].rename(
            columns={"done": "Done", "severity": "Severity", "engine": "Engine", "flights_left": "Flights left",
                     "message": "What to do", "crew": "Crew"}
        )
        edited = st.data_editor(
            view, hide_index=True, use_container_width=True, height=470, key="alert_editor",
            disabled=["Severity", "Engine", "Flights left", "What to do"],
            column_config={
                "Done": st.column_config.CheckboxColumn(width="small"),
                "Crew": st.column_config.SelectboxColumn(options=CREWS, width="small"),
                "Engine": st.column_config.NumberColumn(format="%d", width="small"),
                "Flights left": st.column_config.NumberColumn(format="%d", width="small"),
            },
        )
        # persist edits for the rows currently shown
        for _, row in edited.iterrows():
            e = int(row["Engine"])
            (ss["acked"].add if row["Done"] else ss["acked"].discard)(e)
            ss["crew"][e] = row["Crew"]

        st.download_button("Export alerts (CSV)", edited.to_csv(index=False).encode(), "alerts.csv", "text/csv")
    st.caption("Changes are kept for this browser session.")
