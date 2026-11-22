"""Repair planner: who goes into the workshop, and when."""

import plotly.graph_objects as go
import streamlit as st

from dashboard import core, theme

WEEKS = 4


def money(k: float) -> str:
    return f"${k / 1000:.1f} million" if abs(k) >= 1000 else f"${k:,.0f}k"


def render():
    b = core.load_bundle()
    theme.header(
        "Plan the repairs",
        "A failure in service is far more expensive than a planned repair, but the workshop "
        "has limited space. Tell it your costs and capacity and it picks which engines to repair first.",
    )

    c1, c2, c3 = st.columns(3, gap="large")
    slots = c1.slider("Engines the workshop can take (4 weeks)", 4, 60, 32)
    fail = c2.number_input("Cost of a failure in service ($ thousand)", 50.0, 2000.0, 480.0, 10.0)
    repair = c3.number_input("Cost of a planned repair ($ thousand)", 5.0, 300.0, 45.0, 5.0)
    with st.expander("More settings"):
        horizon = st.slider("How far ahead to look (flights)", 15, 100, 40)
        waste = st.number_input("Cost of repairing too early, per flight wasted ($ thousand)", 0.0, 5.0, 0.9, 0.1)

    bays = -(-slots // WEEKS)
    cm = core.CostModel(repair, fail, waste, horizon, bays, WEEKS, 1e9)
    p = core.plan(b.fleet, cm, b.sigma)
    pol = core.policy_costs(p, cm)
    bt = core.backtest(p, cm)
    saving = pol.iloc[0, 1] - pol.iloc[2, 1]
    n_sel = int(p["selected"].sum())

    theme.verdict(
        f"Repair <b>{n_sel} engines</b> over the next {WEEKS} weeks. "
        f"Expected saving: <b>{money(saving)}</b> compared with waiting for failures."
    )
    st.write(
        f"Check against what really happened: **{bt['caught']} of the {bt['truly_failing']}** engines that "
        f"truly would have failed in this period are on the schedule. {bt['unneeded']} scheduled "
        f"engines did not strictly need it."
    )

    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.markdown("### The schedule")
        sch = p[p["selected"]].sort_values(["window", "pred_rul"])
        if sch.empty:
            st.write("With these costs, no engine is worth repairing yet.")
        else:
            out = sch[["window", "engine", "pred_rul", "p_fail"]].copy()
            out["p_fail"] = out["p_fail"] * 100
            st.dataframe(
                out.rename(columns={"window": "Week", "engine": "Engine", "pred_rul": "Flights left",
                                    "p_fail": "Chance of failing"}),
                hide_index=True, use_container_width=True, height=360,
                column_config={
                    "Flights left": st.column_config.NumberColumn(format="%.0f"),
                    "Chance of failing": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
                },
            )
            st.download_button("Download schedule", sch.to_csv(index=False).encode(),
                               "repair_schedule.csv", "text/csv")
    with right:
        st.markdown("### What each approach costs")
        names = ["Wait for failures", "Repair anything under 50 flights", "This plan"]
        vals = pol["Expected cost (k$)"].tolist()
        fig = go.Figure(
            go.Bar(x=vals, y=names, orientation="h",
                   marker_color=[theme.FAINT, theme.FAINT, theme.ACCENT],
                   text=[money(v) for v in vals], textposition="outside")
        )
        fig.update_xaxes(visible=False, range=[0, max(vals) * 1.7])
        fig.update_yaxes(autorange="reversed")
        theme.style(fig, height=260, margin=dict(l=200, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
        st.markdown(
            '<span class="hint">Cost = repairs plus the expected cost of the failures that still happen. '
            "An engine is repaired when the failure risk it removes is worth more than the repair.</span>",
            unsafe_allow_html=True,
        )
