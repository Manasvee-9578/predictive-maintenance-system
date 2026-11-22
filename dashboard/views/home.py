"""Home: the whole fleet at a glance."""

import plotly.graph_objects as go
import streamlit as st

from dashboard import core, theme
from dashboard.views.planner import money


def render():
    b = core.load_bundle()
    f = b.fleet
    n = {t: int((f["tier"] == t).sum()) for t in core.TIER_ORDER}
    need = n["Ground now"] + n["Service soon"]
    alerts = core.build_alerts(f)
    acked = st.session_state.get("acked", set())
    open_alerts = int((~alerts["engine"].isin(acked)).sum()) if len(alerts) else 0

    theme.header(
        "Know which engine fails next, before it does.",
        "Sentinel reads engine sensor data, estimates how many flights each engine has left, "
        "warns you when readings look unusual, and plans the repairs that save the most money.",
        tag="Predictive maintenance platform", big=True,
        chips=[f"{len(f)} engines monitored", f"Average error {b.metrics.iloc[0]['MAE']:.0f} flights",
               "Trained on NASA C-MAPSS data"],
    )

    theme.kpi_cards(
        [
            ("Engines monitored", f"{len(f)}", "in the test fleet", theme.NAVY),
            ("Need attention", f"{need}", f"{n['Ground now']} ground now, {n['Service soon']} service soon", theme.BAD),
            ("Typical life left", f"{f['pred_rul'].median():.0f}", "flights (fleet median)", theme.GOOD),
            ("Open alerts", f"{open_alerts}", "waiting to be acknowledged", theme.ACCENT),
        ]
    )

    c1, c2 = st.columns([1, 1.5], gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("### Fleet status")
            fig = go.Figure(
                go.Pie(
                    labels=core.TIER_ORDER, values=[n[t] for t in core.TIER_ORDER], hole=0.64, sort=False,
                    marker=dict(colors=[core.TIER_COLOR[t] for t in core.TIER_ORDER], line=dict(color="#fff", width=3)),
                    textinfo="value", textfont=dict(color="#fff", size=14),
                    hovertemplate="%{label}: %{value} engines<extra></extra>",
                )
            )
            fig.add_annotation(text=f"<b>{len(f)}</b><br>engines", showarrow=False, font=dict(size=18, color=theme.INK))
            fig.update_layout(legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"))
            theme.style(fig, height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    with c2:
        with st.container(border=True):
            st.markdown("### How much life is left across the fleet")
            fig2 = go.Figure()
            for t in reversed(core.TIER_ORDER):
                d = f[f["tier"] == t]
                fig2.add_histogram(x=d["pred_rul"], name=t, marker_color=core.TIER_COLOR[t],
                                   xbins=dict(start=0, end=130, size=10),
                                   hovertemplate="%{x} flights: %{y} engines<extra>" + t + "</extra>")
            fig2.update_layout(barmode="stack", bargap=0.08)
            fig2.update_xaxes(title="flights left")
            fig2.update_yaxes(title="engines")
            theme.style(fig2, height=320)
            st.plotly_chart(fig2, use_container_width=True, theme=None, config={"displayModeBar": False})

    c3, c4 = st.columns([1.3, 1], gap="medium")
    with c3:
        with st.container(border=True):
            st.markdown("### Most urgent engines")
            top = f.sort_values("pred_rul").head(6)
            rows = "".join(
                f'<div class="item"><span><b>Engine {r.engine}</b> &nbsp;{theme.pill(r.tier, core.TIER_COLOR[r.tier])}</span>'
                f"<span>{r.pred_rul:.0f} flights left</span></div>"
                for r in top.itertuples()
            )
            st.markdown(rows, unsafe_allow_html=True)
            st.markdown('<span class="hint">Open Alerts to acknowledge them, or Engine report for the full picture.</span>',
                        unsafe_allow_html=True)
    with c4:
        with st.container(border=True):
            st.markdown("### Money at stake")
            cm = core.CostModel()
            p = core.plan(f, cm, b.sigma)
            pol = core.policy_costs(p, cm)
            save = pol.iloc[0, 1] - pol.iloc[2, 1]
            st.markdown(
                f'<div class="kpi" style="--c:{theme.ACCENT};box-shadow:none"><div class="l">Estimated saving</div>'
                f'<div class="v">{money(save)}</div>'
                f'<div class="n">by repairing {int(p["selected"].sum())} engines in 4 weeks instead of waiting for failures</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<span class="hint">Default cost assumptions. Change them on Plan repairs.</span>', unsafe_allow_html=True)
