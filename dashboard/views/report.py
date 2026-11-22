"""Engine report: everything known about one engine, with a downloadable summary."""

from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from dashboard import core, theme

ADVICE = {
    "Ground now": "Remove from service and schedule an immediate shop visit.",
    "Service soon": "Book into the next available workshop window.",
    "Watch": "Keep flying, and re-check after the next few flights.",
    "Good": "No action needed. Continue routine monitoring.",
}


def render():
    b = core.load_bundle()
    f = b.fleet.set_index("engine")
    theme.header("Engine report", "A full health report for one engine: how much life is left, how it got here, "
                 "and what to do about it.", tag="Monitor")

    ids = list(f.index)
    worst = int(f["pred_rul"].idxmin())
    eid = st.selectbox("Choose an engine", ids, index=ids.index(worst), format_func=lambda e: f"Engine {e}")
    r = f.loc[eid]
    hist, sm = core.engine_history(b, eid)
    col = core.TIER_COLOR[r["tier"]]

    theme.verdict(f"Engine {eid} has about <b>{r['pred_rul']:.0f} flights</b> left. {theme.pill(r['tier'], col)}")
    theme.kpi_cards(
        [
            ("Flights left", f"{r['pred_rul']:.0f}", f"give or take {r['spread']:.0f}", col),
            ("Health", f"{r['health']:.0f}%", "100% = like new", theme.NAVY),
            ("Warning level", f"{r['anomaly'] * 100:.0f}%", "above 60% raises a warning", theme.ACCENT if r["anomaly"] > 0.6 else theme.GOOD),
            ("Flights flown", f"{int(r['cycle'])}", "so far", theme.NAVY2),
        ]
    )

    a, c = st.columns(2, gap="medium")
    x_max = int(hist["cycle"].max())
    with a:
        with st.container(border=True):
            st.markdown("### Life left over time")
            fig = go.Figure()
            fig.add_scatter(x=hist["cycle"], y=hist["pred"] + hist["spread"], line=dict(width=0), hoverinfo="skip", showlegend=False)
            fig.add_scatter(x=hist["cycle"], y=(hist["pred"] - hist["spread"]).clip(lower=0), line=dict(width=0),
                            fill="tonexty", fillcolor="rgba(242,106,33,0.15)", hoverinfo="skip", showlegend=False)
            fig.add_scatter(x=hist["cycle"], y=hist["pred"], name="Estimated flights left", line=dict(color=theme.ACCENT, width=3))
            fig.update_xaxes(title="flights flown", range=[0, x_max])
            fig.update_yaxes(title="flights left", range=[0, 140])
            theme.style(fig, height=300)
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    with c:
        with st.container(border=True):
            st.markdown("### Warning level over time")
            fig = go.Figure()
            fig.add_scatter(x=hist["cycle"], y=hist["anom"], line=dict(color=theme.NAVY, width=2.5),
                            fill="tozeroy", fillcolor="rgba(18,41,74,0.10)")
            fig.add_hline(y=0.6, line_color=theme.BAD, line_dash="dash", line_width=1.2,
                          annotation_text="warning line", annotation_font_color=theme.BAD)
            fig.update_xaxes(title="flights flown", range=[0, x_max])
            fig.update_yaxes(range=[0, 1.1], showticklabels=False, title="how unusual it looks")
            theme.style(fig, height=300)
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    dev_now = core.sensor_deviation(b, sm.iloc[-1])
    top = dev_now.reindex(dev_now.abs().sort_values(ascending=False).index)
    with st.container(border=True):
        st.markdown("### Sensors compared with a healthy engine")
        pick = st.multiselect("Sensors to plot", core.S_COLS, default=list(top.index[:3]), format_func=core.sdesc)
        fig = go.Figure()
        palette = [theme.ACCENT, theme.NAVY, "#1f9d62", "#8a5cf6", "#0ea5b7", "#b45309"]
        for i, s in enumerate(pick):
            z = (sm[s] - b.base_mean[s]) / b.base_std[s]
            fig.add_scatter(x=sm["cycle"], y=z, name=core.sdesc(s), line=dict(color=palette[i % len(palette)], width=2))
        fig.add_hline(y=0, line_color=theme.FAINT)
        fig.update_xaxes(title="flights flown")
        fig.update_yaxes(title="difference from healthy (higher = further away)")
        theme.style(fig, height=330)
        st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    l, rr = st.columns([1.2, 1], gap="medium")
    with l:
        with st.container(border=True):
            st.markdown("### Recommendation")
            st.write(ADVICE[r["tier"]])
            st.markdown("**Biggest differences right now**")
            st.markdown(
                "".join(f'<div class="item"><span>{core.sdesc(c)}</span><span>{core.deviation_words(v)}</span></div>'
                        for c, v in top.head(4).items()),
                unsafe_allow_html=True,
            )
    with rr:
        with st.container(border=True):
            st.markdown("### Take it with you")
            lines = [
                "SENTINEL ENGINE HEALTH REPORT",
                f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
                "",
                f"Engine: {eid}",
                f"Status: {r['tier']}",
                f"Estimated flights left: {r['pred_rul']:.0f} (give or take {r['spread']:.0f})",
                f"Health: {r['health']:.0f}%",
                f"Warning level: {r['anomaly'] * 100:.0f}%",
                f"Flights flown: {int(r['cycle'])}",
                "",
                f"Recommendation: {ADVICE[r['tier']]}",
                "",
                "Biggest differences from a healthy engine:",
                *[f"  - {core.sdesc(c)}: {core.deviation_words(v)}" for c, v in top.head(4).items()],
            ]
            st.download_button("Download report (TXT)", "\n".join(lines).encode(), f"engine_{eid}_report.txt",
                               "text/plain", use_container_width=True)
            st.download_button("Download flight history (CSV)", hist.to_csv(index=False).encode(),
                               f"engine_{eid}_history.csv", "text/csv", use_container_width=True)
