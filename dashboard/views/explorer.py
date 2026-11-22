"""Fleet explorer: search, filter and export every engine."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard import core, theme


def render():
    b = core.load_bundle()
    f = b.fleet
    theme.header("Fleet explorer", "Search and filter all engines. Switch between the visual map and the full table.",
                 tag="Monitor")

    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 2, 1.2])
        q = c1.text_input("Find engine number", placeholder="e.g. 81")
        tiers = c2.multiselect("Status", core.TIER_ORDER, default=core.TIER_ORDER)
        sort = c3.selectbox("Sort by", ["Least life left", "Most life left", "Highest warning level", "Engine number"])

    d = f[f["tier"].isin(tiers)]
    if q.strip().isdigit():
        d = d[d["engine"] == int(q)]
    d = {
        "Least life left": d.sort_values("pred_rul"),
        "Most life left": d.sort_values("pred_rul", ascending=False),
        "Highest warning level": d.sort_values("anomaly", ascending=False),
        "Engine number": d.sort_values("engine"),
    }[sort]
    st.caption(f"Showing {len(d)} of {len(f)} engines")

    tab_map, tab_tbl = st.tabs(["Map", "Table"])
    with tab_map:
        with st.container(border=True):
            order = {t: i for i, t in enumerate(core.TIER_ORDER)}
            full = f.set_index("engine")
            shown = set(d["engine"])
            z = np.array([order[full.loc[e, "tier"]] if e in shown else np.nan for e in range(1, 101)], dtype=float).reshape(10, 10)
            txt = np.array([str(e) for e in range(1, 101)], dtype=object).reshape(10, 10)
            hov = np.array(
                [f"Engine {e}<br>about {full.loc[e, 'pred_rul']:.0f} flights left<br>{full.loc[e, 'tier']}" for e in range(1, 101)]
            ).reshape(10, 10)
            scale = []
            for i, t in enumerate(core.TIER_ORDER):
                scale += [[i / 4, core.TIER_COLOR[t]], [(i + 1) / 4, core.TIER_COLOR[t]]]
            fig = go.Figure(
                go.Heatmap(z=z, text=txt, texttemplate="%{text}", textfont=dict(size=12, color="#fff"),
                           customdata=hov, hovertemplate="%{customdata}<extra></extra>", colorscale=scale,
                           zmin=0, zmax=3.999, showscale=False, xgap=5, ygap=5)
            )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False, autorange="reversed", scaleanchor="x")
            theme.style(fig, height=520, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
            st.markdown('<span class="hint">Faded-out squares are hidden by your filters. Hover for details.</span>',
                        unsafe_allow_html=True)
    with tab_tbl:
        out = d[["engine", "tier", "pred_rul", "spread", "anomaly", "cycle", "health"]].rename(
            columns={"engine": "Engine", "tier": "Status", "pred_rul": "Flights left", "spread": "± flights",
                     "anomaly": "Warning level", "cycle": "Flights flown", "health": "Health %"}
        )
        st.dataframe(
            out, hide_index=True, use_container_width=True, height=520,
            column_config={
                "Flights left": st.column_config.NumberColumn(format="%.0f"),
                "± flights": st.column_config.NumberColumn(format="%.0f"),
                "Warning level": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                "Health %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            },
        )
        st.download_button("Download this list (CSV)", out.to_csv(index=False).encode(), "fleet.csv", "text/csv")
