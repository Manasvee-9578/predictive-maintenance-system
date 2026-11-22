"""Try it yourself: age a virtual engine and see what the system concludes."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import core, theme


def _frame(b, wear):
    healthy, eol = core.anchors()
    row = healthy + wear * (eol - healthy)
    return pd.DataFrame([row[core.S_COLS]])


def render():
    b = core.load_bundle()
    theme.header(
        "Try it yourself",
        "Slide the wear from brand new to worn out. Every sensor moves the way a real "
        "engine's would, and the system tells you what it thinks.",
    )

    wear = st.slider("How worn is the engine?", 0, 100, 30, format="%d%%")
    frame = _frame(b, wear / 100)
    rul, spr = core.predict_rul(b, frame)
    rul, spr = float(rul[0]), float(spr[0])
    tier = core.tier_of(rul)
    warn = float(min(core.anomaly_score(b, frame)[0], 1.0))

    theme.verdict(
        f"About <b>{rul:.0f} flights</b> left, give or take {spr:.0f}. {theme.pill(tier, core.TIER_COLOR[tier])}"
    )
    st.markdown(
        f'<span class="hint">Unusualness: {warn * 100:.0f}% '
        f'({"above" if warn > 0.6 else "below"} the 60% warning line)</span>',
        unsafe_allow_html=True,
    )

    xs = np.linspace(0, 1, 41)
    sw = pd.concat([_frame(b, w) for w in xs], ignore_index=True)
    curve, _ = core.predict_rul(b, sw)
    fig = go.Figure()
    fig.add_scatter(x=xs * 100, y=curve, line=dict(color=theme.ACCENT, width=3), name="Flights left")
    fig.add_scatter(x=[wear], y=[rul], mode="markers", name="This engine",
                    marker=dict(size=14, color=theme.NAVY, line=dict(color="#fff", width=3)))
    fig.update_xaxes(title="wear (%)")
    fig.update_yaxes(title="flights left", range=[0, 135])
    theme.style(fig, height=340)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    st.markdown("### What the sensors show")
    z = core.sensor_deviation(b, frame.iloc[0])
    top = z.reindex(z.abs().sort_values(ascending=False).index).head(4)
    st.markdown(
        "".join(
            f'<div class="item"><span>{core.sdesc(c)}</span><span>{core.deviation_words(v)}</span></div>'
            for c, v in top.items()
        ),
        unsafe_allow_html=True,
    )
