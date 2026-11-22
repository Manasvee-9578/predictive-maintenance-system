"""Replay: watch one engine wear out, and see when the system notices."""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import core, theme

ALERT = 0.60


@st.cache_data(show_spinner=False)
def _trace(engine: int, fast_wear: bool):
    """Score every flight of one run-to-failure engine (optionally ageing it faster)."""
    b = core.load_bundle()
    raw = b.train_raw[b.train_raw["engine"] == engine].reset_index(drop=True).copy()
    if fast_wear:
        healthy, eol = core.anchors()
        ramp = np.clip((raw["cycle"] - 50) / 40.0, 0, 1)
        for c in core.S_COLS:
            raw[c] = raw[c] + 0.8 * (eol[c] - healthy[c]) * ramp
    sm = core._smooth(raw)
    pred, spread = core.predict_rul(b, sm)
    anom = core.anomaly_score(b, sm)
    z = (sm[core.S_COLS] - b.base_mean) / b.base_std
    out = pd.DataFrame({"cycle": raw["cycle"], "pred": pred, "spread": spread, "anom": anom})
    out.loc[out["cycle"] <= 5, "anom"] = out["anom"].iloc[5]  # warm-up of the rolling window
    out["true"] = (raw["cycle"].max() - raw["cycle"]).clip(upper=core.MAX_RUL)
    return out, z


def _events(tr, z):
    ev, alerted, tier = [], False, "Good"
    for i, r in tr.iterrows():
        top = z.iloc[i].abs().sort_values(ascending=False).index[0]
        if not alerted and r["anom"] > ALERT:
            alerted = True
            ev.append((int(r["cycle"]), f"Warning raised. Biggest change: {core.sname(top).lower()}.", theme.ACCENT))
        t = core.tier_of(r["pred"])
        if core.TIER_ORDER.index(t) < core.TIER_ORDER.index(tier):
            tier = t
            ev.append((int(r["cycle"]), f"Status changed to {t.lower()} (about {r['pred']:.0f} flights left).", core.TIER_COLOR[t]))
    return ev


def _draw(tr, ev, upto, ph, tick):
    cur = tr.iloc[upto]
    d = tr.iloc[: upto + 1]
    tier = core.tier_of(cur["pred"])
    warned = (d["anom"] > ALERT).any()
    with ph["head"].container():
        theme.verdict(
            f"Flight <b>{int(cur['cycle'])}</b>. The system thinks about <b>{cur['pred']:.0f}</b> flight{'' if round(cur['pred']) == 1 else 's'} remain. "
            f"{theme.pill(tier, core.TIER_COLOR[tier])}"
            + (f' &nbsp;{theme.pill("Warning raised", theme.ACCENT)}' if warned else "")
        )

    x_max = int(tr["cycle"].max())
    f1 = go.Figure()
    f1.add_scatter(x=d["cycle"], y=d["pred"] + d["spread"], line=dict(width=0), hoverinfo="skip", showlegend=False)
    f1.add_scatter(x=d["cycle"], y=(d["pred"] - d["spread"]).clip(lower=0), line=dict(width=0),
                   fill="tonexty", fillcolor="rgba(238,108,43,0.15)", hoverinfo="skip", showlegend=False)
    f1.add_scatter(x=d["cycle"], y=d["true"], name="Real life left", line=dict(color=theme.NAVY, width=2, dash="dot"))
    f1.add_scatter(x=d["cycle"], y=d["pred"], name="System's estimate", line=dict(color=theme.ACCENT, width=3))
    f1.update_xaxes(range=[0, x_max], title="flights flown")
    f1.update_yaxes(range=[0, 140], title="flights left")
    theme.style(f1, height=330)
    ph["rul"].plotly_chart(f1, use_container_width=True, theme=None, key=f"rul{tick}", config={"displayModeBar": False})

    f2 = go.Figure()
    f2.add_scatter(x=d["cycle"], y=d["anom"].clip(upper=1.1), line=dict(color=theme.NAVY, width=2.5),
                   fill="tozeroy", fillcolor="rgba(31,58,95,0.10)", name="Warning level")
    f2.add_hline(y=ALERT, line_color=theme.BAD, line_dash="dash", line_width=1.2,
                 annotation_text="warning line", annotation_font_color=theme.BAD)
    f2.update_xaxes(range=[0, x_max], title="flights flown")
    f2.update_yaxes(range=[0, 1.1], title="how unusual it looks", showticklabels=False)
    theme.style(f2, height=330)
    ph["anom"].plotly_chart(f2, use_container_width=True, theme=None, key=f"an{tick}", config={"displayModeBar": False})

    rows = [e for e in ev if e[0] <= cur["cycle"]][::-1]
    html = "".join(
        f'<div><b style="color:{col}">Flight {c}</b> &nbsp;{m}</div>' for c, m, col in rows
    ) or '<div class="hint">Nothing to report. The engine looks healthy.</div>'
    ph["log"].markdown(f'<div class="log">{html}</div>', unsafe_allow_html=True)


def render():
    core.load_bundle()
    theme.header(
        "Watch an engine wear out",
        "Press play to fly one engine from its first flight to failure. The system only sees "
        "the sensor data up to the current flight, just like in real life.",
    )

    c1, c2, c3 = st.columns([1, 1.4, 1])
    engine = c1.selectbox("Engine", list(range(1, 101)), index=0)
    fast = c2.toggle("Make it wear out faster", value=False,
                     help="Adds extra wear to every sensor from flight 50, so you can see the warning fire early.")
    speed = c3.select_slider("Speed", ["Slow", "Normal", "Fast"], value="Normal")

    tr, z = _trace(engine, fast)
    ev = _events(tr, z)
    n = len(tr)

    play = st.button("Play", type="primary")
    ph = {"head": st.empty()}
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("### How much life is left?")
        ph["rul"] = st.empty()
    with b:
        st.markdown("### Does it look unusual?")
        ph["anom"] = st.empty()
    st.markdown("### What happened")
    ph["log"] = st.empty()

    if play:
        step = {"Slow": 2, "Normal": 5, "Fast": 10}[speed]
        for t, upto in enumerate(list(range(2, n, step)) + [n - 1]):
            _draw(tr, ev, upto, ph, t)
            time.sleep(0.02)
    else:
        jump = st.slider("Or scrub to any flight", 1, n, n)
        _draw(tr, ev, jump - 1, ph, "static")
