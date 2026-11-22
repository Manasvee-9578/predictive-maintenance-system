"""How it works: the pipeline, stage by stage, with a live chart for each stage."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import core, theme

STEPS = [
    ("1", "Read the sensors", "14 measurements per flight"),
    ("2", "Estimate life left", "Random Forest model"),
    ("3", "Spot unusual readings", "Isolation Forest"),
    ("4", "Plan the repairs", "Risk x cost, within capacity"),
]


def _pipeline():
    cells = []
    for i, (n, t, d) in enumerate(STEPS):
        cells.append(
            f'<div style="flex:1;min-width:190px;background:#fff;border:1px solid {theme.LINE};border-radius:14px;'
            f'padding:1rem 1.1rem;box-shadow:0 1px 2px rgba(16,24,40,.05)">'
            f'<div style="width:34px;height:34px;border-radius:50%;background:{theme.ACCENT};color:#fff;font-family:Sora;'
            f'font-weight:700;display:flex;align-items:center;justify-content:center;margin-bottom:.6rem">{n}</div>'
            f'<div style="font-family:Sora;font-weight:600;font-size:1rem">{t}</div>'
            f'<div style="color:{theme.MUTED};font-size:.85rem;margin-top:.15rem">{d}</div></div>'
        )
        if i < len(STEPS) - 1:
            cells.append(f'<div style="align-self:center;color:{theme.ACCENT};font-size:1.6rem;font-weight:700">&rarr;</div>')
    st.markdown(f'<div style="display:flex;gap:.7rem;flex-wrap:wrap;margin-bottom:1.2rem">{"".join(cells)}</div>',
                unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _warning_by_life():
    b = core.load_bundle()
    d = b.train[b.train["engine"] <= 40].copy()
    d["anom"] = np.clip(core.anomaly_score(b, d), 0, 1)
    d["rul_true"] = (d.groupby("engine")["cycle"].transform("max") - d["cycle"])
    d["bucket"] = (d["rul_true"] // 10 * 10).clip(upper=200)
    return d.groupby("bucket")["anom"].mean().reset_index()


def _tab_data(b):
    c1, c2 = st.columns([1, 1.5], gap="large")
    with c1:
        st.markdown("### The data")
        st.write(
            "NASA's public C-MAPSS simulation of turbofan engines. Every engine flew until it failed, "
            "and the sensors were recorded on each flight."
        )
        theme.kpi_cards([
            ("Training engines", "100", "flown until failure", theme.NAVY),
            ("Test engines", "100", "stopped before failure", theme.NAVY2),
            ("Useful sensors", "14", "of 21 (7 never change)", theme.ACCENT),
        ])
    with c2:
        with st.container(border=True):
            st.markdown("### Sensors drift as an engine ages")
            e = b.train[b.train["engine"] == 1]
            fig = go.Figure()
            for s, col in zip(["s4", "s11", "s12"], [theme.ACCENT, theme.NAVY, theme.GOOD]):
                z = (e[s] - b.base_mean[s]) / b.base_std[s]
                fig.add_scatter(x=e["cycle"], y=z, name=core.sdesc(s), line=dict(color=col, width=2.2))
            fig.update_xaxes(title="flights flown (engine 1)")
            fig.update_yaxes(title="difference from healthy")
            theme.style(fig, height=300)
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})


def _tab_life(b):
    c1, c2 = st.columns([1, 1.5], gap="large")
    with c1:
        st.markdown("### Estimating life left")
        st.write(
            "A Random Forest studies engines that ran to failure. For each flight it learns the link between "
            "the sensor readings and the number of flights that were still left. Many decision trees vote, "
            "and how much they disagree becomes the uncertainty band."
        )
        st.write(
            f"Early in life nothing measurable changes, so the target is capped at **{core.MAX_RUL} flights**. "
            "This is the standard approach for this dataset."
        )
        m = b.metrics.iloc[0]
        theme.kpi_cards([
            ("Average error", f"{m['MAE']:.0f}", "flights, on unseen engines", theme.ACCENT),
            ("Fit", f"{m['R2']:.2f}", "1.0 would be perfect", theme.NAVY),
        ])
    with c2:
        with st.container(border=True):
            st.markdown("### Why the target is capped")
            n = 230
            x = np.arange(n)
            fig = go.Figure()
            fig.add_scatter(x=x, y=n - x, name="Actual flights left", line=dict(color=theme.FAINT, width=2, dash="dot"))
            fig.add_scatter(x=x, y=np.minimum(n - x, core.MAX_RUL), name=f"What the model learns (cap {core.MAX_RUL})",
                            line=dict(color=theme.ACCENT, width=3))
            fig.update_xaxes(title="flights flown")
            fig.update_yaxes(title="flights left")
            theme.style(fig, height=300)
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})


def _tab_warn(b):
    c1, c2 = st.columns([1, 1.5], gap="large")
    with c1:
        st.markdown("### Spotting unusual readings")
        st.write(
            "A second model, an Isolation Forest, learns only what a **healthy** engine looks like, using "
            "each engine's first 40 flights. It never sees failure data."
        )
        st.write(
            "When readings drift away from that healthy pattern, the warning level rises from 0 towards 1. "
            "Above **0.60** a warning is raised, and the sensors that changed most are named."
        )
    with c2:
        with st.container(border=True):
            st.markdown("### Warning level rises as failure approaches")
            w = _warning_by_life()
            w = w[w["bucket"] <= 200]
            fig = go.Figure(go.Bar(x=w["bucket"], y=w["anom"], marker_color=[theme.BAD if v > 0.6 else theme.NAVY for v in w["anom"]]))
            fig.add_hline(y=0.6, line_dash="dash", line_color=theme.BAD, annotation_text="warning line",
                          annotation_font_color=theme.BAD)
            fig.update_xaxes(title="real flights left (grouped by 10)", autorange="reversed")
            fig.update_yaxes(title="average warning level", range=[0, 1])
            theme.style(fig, height=300)
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})


def _tab_plan(b):
    c1, c2 = st.columns([1, 1.5], gap="large")
    with c1:
        st.markdown("### Planning the repairs")
        st.write("For every engine the planner works out three things:")
        st.markdown(
            "1. **Chance of failing** within the planning window, from the estimate and its uncertainty.\n"
            "2. **Money at risk** = that chance x the cost of a failure.\n"
            "3. **Net benefit** = money at risk minus the cost of a repair and the life wasted by repairing early."
        )
        st.write(
            "Engines are picked by net benefit until the workshop is full, and the ones closest to failing "
            "go first. The result is checked against NASA's true answers."
        )
    with c2:
        with st.container(border=True):
            st.markdown("### From flights left to chance of failing")
            x = np.arange(0, 121)
            fig = go.Figure()
            for h, col in [(20, theme.GOOD), (40, theme.ACCENT), (70, theme.NAVY)]:
                fig.add_scatter(x=x, y=[core.fail_prob(r, h, b.sigma) * 100 for r in x],
                                name=f"look ahead {h} flights", line=dict(color=col, width=2.6))
            fig.update_xaxes(title="estimated flights left")
            fig.update_yaxes(title="chance of failing in the window (%)", range=[0, 102])
            theme.style(fig, height=300)
            st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})


def render():
    b = core.load_bundle()
    theme.header(
        "How Sentinel works",
        "From raw sensor readings to a costed repair schedule in four steps. Click through the stages "
        "below to see what each one does and why.",
        tag="About",
    )
    _pipeline()

    t1, t2, t3, t4 = st.tabs(["1 · Data", "2 · Life left", "3 · Warnings", "4 · Repair plan"])
    with t1:
        _tab_data(b)
    with t2:
        _tab_life(b)
    with t3:
        _tab_warn(b)
    with t4:
        _tab_plan(b)

    st.markdown("### Built with")
    stack = ["Python", "Streamlit", "scikit-learn", "Pandas", "NumPy", "Plotly", "NASA C-MAPSS FD001"]
    st.markdown(
        "".join(
            f'<span style="display:inline-block;background:#fff;border:1px solid {theme.LINE};border-radius:99px;'
            f'padding:.3rem .9rem;margin:0 .5rem .5rem 0;font-weight:600;font-size:.88rem">{s}</span>'
            for s in stack
        ),
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("### Plain-English glossary")
            for term, d in [
                ("Flights left (RUL)", "How many more flights an engine can make before it fails."),
                ("Warning level", "0 to 1 score of how different the readings are from a healthy engine."),
                ("Uncertainty band", "The range the model gives instead of pretending to be exact."),
                ("Backtest", "Checking the plan against the real outcomes that were held back."),
            ]:
                st.markdown(f'<div class="item"><span><b>{term}</b></span><span style="max-width:62%;text-align:right">{d}</span></div>',
                            unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            st.markdown("### Honest limitations")
            st.markdown(
                f"""
- One engine type and one fault, so results should not be assumed for other engines.
- The estimate is off by about **{b.metrics.iloc[0]['MAE']:.0f} flights** on average. The planner allows for that.
- Costs are example numbers you can change, not real airline data.
- "Make it wear out faster" adds artificial wear, only to show how the warning reacts.
"""
            )
