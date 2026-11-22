"""Use your own data: upload sensor readings and get life-left estimates."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import core, theme


@st.cache_data(show_spinner=False)
def _sample_csv(engine: int = 5) -> bytes:
    b = core.load_bundle()
    d = b.test_raw[b.test_raw["engine"] == engine][["engine", "cycle"] + core.S_COLS]
    return d.to_csv(index=False).encode()


def render():
    b = core.load_bundle()
    theme.header("Use your own data", "Upload a CSV of sensor readings, one row per flight, and get a life-left "
                 "estimate and a warning level for every flight.", tag="Analyse")

    with st.container(border=True):
        c1, c2 = st.columns([2, 1], gap="large")
        with c1:
            up = st.file_uploader("Sensor readings (CSV)", type="csv")
            st.markdown(
                '<span class="hint">Needs these 14 columns: ' + ", ".join(core.S_COLS) +
                ". Optional: <b>engine</b> and <b>cycle</b>. Names like sensor_11 also work.</span>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown("**No data handy?**")
            st.download_button("Download a sample file", _sample_csv(), "sample_engine.csv", "text/csv",
                               use_container_width=True)
            use_sample = st.toggle("Just use the sample", value=False)

    if up is None and not use_sample:
        st.info("Upload a file or switch on the sample to see results.")
        return

    try:
        raw = pd.read_csv(up) if up is not None else pd.read_csv(pd.io.common.BytesIO(_sample_csv()))
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read that file: {e}")
        return
    res, err = core.predict_upload(b, raw)
    if err:
        st.error(err)
        return

    last = res.sort_values("cycle").groupby("engine").tail(1).sort_values("pred_rul")
    theme.kpi_cards(
        [
            ("Rows read", f"{len(res):,}", "flights", theme.NAVY),
            ("Engines", f"{res['engine'].nunique()}", "in your file", theme.NAVY2),
            ("Need action", f"{int(last['status'].isin(['Ground now', 'Service soon']).sum())}",
             "ground now or service soon", theme.BAD),
        ]
    )

    ids = sorted(res["engine"].unique())
    eid = st.selectbox("Engine to inspect", ids, format_func=lambda e: f"Engine {e}")
    d = res[res["engine"] == eid]
    cur = d.iloc[-1]
    theme.verdict(
        f"Engine {eid}: about <b>{cur['pred_rul']:.0f} flights</b> left. "
        f"{theme.pill(cur['status'], core.TIER_COLOR[cur['status']])}"
    )
    with st.container(border=True):
        st.markdown("### Estimated life left by flight")
        fig = go.Figure()
        fig.add_scatter(x=d["cycle"], y=d["pred_rul"] + d["spread"], line=dict(width=0), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=d["cycle"], y=(d["pred_rul"] - d["spread"]).clip(lower=0), line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(242,106,33,0.15)", hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=d["cycle"], y=d["pred_rul"], name="Flights left", line=dict(color=theme.ACCENT, width=3))
        fig.add_scatter(x=d["cycle"], y=d["anomaly"] * core.MAX_RUL, name="Warning level (scaled)",
                        line=dict(color=theme.NAVY, width=1.5, dash="dot"),
                        customdata=d["anomaly"], hovertemplate="%{customdata:.2f}<extra>warning level</extra>")
        fig.update_xaxes(title="flight")
        fig.update_yaxes(title="flights left", range=[0, 140])
        theme.style(fig, height=330)
        st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})

    with st.expander("See and download all results"):
        out = res.rename(columns={"pred_rul": "flights_left", "spread": "plus_minus", "anomaly": "warning_level"})
        st.dataframe(out.round(2), hide_index=True, use_container_width=True, height=300)
        st.download_button("Download results (CSV)", out.round(3).to_csv(index=False).encode(), "predictions.csv", "text/csv")
