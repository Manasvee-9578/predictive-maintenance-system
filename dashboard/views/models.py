"""How accurate is it? Tested on engines the models never saw."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import core, theme


def _lstm_row():
    p = core.ROOT / "outputs" / "predictions" / "rul_training_metrics.csv"
    if not p.exists():
        return None
    m = pd.read_csv(p)
    m = m[m["model"].str.contains("Test")]
    if m.empty:
        return None
    r = m.iloc[0]
    return {"Model": "Deep learning (LSTM)", "MAE": r["mae"], "RMSE": r["rmse"], "R2": r["r2"], "NASA score": r["nasa_score"]}


def render():
    b = core.load_bundle()
    mae = b.metrics.iloc[0]["MAE"]
    theme.header(
        "How accurate is it?",
        "We trained on 100 engines and tested on 100 different ones, then compared the "
        "estimate with the real answer that NASA published.",
    )
    theme.verdict(f"On average the estimate is off by <b>{mae:.0f} flights</b>.")

    last = b.test.sort_values("cycle").groupby("engine").tail(1).set_index("engine")
    pred, _ = core.predict_rul(b, last)
    truth = b.fleet.set_index("engine").loc[last.index, "true_rul"].values

    fig = go.Figure()
    fig.add_scatter(x=[0, 125], y=[0, 125], mode="lines", line=dict(color=theme.FAINT, dash="dash"),
                    name="perfect", hoverinfo="skip")
    fig.add_scatter(x=truth, y=pred, mode="markers", name="an engine",
                    marker=dict(size=9, color=theme.ACCENT, opacity=0.85, line=dict(color="#fff", width=1)),
                    text=[f"Engine {e}" for e in last.index],
                    hovertemplate="%{text}<br>real: %{x:.0f} flights<br>estimate: %{y:.0f}<extra></extra>")
    fig.update_xaxes(title="real flights left", range=[0, 130])
    fig.update_yaxes(title="system's estimate", range=[0, 130])
    theme.style(fig, height=430)
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
    st.markdown(
        '<span class="hint">Each dot is one engine. Dots on the dashed line are perfect predictions.</span>',
        unsafe_allow_html=True,
    )

    with st.expander("Compare the different models"):
        m = b.metrics.copy()
        lstm = _lstm_row()
        if lstm:
            m = pd.concat([m, pd.DataFrame([lstm])], ignore_index=True)
        st.dataframe(
            m, hide_index=True, use_container_width=True,
            column_config={
                "MAE": st.column_config.NumberColumn("Average error (flights)", format="%.1f"),
                "RMSE": st.column_config.NumberColumn("Typical big error", format="%.1f"),
                "R2": st.column_config.NumberColumn("Fit (1 is perfect)", format="%.2f"),
                "NASA score": st.column_config.NumberColumn("NASA penalty (lower is better)", format="%.0f"),
            },
        )
        st.write(
            "The first row, Random Forest, is the one this dashboard uses. The deep learning model "
            "from the offline pipeline did worse on this small dataset, so we show it for honesty."
        )
