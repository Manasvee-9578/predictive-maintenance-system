# Sentinel: Predictive Maintenance for Turbofan Engines

An end-to-end predictive maintenance system built on the NASA C-MAPSS FD001 turbofan
dataset. It does not stop at predicting Remaining Useful Life (RUL): it turns the
forecast into a **costed, capacity-limited shop schedule** and checks that schedule
against the held-back ground truth.

## What makes it different

- **Uncertainty-aware RUL.** The Random Forest's tree-to-tree spread gives every
  prediction a confidence band.
- **Explainable anomalies.** An Isolation Forest fitted only on healthy early-life data
  scores each cycle from 0 (healthy) to 1 (end of life), and sensor z-scores name the drivers.
- **Decision layer.** RUL and uncertainty become a failure probability inside a planning
  horizon, then an expected cost, then a schedule under bay, window and budget limits.
- **Honest evaluation.** A backtest shows how many real failures the schedule catches,
  and the model lab reports the weaker LSTM result instead of hiding it.

## Run the dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Models are trained from `data/nasa/` on first launch (about 15 seconds, then cached).

| Page | What it does |
|---|---|
| Home | Fleet status donut, life-left distribution, most urgent engines, estimated savings |
| Fleet explorer | Search, filter and sort all engines; map and table views; CSV export |
| Engine report | Per-engine history, sensor comparison, recommendation, downloadable report |
| Alerts centre | Severity-ranked alert queue; tick off alerts and assign crews |
| Watch it wear | Plays an engine from first flight to failure; extra-wear toggle shows the warning firing early |
| What-if simulator | One wear slider on a virtual engine |
| Plan repairs | Workshop capacity and costs in, repair schedule, cost comparison and backtest out |
| Use your data | Upload a sensor CSV and get life-left estimates and warnings for every flight |
| Accuracy | Predicted vs real life left on unseen engines, with a model comparison |
| How it works | Four-step explanation and limitations |

## Results (100 unseen test engines, last observed cycle)

| Model | MAE (cycles) | RMSE | R² |
|---|---|---|---|
| Random Forest | 14.3 | 20.5 | 0.74 |
| Gradient Boosting | 14.6 | 21.1 | 0.72 |
| Ridge | 17.2 | 22.0 | 0.70 |
| LSTM (offline pipeline) | 29.3 | 40.7 | 0.04 |

## Project layout

```
dashboard/        Streamlit app: app.py, core.py (data, models, cost model), theme.py, views/
src/              Offline pipeline: preprocessing, EDA, anomaly detection, RUL models (incl. LSTM),
                  maintenance rules, streaming simulator
configs/          Settings
data/nasa/        Raw C-MAPSS FD001 files
models/           Models saved by the offline pipeline
outputs/          Pipeline outputs (predictions, plots)
docs/             Demo script and viva guide
```

The dashboard is self-contained (`dashboard/core.py` trains its own models). The `src/`
pipeline is the original research code, including the TensorFlow LSTM, which is not
required to run the dashboard.

## Limitations

FD001 has a single operating condition and a single fault mode. Cost figures are
illustrative defaults. The injected faults in the replay page are synthetic.
