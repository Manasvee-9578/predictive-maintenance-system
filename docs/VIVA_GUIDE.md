# Demo script and viva guide

## 5-minute demo flow

1. **Home, then Alerts centre (60 s).** "100 engines in service, each scored on its latest flight." Show the status donut and savings figure, then open *Alerts centre*, tick one alert and assign it to a crew.
   Open *Engine report* for engine 81 and download its report.
2. **Watch it wear (90 s).** Engine 1, no fault, press *Play*. Show the amber RUL tracking
   the dotted true RUL with its uncertainty band, and the anomaly score crossing the alert line
   late in life. Then switch on *Make it wear out faster* and play again: the alert fires
   roughly 70 flights earlier (re-check this on screen). The event log names the sensors responsible.
3. **What-if simulator (30 s).** Drag the wear slider from 0 to 100%. Flights left falls steadily and the sensor list changes.
4. **Plan repairs (90 s).** Show the schedule and the three-strategy cost chart. Lower
   *engines the workshop can take* from 32 to 12 and watch engines drop off the schedule and the cost rise.
   Quote the backtest: "25 of 28 engines that really fail inside the horizon were scheduled."
5. **Use your data (30 s).** Switch on *Just use the sample* to show predictions for an uploaded file, or upload your own CSV.
6. **Accuracy (30 s).** Open *Compare the different models*. Mention the LSTM result and why the tree models are used.

## Likely questions

**Why Random Forest and not the LSTM?** The LSTM trained in the offline pipeline scored
RMSE 40.7 and R² 0.04 on the test fleet, far worse than the forest (20.5, 0.74). With
only 100 training engines and noisy sensors, the tree models on smoothed sensors
generalise better. The LSTM result is reported, not hidden.

**Why cap RUL at 125?** Engines show no measurable degradation early in life, so asking a
model to tell RUL 300 from RUL 200 is meaningless. A piecewise-linear cap is the standard
treatment for C-MAPSS.

**How is the anomaly detector trained without labels?** It is unsupervised: an Isolation
Forest fitted on the first 40 cycles of each engine (healthy behaviour). Cycles that look
unlike that baseline score higher. The score is rescaled using healthy and end-of-life medians.

**Where does the uncertainty come from?** The standard deviation of the individual trees'
predictions. The planner never trusts it blindly: it uses the larger of that spread and the
validation error (about 20 cycles).

**How is failure probability computed?** P(fail within H cycles) = Φ((H − RUL) / σ), a normal
approximation around the predicted RUL.

**Is the cost data real?** No. They are illustrative defaults, editable on the planner page.
The method is what matters: the same pipeline works with real costs.

**What is the NASA score?** An asymmetric error metric from the C-MAPSS challenge that
punishes late predictions (predicting more life than remains) harder than early ones.

**What are the limitations?** FD001 is one operating condition and one fault mode (HPC
degradation). Results are not claimed to transfer to other engines without retraining.
Injected faults in the replay are synthetic.

**Why is there no single-sensor fault option?** Early testing showed the models learn from the joint pattern of
14 sensors, so one drifting sensor barely registers. Real degradation moves all sensors together,
so the faster-wear toggle does the same.
