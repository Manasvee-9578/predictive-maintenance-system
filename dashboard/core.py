"""
Sentinel core: data loading, models, scoring and the maintenance cost model.

Everything here is trained / derived from the raw NASA C-MAPSS FD001 files at
start-up (cached by Streamlit), so the dashboard never depends on pickled
models that may have been saved with a different scikit-learn version.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "nasa"

MAX_RUL = 125  # piece-wise linear RUL cap (standard for C-MAPSS)
SMOOTH = 10  # rolling window (cycles) applied before modelling

# the 14 informative sensors (the other 7 are constant in FD001)
SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
S_COLS = [f"s{i}" for i in SENSORS]

# plain-English names so the demo needs no engineering vocabulary
SENSOR_INFO = {
    "s2": "Compressor inlet temperature",
    "s3": "Compressor outlet temperature",
    "s4": "Turbine exit temperature",
    "s7": "Compressor pressure",
    "s8": "Fan speed",
    "s9": "Core speed",
    "s11": "Compressor static pressure",
    "s12": "Fuel flow ratio",
    "s13": "Corrected fan speed",
    "s14": "Corrected core speed",
    "s15": "Bypass ratio",
    "s17": "Bleed air energy",
    "s20": "Turbine cooling air (hot side)",
    "s21": "Turbine cooling air (cool side)",
}


def sname(col: str) -> str:
    return SENSOR_INFO[col]


def sdesc(col: str) -> str:
    return SENSOR_INFO[col]


def deviation_words(z: float) -> str:
    a = abs(z)
    if a < 1:
        return "normal"
    size = "slightly" if a < 2 else ("clearly" if a < 4 else "far")
    return f"{size} {'higher' if z > 0 else 'lower'} than normal"


# ── Raw data ──────────────────────────────────────────────────
def _read(name: str) -> pd.DataFrame:
    cols = ["engine", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    return pd.read_csv(RAW / name, sep=r"\s+", header=None, names=cols)


def _smooth(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[S_COLS] = (
        df.groupby("engine")[S_COLS]
        .rolling(SMOOTH, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return out


@dataclass
class Bundle:
    train: pd.DataFrame  # smoothed, with rul
    train_raw: pd.DataFrame
    test: pd.DataFrame  # smoothed
    test_raw: pd.DataFrame
    true_rul: pd.Series  # indexed by engine, ground truth for test fleet
    models: dict
    rf: RandomForestRegressor
    iforest: IsolationForest
    scaler: StandardScaler
    base_mean: pd.Series
    base_std: pd.Series
    a_lo: float
    a_hi: float
    metrics: pd.DataFrame
    sigma: float  # residual std (cycles) used for failure probability
    fleet: pd.DataFrame  # one row per test engine at its last cycle


def nasa_score(err: np.ndarray) -> float:
    """Asymmetric C-MAPSS score: late predictions are punished harder."""
    err = np.asarray(err, dtype=float)
    return float(np.sum(np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1)))


@st.cache_resource(show_spinner="Training models on NASA C-MAPSS FD001 ...")
def load_bundle() -> Bundle:
    tr_raw, te_raw = _read("train_FD001.txt"), _read("test_FD001.txt")
    true_rul = pd.read_csv(RAW / "RUL_FD001.txt", header=None)[0]
    true_rul.index = np.arange(1, len(true_rul) + 1)

    tr_raw["rul"] = (
        tr_raw.groupby("engine")["cycle"].transform("max") - tr_raw["cycle"]
    ).clip(upper=MAX_RUL)
    tr, te = _smooth(tr_raw), _smooth(te_raw)

    X, y = tr[S_COLS].values, tr["rul"].values
    # hold out 20 engines to measure honest generalisation + residual spread
    val_ids = np.arange(81, 101)
    m_tr, m_va = ~tr["engine"].isin(val_ids), tr["engine"].isin(val_ids)

    def mk():
        return {
            "Random Forest": RandomForestRegressor(
                n_estimators=160, max_depth=14, min_samples_leaf=3,
                n_jobs=-1, random_state=7,
            ),
            "Gradient Boosting": HistGradientBoostingRegressor(
                max_iter=250, learning_rate=0.06, max_depth=6, random_state=7
            ),
            "Ridge (linear)": Ridge(alpha=1.0),
        }

    # honest check on the real test fleet (last observed cycle vs ground truth)
    last = te.sort_values("cycle").groupby("engine").tail(1).set_index("engine")
    models, rows = {}, []
    for name, mdl in mk().items():
        mdl.fit(X, y)
        models[name] = mdl
        pred = np.clip(mdl.predict(last[S_COLS].values), 0, MAX_RUL)
        t = np.minimum(true_rul.loc[last.index].values, MAX_RUL)
        err = pred - t
        rows.append(
            {
                "Model": name,
                "MAE": np.abs(err).mean(),
                "RMSE": np.sqrt((err**2).mean()),
                "R2": 1 - (err**2).sum() / ((t - t.mean()) ** 2).sum(),
                "NASA score": nasa_score(err),
            }
        )
    metrics = pd.DataFrame(rows)

    # residual spread from the held-out validation engines (fit a copy without them)
    chk = mk()["Random Forest"].fit(X[m_tr], y[m_tr])
    res = chk.predict(X[m_va]) - y[m_va]
    sigma = float(np.std(res))

    # anomaly detector: Isolation Forest fitted on the healthy early life only
    scaler = StandardScaler().fit(tr.loc[tr["cycle"] <= 40, S_COLS])
    healthy = scaler.transform(tr.loc[tr["cycle"] <= 40, S_COLS])
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=7).fit(healthy)
    raw_all = -iso.score_samples(scaler.transform(tr[S_COLS]))
    a_lo = float(np.median(-iso.score_samples(healthy)))
    a_hi = float(np.median(raw_all[tr["rul"].values < 15]))

    base = tr.loc[tr["cycle"] <= 40, S_COLS]
    bundle = Bundle(
        train=tr, train_raw=tr_raw, test=te, test_raw=te_raw, true_rul=true_rul,
        models=models, rf=models["Random Forest"], iforest=iso, scaler=scaler,
        base_mean=base.mean(), base_std=base.std().replace(0, 1e-9),
        a_lo=a_lo, a_hi=a_hi, metrics=metrics, sigma=max(sigma, 8.0),
        fleet=pd.DataFrame(),
    )
    bundle.fleet = _build_fleet(bundle, last)
    return bundle


# ── Scoring helpers ───────────────────────────────────────────
def anomaly_score(b: Bundle, frame: pd.DataFrame) -> np.ndarray:
    """0 = looks like early-life healthy behaviour, 1 = looks like end of life."""
    raw = -b.iforest.score_samples(b.scaler.transform(frame[S_COLS]))
    return np.clip((raw - b.a_lo) / (b.a_hi - b.a_lo + 1e-9), 0, 1.2)


def predict_rul(b: Bundle, frame: pd.DataFrame, model: str = "Random Forest"):
    """Return (rul, spread) — spread is the std-dev across the forest's trees."""
    X = frame[S_COLS].values
    mdl = b.models[model]
    pred = np.clip(mdl.predict(X), 0, MAX_RUL)
    if model == "Random Forest":
        per_tree = np.stack([t.predict(X) for t in mdl.estimators_])
        spread = per_tree.std(axis=0)
    else:
        spread = np.full(len(pred), b.sigma)
    return pred, spread


@st.cache_data(show_spinner=False)
def anchors():
    """Typical healthy (early-life) and end-of-life sensor values, from the training fleet."""
    b = load_bundle()
    healthy = b.train.loc[b.train["cycle"] <= 40, S_COLS].median()
    eol = b.train.loc[b.train["rul"] < 8, S_COLS].median()
    return healthy, eol


def sensor_deviation(b: Bundle, row: pd.Series) -> pd.Series:
    """z-score of every sensor against this fleet's healthy baseline."""
    return ((row[S_COLS] - b.base_mean) / b.base_std).astype(float)


def phi(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def fail_prob(rul: float, horizon: float, sigma: float) -> float:
    """P(engine fails within `horizon` cycles) assuming N(rul, sigma) uncertainty."""
    return phi((horizon - rul) / max(sigma, 1e-6))


TIERS = [  # (name, upper RUL bound, colour)
    ("Ground now", 20, "#d64545"),
    ("Service soon", 45, "#e8742b"),
    ("Watch", 80, "#e5a82b"),
    ("Good", 10_000, "#2f9e66"),
]
TIER_COLOR = {n: c for n, _, c in TIERS}
TIER_ORDER = [n for n, _, _ in TIERS]


def tier_of(rul: float) -> str:
    for name, hi, _ in TIERS:
        if rul < hi:
            return name
    return "Good"


def _build_fleet(b: Bundle, last: pd.DataFrame) -> pd.DataFrame:
    pred, spread = predict_rul(b, last)
    f = pd.DataFrame(
        {
            "engine": last.index,
            "cycle": last["cycle"].values,
            "pred_rul": pred,
            "spread": spread,
            "anomaly": np.clip(anomaly_score(b, last), 0, 1),
            "true_rul": np.minimum(b.true_rul.loc[last.index].values, MAX_RUL),
        }
    )
    f["tier"] = f["pred_rul"].map(tier_of)
    f["health"] = (f["pred_rul"] / MAX_RUL * 100).round(1)
    return f.reset_index(drop=True)


# ── Maintenance planner ──────────────────────────────────────
@dataclass
class CostModel:
    planned: float = 45.0  # k$ per planned shop visit
    failure: float = 480.0  # k$ per in-service failure (AOG, tow, damage)
    waste_per_cycle: float = 0.9  # k$ of life thrown away per unused cycle
    horizon: int = 40  # cycles we look ahead
    bays: int = 8  # engines the shop can take per window
    windows: int = 4  # number of shop windows in the horizon
    budget: float = 1500.0  # k$ available


def plan(fleet: pd.DataFrame, cm: CostModel, sigma: float) -> pd.DataFrame:
    f = fleet.copy()
    sig = np.maximum(f["spread"].values, sigma)  # never be more certain than validation
    f["p_fail"] = [fail_prob(r, cm.horizon, s) for r, s in zip(f["pred_rul"], sig)]
    f["exp_loss"] = f["p_fail"] * cm.failure
    f["benefit"] = f["exp_loss"] - cm.planned - cm.waste_per_cycle * f["pred_rul"]
    f = f.sort_values("benefit", ascending=False).reset_index(drop=True)

    capacity = cm.bays * cm.windows
    chosen, spent = [], 0.0
    for i, r in f.iterrows():
        if r["benefit"] <= 0 or len(chosen) >= capacity or spent + cm.planned > cm.budget:
            continue
        chosen.append(i)
        spent += cm.planned
    f["selected"] = f.index.isin(chosen)

    # earliest-to-fail engines get the earliest shop windows
    sel = f[f["selected"]].sort_values("pred_rul")
    f["window"] = pd.NA
    for k, idx in enumerate(sel.index):
        f.loc[idx, "window"] = k // cm.bays + 1
    return f


def policy_costs(f: pd.DataFrame, cm: CostModel) -> pd.DataFrame:
    """Expected cost of three strategies, plus a backtest against ground truth."""
    n = len(f)
    rtf = f["exp_loss"].sum()  # run-to-failure: wait and pay for every failure
    thr = f["pred_rul"] < 50  # static rule used by many fleets
    static = thr.sum() * cm.planned + f.loc[~thr, "exp_loss"].sum()
    sel = f["selected"]
    opt = sel.sum() * cm.planned + f.loc[~sel, "exp_loss"].sum()
    return pd.DataFrame(
        {
            "Strategy": ["Run to failure", "Fixed rule: RUL < 50", "Sentinel planner"],
            "Expected cost (k$)": [rtf, static, opt],
            "Engines serviced": [0, int(thr.sum()), int(sel.sum())],
        }
    )


def backtest(f: pd.DataFrame, cm: CostModel) -> dict:
    """Use the held-back ground truth: who *really* fails inside the horizon?"""
    will_fail = f["true_rul"] <= cm.horizon
    sel = f["selected"]
    return {
        "truly_failing": int(will_fail.sum()),
        "caught": int((will_fail & sel).sum()),
        "missed": int((will_fail & ~sel).sum()),
        "unneeded": int((~will_fail & sel).sum()),
    }


# ── Extra helpers for the product pages ──────────────────────
def engine_history(b: Bundle, eid: int):
    """Score every flight of one in-service (test) engine."""
    sm = b.test[b.test["engine"] == eid].sort_values("cycle").reset_index(drop=True)
    pred, spread = predict_rul(b, sm)
    hist = pd.DataFrame(
        {"cycle": sm["cycle"], "pred": pred, "spread": spread,
         "anom": np.clip(anomaly_score(b, sm), 0, 1.1)}
    )
    return hist, sm


SEVERITY = {"Ground now": "Critical", "Service soon": "High", "Watch": "Medium"}
SEV_ORDER = ["Critical", "High", "Medium"]
SEV_COLOR = {"Critical": "#d63b3b", "High": "#ee7623", "Medium": "#e8a317"}


def build_alerts(fleet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in fleet.itertuples():
        if r.tier in SEVERITY:
            sev = SEVERITY[r.tier]
            if r.tier == "Ground now":
                msg = f"Expected to fail within about {r.pred_rul:.0f} flights. Remove from service."
            elif r.tier == "Service soon":
                msg = f"About {r.pred_rul:.0f} flights left. Book into the workshop."
            else:
                msg = f"About {r.pred_rul:.0f} flights left. Keep monitoring."
        elif r.anomaly > 0.6:
            sev, msg = "Medium", "Sensor readings look unusual even though life left is high."
        else:
            continue
        rows.append({"engine": r.engine, "severity": sev, "message": msg,
                     "flights_left": round(r.pred_rul), "anomaly": round(float(r.anomaly), 2)})
    a = pd.DataFrame(rows)
    if a.empty:
        return a
    a["_o"] = a["severity"].map({s: i for i, s in enumerate(SEV_ORDER)})
    return a.sort_values(["_o", "flights_left"]).drop(columns="_o").reset_index(drop=True)


def normalise_upload(df: pd.DataFrame) -> pd.DataFrame:
    """Accept 'sensor_11', 'Sensor 11', 's11' ... and return columns named s11."""
    import re

    ren = {}
    for c in df.columns:
        m = re.fullmatch(r"\s*(?:sensor[_ ]?|s)(\d+)\s*", str(c), flags=re.I)
        if m:
            ren[c] = f"s{int(m.group(1))}"
        elif str(c).strip().lower() in ("engine", "engine_id", "unit", "unit_id", "id"):
            ren[c] = "engine"
        elif str(c).strip().lower() in ("cycle", "flight", "time", "time_in_cycles"):
            ren[c] = "cycle"
    return df.rename(columns=ren)


def predict_upload(b: Bundle, df: pd.DataFrame):
    """Return (per-row results, error message or None)."""
    df = normalise_upload(df)
    missing = [c for c in S_COLS if c not in df.columns]
    if missing:
        return None, "Missing columns: " + ", ".join(missing)
    df = df.copy()
    if "engine" not in df.columns:
        df["engine"] = 1
    if "cycle" not in df.columns:
        df["cycle"] = df.groupby("engine").cumcount() + 1
    df = df.sort_values(["engine", "cycle"]).reset_index(drop=True)
    df[S_COLS] = df[S_COLS].apply(pd.to_numeric, errors="coerce")
    if df[S_COLS].isna().any().any():
        return None, "Some sensor values are empty or not numbers."
    sm = _smooth(df)
    pred, spread = predict_rul(b, sm)
    out = df[["engine", "cycle"]].copy()
    out["pred_rul"], out["spread"] = pred, spread
    out["anomaly"] = np.clip(anomaly_score(b, sm), 0, 1.1)
    out["status"] = out["pred_rul"].map(tier_of)
    return out, None
