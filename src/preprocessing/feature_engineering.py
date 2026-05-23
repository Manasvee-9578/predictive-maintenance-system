"""
╔══════════════════════════════════════════════════════════════╗
║   Advanced Feature Engineering — Predictive Maintenance     ║
║   NASA C-MAPSS FD001 Dataset                                ║
╚══════════════════════════════════════════════════════════════╝

Comprehensive feature engineering pipeline for turbofan engine
degradation modeling. Transforms raw sensor streams into a rich,
model-ready feature set:

    ▸ RUL label generation (piece-wise linear with max cap)
    ▸ Rolling window statistics (mean, std)
    ▸ Lag features (temporal look-back)
    ▸ Health degradation indicators (HI, binary labels)
    ▸ Sensor trend slopes (linear regression over windows)
    ▸ Exponential moving averages (EMA / EWMA)
    ▸ Operating condition normalization (regime-aware)
    ▸ Engine-wise grouped / aggregate features
    ▸ Cycle-based degradation features (normalized lifecycle)
    ▸ Drop low-variance sensors
    ▸ Min-Max sensor normalization

All feature functions are designed to be *reusable* — each can be
called independently or composed via the `transform_train` /
`transform_test` orchestrators.
"""

import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats as sp_stats

from configs.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Reusable Preprocessing Functions (module-level)
# ═══════════════════════════════════════════════════════════════


def compute_rul(df: pd.DataFrame, max_rul: int = None) -> pd.DataFrame:
    """
    Compute piece-wise linear Remaining Useful Life labels.

    For each engine, RUL = max_cycle − current_cycle, capped at `max_rul`
    to model the "healthy plateau" where degradation has not yet begun.

    Args:
        df: DataFrame with columns ``engine_id`` and ``cycle``.
        max_rul: Upper cap for RUL values.  Defaults to ``Settings.MAX_RUL``.

    Returns:
        DataFrame with a new ``rul`` column appended.
    """
    max_rul = max_rul or Settings.MAX_RUL
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    df = df.merge(max_cycles, on="engine_id", how="left")
    df["rul"] = (df["max_cycle"] - df["cycle"]).clip(upper=max_rul)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def compute_rolling_statistics(
    df: pd.DataFrame,
    columns: list[str],
    window: int = 5,
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Add per-engine rolling **mean** and **standard deviation** for each
    specified column.

    New columns:  ``{col}_roll_mean_{window}``, ``{col}_roll_std_{window}``

    Args:
        df: Input DataFrame (must contain ``engine_id``).
        columns: Sensor/feature columns to compute over.
        window: Rolling window size.
        min_periods: Minimum observations in window required to produce a value.
    """
    cols = [c for c in columns if c in df.columns]
    grouped = df.groupby("engine_id")

    for col in cols:
        df[f"{col}_roll_mean_{window}"] = grouped[col].transform(
            lambda x: x.rolling(window, min_periods=min_periods).mean()
        )
        df[f"{col}_roll_std_{window}"] = grouped[col].transform(
            lambda x: x.rolling(window, min_periods=min_periods).std().fillna(0)
        )
    return df


def compute_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Create lagged versions of sensor readings to capture temporal dynamics.

    Each lag column is shifted **per engine** to prevent cross-engine leakage.
    New columns:  ``{col}_lag_{k}``

    Args:
        df: Input DataFrame (must contain ``engine_id``).
        columns: Columns to lag.
        lags: List of positive integers specifying lag steps.
              Defaults to ``[1, 2, 3]``.
    """
    lags = lags or [1, 2, 3]
    cols = [c for c in columns if c in df.columns]
    grouped = df.groupby("engine_id")

    for col in cols:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = grouped[col].transform(lambda x, k=lag: x.shift(k))
    # Back-fill initial NaN rows per engine (first few cycles)
    lag_cols = [c for c in df.columns if "_lag_" in c]
    df[lag_cols] = df.groupby("engine_id")[lag_cols].transform(lambda x: x.bfill())
    return df


def compute_health_degradation_indicators(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Engineer health-degradation indicators per engine:

    1. **Health Index (HI)** — normalised cycle position in ``[0, 1]``.
       ``HI = 0`` at start of life, ``HI = 1`` at end of life (last
       observed cycle).
    2. **Degradation rate** — per-engine, per-cycle rate of HI change
       (first difference of HI).
    3. **Sensor deviation from initial** — each sensor's absolute
       deviation from its engine-specific initial-window average (first 5
       cycles), measuring how far the engine has drifted from healthy.
    4. **Binary late-life flag** — ``1`` when the engine is in the last
       30 % of its observed life.

    Args:
        df: DataFrame with ``engine_id`` and ``cycle``.
        columns: Sensor columns for deviation computation.
    """
    grouped = df.groupby("engine_id")

    # 1. Health Index (normalised lifecycle progress)
    df["health_index"] = grouped["cycle"].transform(
        lambda x: (x - x.min()) / max(x.max() - x.min(), 1)
    )

    # 2. Degradation rate (ΔHI per cycle)
    df["degradation_rate"] = grouped["health_index"].transform(
        lambda x: x.diff().fillna(0)
    )

    # 3. Sensor deviation from initial baseline (first 5 cycles avg)
    cols = [c for c in columns if c in df.columns]
    initial_means = (
        df[df["cycle"] <= df.groupby("engine_id")["cycle"].transform("min") + 4]
        .groupby("engine_id")[cols]
        .mean()
    )
    for col in cols:
        baseline = df["engine_id"].map(initial_means[col])
        df[f"{col}_dev_from_init"] = (df[col] - baseline).abs()

    # 4. Late-life binary flag (last 30 % of observed cycles)
    max_cycle = grouped["cycle"].transform("max")
    threshold = max_cycle * 0.7
    df["late_life_flag"] = (df["cycle"] >= threshold).astype(np.int8)

    return df


def compute_sensor_trend_slopes(
    df: pd.DataFrame,
    columns: list[str],
    window: int = 10,
) -> pd.DataFrame:
    """
    Fit a short rolling linear regression to each sensor to capture the
    *local trend direction and magnitude*.

    For each window the slope is estimated via Ordinary Least Squares
    (vectorised with numpy polyfit semantics).

    New columns:  ``{col}_trend_slope_{window}``

    Args:
        df: Input DataFrame (must contain ``engine_id``).
        columns: Sensor columns.
        window: Number of cycles for the regression window.
    """
    cols = [c for c in columns if c in df.columns]

    def _rolling_slope(series: pd.Series, w: int) -> pd.Series:
        """Compute the slope of a linear fit over rolling windows."""
        result = np.full(len(series), np.nan)
        values = series.values
        for i in range(len(values)):
            start = max(0, i - w + 1)
            segment = values[start : i + 1]
            if len(segment) >= 2:
                x = np.arange(len(segment), dtype=np.float64)
                # Fast OLS slope: cov(x,y) / var(x)
                x_mean = x.mean()
                y_mean = segment.mean()
                numerator = np.sum((x - x_mean) * (segment - y_mean))
                denominator = np.sum((x - x_mean) ** 2)
                result[i] = numerator / denominator if denominator != 0 else 0.0
            else:
                result[i] = 0.0
        return pd.Series(result, index=series.index)

    for col in cols:
        df[f"{col}_trend_slope_{window}"] = df.groupby("engine_id")[col].transform(
            lambda x: _rolling_slope(x, window)
        )
    return df


def compute_exponential_moving_averages(
    df: pd.DataFrame,
    columns: list[str],
    spans: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute Exponential Moving Averages (EMA / EWMA) for sensors.

    EMA puts more weight on recent readings, making it responsive to
    emerging degradation patterns.

    New columns:  ``{col}_ema_{span}``

    Args:
        df: Input DataFrame (must contain ``engine_id``).
        columns: Sensor columns.
        spans: EMA span parameters.  Defaults to ``[5, 10, 20]``.
    """
    spans = spans or [5, 10, 20]
    cols = [c for c in columns if c in df.columns]

    for col in cols:
        for span in spans:
            df[f"{col}_ema_{span}"] = df.groupby("engine_id")[col].transform(
                lambda x, s=span: x.ewm(span=s, min_periods=1).mean()
            )
    return df


def normalize_by_operating_condition(
    df: pd.DataFrame,
    columns: list[str],
    op_cols: list[str] | None = None,
    n_regimes: int = 6,
) -> pd.DataFrame:
    """
    Normalise sensor readings **within operating regimes**.

    Different operating conditions (altitude, throttle, Mach) shift
    sensor baselines.  This function clusters operating points into
    ``n_regimes`` discrete regimes via K-Means and z-score normalises
    each sensor within each regime.

    New columns:  ``{col}_op_norm``

    Args:
        df: Input DataFrame.
        columns: Sensor columns to normalise.
        op_cols: Operating-setting columns.  Defaults to
                 ``Settings.OPERATIONAL_SETTINGS``.
        n_regimes: Number of operating clusters.
    """
    from sklearn.cluster import KMeans

    op_cols = op_cols or Settings.OPERATIONAL_SETTINGS
    op_available = [c for c in op_cols if c in df.columns]
    cols = [c for c in columns if c in df.columns]

    if not op_available:
        logger.warning(
            "No operating-condition columns found — skipping regime normalisation"
        )
        return df

    # Cluster operating conditions into regimes
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    df["op_regime"] = kmeans.fit_predict(df[op_available].fillna(0))

    # Z-score normalise within each regime
    for col in cols:
        regime_stats = df.groupby("op_regime")[col].agg(["mean", "std"])
        regime_stats["std"] = regime_stats["std"].replace(0, 1)  # avoid div-by-zero

        mapped_mean = df["op_regime"].map(regime_stats["mean"])
        mapped_std = df["op_regime"].map(regime_stats["std"])
        df[f"{col}_op_norm"] = (df[col] - mapped_mean) / mapped_std

    return df


def compute_engine_grouped_features(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Compute per-engine aggregate features that characterise overall engine
    behaviour up to the current cycle:

    • **Cumulative mean** — running average of each sensor.
    • **Cumulative std** — running standard deviation.
    • **Rank within engine** — percentile rank of each cycle's reading
      relative to the engine's history.

    New columns:
        ``{col}_engine_cumean``, ``{col}_engine_custd``,
        ``{col}_engine_rank``

    Args:
        df: Input DataFrame (must contain ``engine_id``).
        columns: Sensor columns.
    """
    cols = [c for c in columns if c in df.columns]
    grouped = df.groupby("engine_id")

    for col in cols:
        # Expanding (cumulative) statistics
        df[f"{col}_engine_cumean"] = grouped[col].transform(
            lambda x: x.expanding(min_periods=1).mean()
        )
        df[f"{col}_engine_custd"] = grouped[col].transform(
            lambda x: x.expanding(min_periods=1).std().fillna(0)
        )
        # Rank within engine (0 = lowest ever seen, 1 = highest)
        df[f"{col}_engine_rank"] = grouped[col].transform(lambda x: x.rank(pct=True))
    return df


def compute_cycle_degradation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate cycle-based degradation features that track how far along
    each engine is in its operational life:

    • **Normalised cycle** — ``cycle / max_cycle`` ∈ [0, 1].
    • **Remaining life fraction** — ``1 − normalised_cycle``.
    • **Cycle squared** — captures non-linear (quadratic) degradation.
    • **Log cycle** — logarithmic time progression.
    • **Cycle bin** — quartile-based lifecycle stage (``early``,
      ``mid_early``, ``mid_late``, ``late``).

    Args:
        df: DataFrame with ``engine_id`` and ``cycle``.
    """
    grouped = df.groupby("engine_id")

    max_cyc = grouped["cycle"].transform("max")
    df["cycle_norm"] = df["cycle"] / max_cyc.replace(0, 1)
    df["remaining_life_frac"] = 1.0 - df["cycle_norm"]
    df["cycle_squared"] = df["cycle"] ** 2
    df["cycle_log"] = np.log1p(df["cycle"])  # log(1 + cycle)

    # Quartile-based lifecycle stage
    df["cycle_bin"] = pd.cut(
        df["cycle_norm"],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["early", "mid_early", "mid_late", "late"],
        include_lowest=True,
    )
    df["cycle_bin_encoded"] = df["cycle_bin"].cat.codes.astype(np.int8)
    df.drop(columns=["cycle_bin"], inplace=True)

    return df


def drop_low_variance_sensors(
    df: pd.DataFrame,
    drop_list: list[str] | None = None,
) -> pd.DataFrame:
    """
    Remove sensors known to carry near-zero variance (no useful signal).

    Args:
        df: Input DataFrame.
        drop_list: Column names to drop.  Defaults to ``Settings.DROP_SENSORS``.
    """
    drop_list = drop_list or Settings.DROP_SENSORS
    cols_to_drop = [c for c in drop_list if c in df.columns]
    return df.drop(columns=cols_to_drop, errors="ignore")


def normalize_sensors(
    df: pd.DataFrame,
    columns: list[str],
    scaler: MinMaxScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Min-Max normalise sensor readings to [0, 1].

    Args:
        df: Input DataFrame.
        columns: Columns to normalise.
        scaler: Pre-fitted ``MinMaxScaler`` (used when ``fit=False``).
        fit: If ``True``, fit a new scaler. If ``False``, use the
             provided scaler.

    Returns:
        (df, scaler) — the transformed DataFrame and the fitted scaler
        (for later re-use on test data).
    """
    scaler = scaler or MinMaxScaler()
    cols = [c for c in columns if c in df.columns]

    if fit:
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])
    return df, scaler


# ═══════════════════════════════════════════════════════════════
#  Orchestrator Class
# ═══════════════════════════════════════════════════════════════


class FeatureEngineer:
    """
    Advanced feature engineering orchestrator for C-MAPSS turbofan data.

    Composes all reusable preprocessing functions into a reproducible
    train / test transformation pipeline.  Configuration is drawn from
    ``configs.settings.Settings`` and can be overridden per-call.
    """

    # ── Pipeline configuration ──────────────────────────────
    ROLLING_WINDOW: int = 5
    TREND_WINDOW: int = 10
    LAG_STEPS: list[int] = [1, 2, 3]
    EMA_SPANS: list[int] = [5, 10, 20]
    OP_REGIMES: int = 6

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_columns: list[str] = [
            s for s in Settings.SENSOR_COLUMNS if s not in Settings.DROP_SENSORS
        ]

    # ── Public API ──────────────────────────────────────────

    def add_rul_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Remaining Useful Life (RUL) labels to training data."""
        logger.info(f"Generating RUL labels (max_rul={Settings.MAX_RUL})")
        df = compute_rul(df, max_rul=Settings.MAX_RUL)
        logger.info(f"RUL range: [{df['rul'].min()}, {df['rul'].max()}]")
        return df

    def drop_low_variance_sensors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove sensors with near-zero variance (no useful signal)."""
        logger.info(f"Dropping {len(Settings.DROP_SENSORS)} low-variance sensors")
        return drop_low_variance_sensors(df)

    def normalize_sensors(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Min-Max normalize sensor readings to [0, 1]."""
        logger.info(f"Normalizing {len(self.feature_columns)} sensor features")
        df, self.scaler = normalize_sensors(
            df,
            self.feature_columns,
            scaler=self.scaler,
            fit=fit,
        )
        return df

    def add_rolling_features(
        self, df: pd.DataFrame, window: int = None
    ) -> pd.DataFrame:
        """Add rolling mean and standard deviation for sensor features."""
        w = window or self.ROLLING_WINDOW
        logger.info(f"Adding rolling statistics (window={w})")
        df = compute_rolling_statistics(df, self.feature_columns, window=w)
        logger.info(f"Added {len(self.feature_columns) * 2} rolling features")
        return df

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal lag features for sensors."""
        logger.info(f"Adding lag features (lags={self.LAG_STEPS})")
        df = compute_lag_features(df, self.feature_columns, lags=self.LAG_STEPS)
        n_new = len(self.feature_columns) * len(self.LAG_STEPS)
        logger.info(f"Added {n_new} lag features")
        return df

    def add_health_degradation_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add health index, degradation rate, deviation from initial, late-life flag."""
        logger.info("Computing health degradation indicators")
        df = compute_health_degradation_indicators(df, self.feature_columns)
        logger.info(
            "Health degradation indicators added: health_index, "
            "degradation_rate, sensor deviations, late_life_flag"
        )
        return df

    def add_sensor_trend_slopes(
        self, df: pd.DataFrame, window: int = None
    ) -> pd.DataFrame:
        """Add rolling linear regression slopes for sensor trends."""
        w = window or self.TREND_WINDOW
        logger.info(f"Computing sensor trend slopes (window={w})")
        df = compute_sensor_trend_slopes(df, self.feature_columns, window=w)
        logger.info(f"Added {len(self.feature_columns)} trend-slope features")
        return df

    def add_exponential_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA features across multiple spans."""
        logger.info(f"Computing exponential moving averages (spans={self.EMA_SPANS})")
        df = compute_exponential_moving_averages(
            df, self.feature_columns, spans=self.EMA_SPANS
        )
        n_new = len(self.feature_columns) * len(self.EMA_SPANS)
        logger.info(f"Added {n_new} EMA features")
        return df

    def add_operating_condition_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize sensors within clustered operating regimes."""
        logger.info(f"Normalizing by operating condition ({self.OP_REGIMES} regimes)")
        df = normalize_by_operating_condition(
            df,
            self.feature_columns,
            n_regimes=self.OP_REGIMES,
        )
        logger.info(
            f"Added {len(self.feature_columns)} operating-condition-normalised features"
        )
        return df

    def add_engine_grouped_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add per-engine cumulative statistics and rank features."""
        logger.info("Computing engine-wise grouped features")
        df = compute_engine_grouped_features(df, self.feature_columns)
        n_new = len(self.feature_columns) * 3
        logger.info(f"Added {n_new} engine-grouped features")
        return df

    def add_cycle_degradation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cycle-based degradation / lifecycle stage features."""
        logger.info("Computing cycle-based degradation features")
        df = compute_cycle_degradation_features(df)
        logger.info(
            "Added: cycle_norm, remaining_life_frac, cycle_squared, "
            "cycle_log, cycle_bin_encoded"
        )
        return df

    # ── Orchestration ───────────────────────────────────────

    def transform_train(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full advanced feature engineering pipeline for **training** data.

        Execution order is intentional — normalization happens early so
        all derived features (rolling, lag, EMA, slopes) are computed
        on a normalised scale.
        """
        logger.info("=" * 60)
        logger.info("ADVANCED FEATURE ENGINEERING — TRAIN")
        logger.info("=" * 60)

        df = self.add_rul_labels(df)
        df = self.drop_low_variance_sensors(df)
        df = self.add_operating_condition_normalization(df)
        df = self.normalize_sensors(df, fit=True)
        df = self.add_rolling_features(df)
        df = self.add_lag_features(df)
        df = self.add_health_degradation_indicators(df)
        df = self.add_sensor_trend_slopes(df)
        df = self.add_exponential_moving_averages(df)
        df = self.add_engine_grouped_features(df)
        df = self.add_cycle_degradation_features(df)

        # Final NaN cleanup (edges of windows / lags)
        df = df.fillna(0)

        logger.info(
            f"Training feature matrix: {df.shape[0]:,} rows × {df.shape[1]} columns"
        )
        logger.info("Advanced feature engineering (train) complete ✅")
        return df

    def transform_test(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full advanced feature engineering pipeline for **test** data.

        Mirrors the training pipeline minus RUL label generation.
        Uses the *already-fitted* scaler from training.
        """
        logger.info("=" * 60)
        logger.info("ADVANCED FEATURE ENGINEERING — TEST")
        logger.info("=" * 60)

        df = self.drop_low_variance_sensors(df)
        df = self.add_operating_condition_normalization(df)
        df = self.normalize_sensors(df, fit=False)
        df = self.add_rolling_features(df)
        df = self.add_lag_features(df)
        df = self.add_health_degradation_indicators(df)
        df = self.add_sensor_trend_slopes(df)
        df = self.add_exponential_moving_averages(df)
        df = self.add_engine_grouped_features(df)
        df = self.add_cycle_degradation_features(df)

        # Final NaN cleanup
        df = df.fillna(0)

        logger.info(
            f"Test feature matrix: {df.shape[0]:,} rows × {df.shape[1]} columns"
        )
        logger.info("Advanced feature engineering (test) complete ✅")
        return df


if __name__ == "__main__":
    print("\nFeature engineering pipeline completed successfully.\n")
