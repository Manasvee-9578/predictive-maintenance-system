"""
╔══════════════════════════════════════════════════════════════╗
║   Risk Scoring — Composite Risk & Business Impact Metrics   ║
╚══════════════════════════════════════════════════════════════╝

Computes multi-dimensional risk scores that combine equipment
telemetry, anomaly intelligence, and RUL forecasts into
business-actionable metrics:

    • Equipment reliability score
    • Estimated downtime reduction
    • Estimated maintenance cost savings
    • Fleet-wide risk summary
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from configs.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ── Business Impact Constants ────────────────────────────────────


@dataclass
class BusinessParameters:
    """Configurable parameters for business impact estimation.

    These values should be calibrated to the specific industrial
    context.  Defaults represent a typical turbofan engine fleet.
    """

    # Cost per unplanned failure event ($)
    unplanned_failure_cost: float = 250_000.0

    # Cost per planned maintenance event ($)
    planned_maintenance_cost: float = 35_000.0

    # Average unplanned downtime per failure (hours)
    unplanned_downtime_hours: float = 72.0

    # Average planned downtime per maintenance (hours)
    planned_downtime_hours: float = 8.0

    # Revenue loss per hour of downtime ($)
    revenue_loss_per_hour: float = 5_000.0

    # Inspection cost ($)
    inspection_cost: float = 5_000.0

    # Component replacement cost ($)
    component_replacement_cost: float = 75_000.0

    # Baseline failure rate without predictive maintenance (%)
    baseline_failure_rate: float = 12.0

    # Predicted failure rate with this system (%)
    predicted_failure_rate: float = 3.0


class RiskScorer:
    """Computes composite risk scores and business impact metrics
    for individual equipment units and fleet-wide summaries.

    Usage::

        scorer = RiskScorer()
        reliability = scorer.compute_reliability_score(
            health=72.5, anomaly=0.35, rul=45.0
        )
        impact = scorer.compute_business_impact(recommendations_df)
    """

    def __init__(
        self,
        business_params: Optional[BusinessParameters] = None,
    ) -> None:
        self.params = business_params or BusinessParameters()
        logger.info("RiskScorer initialized")

    # ── Equipment Reliability Score ──────────────────────────

    def compute_reliability_score(
        self,
        health: float,
        anomaly: float,
        rul: float,
        degradation: float = 0.0,
    ) -> float:
        """Compute a 0-100 equipment reliability score.

        Reliability is a weighted combination of health stability,
        anomaly absence, remaining life, and degradation slowness.

        Args:
            health:      Health score (0-100)
            anomaly:     Mean anomaly score (0-1)
            rul:         Predicted remaining useful life (cycles)
            degradation: Degradation rate per cycle

        Returns:
            Reliability percentage (0.0 – 100.0)
        """
        max_rul = Settings.MAX_RUL

        # Health directly contributes to reliability
        health_factor = health / 100.0

        # Low anomaly → high reliability
        anomaly_factor = 1.0 - min(anomaly, 1.0)

        # Higher RUL → higher reliability
        rul_factor = min(rul / max_rul, 1.0)

        # Lower degradation → higher reliability
        max_deg = 0.05  # cap
        deg_factor = 1.0 - min(degradation / max_deg, 1.0)

        reliability = (
            health_factor * 0.30
            + anomaly_factor * 0.25
            + rul_factor * 0.30
            + deg_factor * 0.15
        ) * 100.0

        return round(np.clip(reliability, 0.0, 100.0), 2)

    def compute_reliability_batch(self, df: pd.DataFrame) -> pd.Series:
        """Compute reliability scores for every row in a DataFrame.

        Expects columns: ``health_score``, ``mean_anomaly_score``
        (or ``anomaly_score_norm``), ``predicted_rul``,
        ``degradation_rate``.

        Returns:
            pd.Series of reliability scores
        """
        anomaly_col = (
            "mean_anomaly_score"
            if "mean_anomaly_score" in df.columns
            else "anomaly_score_norm"
        )

        scores = df.apply(
            lambda row: self.compute_reliability_score(
                health=float(row.get("health_score", 100)),
                anomaly=float(row.get(anomaly_col, 0)),
                rul=float(row.get("predicted_rul", Settings.MAX_RUL)),
                degradation=float(row.get("degradation_rate", 0)),
            ),
            axis=1,
        )

        logger.info(
            f"Reliability scores — mean: {scores.mean():.1f}, "
            f"min: {scores.min():.1f}, max: {scores.max():.1f}"
        )
        return scores

    # ── Business Impact Metrics ──────────────────────────────

    def compute_business_impact(
        self,
        recommendations_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Estimate business impact of predictive maintenance.

        Calculates downtime reduction, cost savings, and
        operational improvements versus a reactive-only strategy.

        Args:
            recommendations_df: DataFrame with ``maintenance_priority``,
                ``failure_risk_pct``, ``recommended_action`` columns

        Returns:
            Dictionary with business impact metrics
        """
        p = self.params
        n_equipment = len(recommendations_df)

        if n_equipment == 0:
            logger.warning("Empty DataFrame — returning zero impact")
            return self._zero_impact()

        # ── Baseline (reactive-only) costs ───────────────────
        expected_failures_reactive = n_equipment * (p.baseline_failure_rate / 100.0)
        reactive_cost = expected_failures_reactive * p.unplanned_failure_cost
        reactive_downtime = expected_failures_reactive * p.unplanned_downtime_hours

        # ── Predictive maintenance costs ─────────────────────
        from src.maintenance.maintenance_rules import (
            PRIORITY_CRITICAL,
            PRIORITY_HIGH,
            PRIORITY_MEDIUM,
            ACTION_REPLACE,
            ACTION_INSPECT,
            ACTION_SCHEDULE,
        )

        priorities = recommendations_df["maintenance_priority"]
        actions = recommendations_df["recommended_action"]

        n_critical = (priorities == PRIORITY_CRITICAL).sum()
        n_high = (priorities == PRIORITY_HIGH).sum()
        n_medium = (priorities == PRIORITY_MEDIUM).sum()

        n_replace = (actions == ACTION_REPLACE).sum()
        n_inspect = (actions == ACTION_INSPECT).sum()
        n_schedule = (actions == ACTION_SCHEDULE).sum()

        # With predictive maintenance, most failures are caught early
        expected_failures_predictive = n_equipment * (p.predicted_failure_rate / 100.0)

        predictive_cost = (
            expected_failures_predictive * p.unplanned_failure_cost
            + n_replace * p.component_replacement_cost
            + n_inspect * p.inspection_cost
            + n_schedule * p.planned_maintenance_cost
        )

        predictive_downtime = (
            expected_failures_predictive * p.unplanned_downtime_hours
            + (n_replace + n_schedule) * p.planned_downtime_hours
            + n_inspect * 2.0  # inspections are quick
        )

        # ── Savings ──────────────────────────────────────────
        cost_savings = max(0.0, reactive_cost - predictive_cost)
        downtime_reduction = max(0.0, reactive_downtime - predictive_downtime)

        cost_savings_pct = (
            (cost_savings / reactive_cost * 100.0) if reactive_cost > 0 else 0.0
        )
        downtime_reduction_pct = (
            (downtime_reduction / reactive_downtime * 100.0)
            if reactive_downtime > 0
            else 0.0
        )

        # ── Fleet reliability ────────────────────────────────
        if "equipment_reliability" in recommendations_df.columns:
            fleet_reliability = float(
                recommendations_df["equipment_reliability"].mean()
            )
        else:
            fleet_reliability = 100.0 - (
                recommendations_df.get("failure_risk_pct", pd.Series([0])).mean()
            )

        impact = {
            "fleet_size": n_equipment,
            "critical_equipment_count": int(n_critical),
            "high_priority_count": int(n_high),
            "medium_priority_count": int(n_medium),
            "estimated_cost_savings_usd": round(cost_savings, 2),
            "cost_savings_pct": round(cost_savings_pct, 2),
            "estimated_downtime_reduction_hours": round(downtime_reduction, 2),
            "downtime_reduction_pct": round(downtime_reduction_pct, 2),
            "reactive_annual_cost_usd": round(reactive_cost, 2),
            "predictive_annual_cost_usd": round(predictive_cost, 2),
            "fleet_reliability_score": round(fleet_reliability, 2),
            "failure_prevention_rate": round(
                (1 - p.predicted_failure_rate / p.baseline_failure_rate) * 100, 2
            ),
        }

        logger.info(
            f"Business impact — Cost savings: ${cost_savings:,.0f} "
            f"({cost_savings_pct:.1f}%), Downtime reduction: "
            f"{downtime_reduction:.0f}h ({downtime_reduction_pct:.1f}%)"
        )

        return impact

    def _zero_impact(self) -> Dict[str, float]:
        """Return a zeroed-out impact dictionary."""
        return {
            "fleet_size": 0,
            "critical_equipment_count": 0,
            "high_priority_count": 0,
            "medium_priority_count": 0,
            "estimated_cost_savings_usd": 0.0,
            "cost_savings_pct": 0.0,
            "estimated_downtime_reduction_hours": 0.0,
            "downtime_reduction_pct": 0.0,
            "reactive_annual_cost_usd": 0.0,
            "predictive_annual_cost_usd": 0.0,
            "fleet_reliability_score": 0.0,
            "failure_prevention_rate": 0.0,
        }

    # ── Risk Summary Report ──────────────────────────────────

    def generate_risk_summary(
        self,
        recommendations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate a per-equipment risk summary table.

        Args:
            recommendations_df: DataFrame with recommendation columns

        Returns:
            Summarized DataFrame with key risk columns for dashboards
        """
        cols_to_include = [
            "engine_id",
            "predicted_rul",
            "health_score",
            "failure_risk_pct",
            "urgency_score",
            "maintenance_priority",
            "recommended_action",
            "equipment_reliability",
        ]

        available = [c for c in cols_to_include if c in recommendations_df.columns]
        summary = recommendations_df[available].copy()

        # Sort by urgency (most urgent first)
        if "urgency_score" in summary.columns:
            summary = summary.sort_values("urgency_score", ascending=False)

        # Add risk tier
        if "failure_risk_pct" in summary.columns:
            summary["risk_tier"] = pd.cut(
                summary["failure_risk_pct"],
                bins=[-1, 25, 50, 75, 101],
                labels=["Low", "Moderate", "Elevated", "Critical"],
            )

        logger.info(f"Risk summary generated for {len(summary)} units")
        return summary.reset_index(drop=True)

    # ── Degradation Trend Analysis ───────────────────────────

    def analyze_degradation_trends(
        self,
        trend_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute per-engine degradation statistics from cycle-level data.

        Expects columns: ``engine_id``, ``cycle``, ``predicted_rul``,
        and optionally sensor EMA columns.

        Returns:
            DataFrame with engine-level degradation metrics:
            ``rul_decline_rate``, ``avg_rul``, ``min_rul``,
            ``rul_volatility``, ``cycles_observed``
        """
        grouped = trend_df.groupby("engine_id")

        stats = grouped.agg(
            cycles_observed=("cycle", "count"),
            avg_rul=("predicted_rul", "mean"),
            min_rul=("predicted_rul", "min"),
            max_rul=("predicted_rul", "max"),
            rul_volatility=("predicted_rul", "std"),
        ).reset_index()

        # Compute decline rate: (first RUL - last RUL) / cycles
        decline_rates = []
        for engine_id, group in grouped:
            group_sorted = group.sort_values("cycle")
            if len(group_sorted) >= 2:
                first_rul = group_sorted["predicted_rul"].iloc[0]
                last_rul = group_sorted["predicted_rul"].iloc[-1]
                n_cycles = (
                    group_sorted["cycle"].iloc[-1] - group_sorted["cycle"].iloc[0]
                )
                rate = (first_rul - last_rul) / max(n_cycles, 1)
            else:
                rate = 0.0
            decline_rates.append(
                {"engine_id": engine_id, "rul_decline_rate": round(rate, 6)}
            )

        decline_df = pd.DataFrame(decline_rates)
        stats = stats.merge(decline_df, on="engine_id", how="left")

        # Round numeric columns
        for col in ["avg_rul", "min_rul", "max_rul", "rul_volatility"]:
            stats[col] = stats[col].round(2)

        logger.info(
            f"Degradation analysis — {len(stats)} engines, "
            f"avg decline rate: {stats['rul_decline_rate'].mean():.4f}/cycle"
        )

        return stats
