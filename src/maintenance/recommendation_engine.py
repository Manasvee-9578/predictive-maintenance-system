"""
╔══════════════════════════════════════════════════════════════╗
║   Maintenance Recommendation Engine                         ║
║   End-to-End Intelligent Maintenance Intelligence Pipeline  ║
╚══════════════════════════════════════════════════════════════╝

Orchestrates the full maintenance recommendation workflow:

    1. Load anomaly scores, health scores, and RUL predictions
    2. Merge into a unified equipment profile
    3. Apply maintenance rules to classify priority & actions
    4. Compute business impact metrics
    5. Generate dashboard-ready outputs (CSVs, charts, alerts)

Usage::

    engine = MaintenanceRecommendationEngine()
    results = engine.run()
    # results contains recommendations, risk summary, impact metrics,
    # charts, and alerts — all ready for dashboard integration.
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

from configs.settings import Settings
from src.utils.logger import setup_logger
from src.utils.helpers import timer

from src.maintenance.maintenance_rules import (
    MaintenanceRules,
    RuleThresholds,
)
from src.maintenance.risk_scoring import (
    RiskScorer,
    BusinessParameters,
)
from src.maintenance.maintenance_utils import (
    load_prediction_data,
    merge_equipment_data,
    format_recommendation_card,
    generate_machine_summary,
    generate_alert_report,
    save_recommendations_csv,
    save_risk_summary_csv,
    save_figure,
    plot_maintenance_priority_distribution,
    plot_risk_distribution,
    plot_equipment_criticality_dashboard,
    plot_recommendation_action_chart,
    plot_urgency_heatmap,
    plot_business_impact_summary,
)

logger = setup_logger(__name__)


class MaintenanceRecommendationEngine:
    """End-to-end maintenance intelligence pipeline.

    Combines anomaly detection outputs, health scores, and RUL
    predictions to generate actionable maintenance recommendations
    with business impact analysis.

    Attributes:
        rules:  Maintenance rule engine
        scorer: Risk and business impact scorer
    """

    def __init__(
        self,
        thresholds: Optional[RuleThresholds] = None,
        business_params: Optional[BusinessParameters] = None,
        predictions_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the recommendation engine.

        Args:
            thresholds:      Custom rule thresholds (default: standard)
            business_params: Custom business parameters (default: turbofan)
            predictions_dir: Directory containing prediction CSVs
            output_dir:      Directory for output files
        """
        self.rules = MaintenanceRules(thresholds)
        self.scorer = RiskScorer(business_params)
        self.predictions_dir = predictions_dir or (Settings.OUTPUT_DIR / "predictions")
        self.output_dir = output_dir or Settings.OUTPUT_DIR
        self._results: Dict[str, Any] = {}

        logger.info("╔══════════════════════════════════════════════════════════╗")
        logger.info("║   Maintenance Recommendation Engine Initialized         ║")
        logger.info("╚══════════════════════════════════════════════════════════╝")

    # ── Main Pipeline ────────────────────────────────────────

    @timer
    def run(
        self,
        save_outputs: bool = True,
        generate_charts: bool = True,
    ) -> Dict[str, Any]:
        """Execute the full recommendation pipeline.

        Args:
            save_outputs:    Save CSVs to disk
            generate_charts: Generate and save visualization HTML files

        Returns:
            Dictionary with keys:
                ``recommendations``  — full recommendations DataFrame
                ``risk_summary``     — risk summary DataFrame
                ``business_impact``  — business impact metrics dict
                ``machine_summaries``— list of card dicts
                ``alert_report``     — alert report string
                ``charts``           — dict of Plotly Figures
                ``degradation_stats``— degradation analysis DataFrame
        """
        logger.info("=" * 60)
        logger.info("MAINTENANCE RECOMMENDATION ENGINE — Starting Pipeline")
        logger.info("=" * 60)

        # ── Step 1: Load data ────────────────────────────────
        logger.info("Step 1/6: Loading prediction data...")
        data = self._load_data()

        # ── Step 2: Merge equipment profiles ─────────────────
        logger.info("Step 2/6: Merging equipment profiles...")
        merged_df = self._merge_profiles(data)

        if merged_df.empty:
            logger.error("No data available — pipeline aborted")
            return {"error": "No data available for recommendation generation"}

        # ── Step 3: Generate recommendations ─────────────────
        logger.info("Step 3/6: Generating maintenance recommendations...")
        recommendations_df = self._generate_recommendations(merged_df)

        # ── Step 4: Compute business impact ──────────────────
        logger.info("Step 4/6: Computing business impact metrics...")
        business_impact = self._compute_impact(recommendations_df)

        # ── Step 5: Generate risk summary & alerts ───────────
        logger.info("Step 5/6: Generating risk summary and alerts...")
        risk_summary = self.scorer.generate_risk_summary(recommendations_df)
        machine_summaries = generate_machine_summary(recommendations_df)
        alert_report = generate_alert_report(recommendations_df)

        # ── Step 6: Degradation trend analysis ───────────────
        logger.info("Step 6/6: Analyzing degradation trends...")
        degradation_stats = self._analyze_trends(data)

        # ── Assemble results ─────────────────────────────────
        self._results = {
            "recommendations": recommendations_df,
            "risk_summary": risk_summary,
            "business_impact": business_impact,
            "machine_summaries": machine_summaries,
            "alert_report": alert_report,
            "degradation_stats": degradation_stats,
            "charts": {},
        }

        # ── Save outputs ─────────────────────────────────────
        if save_outputs:
            self._save_outputs(recommendations_df, risk_summary)

        # ── Generate visualizations ──────────────────────────
        if generate_charts:
            self._results["charts"] = self._generate_charts(
                recommendations_df, business_impact, save_outputs
            )

        # ── Summary log ──────────────────────────────────────
        self._log_summary(recommendations_df, business_impact)

        return self._results

    # ── Pipeline Steps ───────────────────────────────────────

    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all prediction data from disk."""
        try:
            data = load_prediction_data(self.predictions_dir)
            loaded = {k: len(v) for k, v in data.items() if not v.empty}
            logger.info(f"  Loaded datasets: {loaded}")
            return data
        except Exception as e:
            logger.error(f"Failed to load prediction data: {e}")
            raise

    def _merge_profiles(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all data sources into unified equipment profiles."""
        rul_df = data.get("rul", pd.DataFrame())
        health_df = data.get("health_pred", pd.DataFrame())

        # Fall back to training health scores if prediction set is empty
        if health_df.empty:
            health_df = data.get("health_train", pd.DataFrame())
            logger.info("  Using training health scores (prediction set unavailable)")

        anomaly_df = data.get("anomaly_batch", pd.DataFrame())

        merged = merge_equipment_data(rul_df, health_df, anomaly_df)
        logger.info(
            f"  Merged profile: {merged.shape[0]} engines, {merged.shape[1]} features"
        )
        return merged

    def _generate_recommendations(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Apply maintenance rules and compute scores."""
        # Apply rule engine
        recommendations = self.rules.classify_batch(merged_df)

        # Compute equipment reliability scores
        recommendations["equipment_reliability"] = (
            self.scorer.compute_reliability_batch(recommendations)
        )

        # Add recommendation metadata
        recommendations["recommendation_timestamp"] = pd.Timestamp.now().isoformat(
            timespec="seconds"
        )

        logger.info(f"  Generated {len(recommendations)} recommendations")
        return recommendations

    def _compute_impact(self, recommendations_df: pd.DataFrame) -> Dict[str, float]:
        """Compute business impact metrics."""
        try:
            impact = self.scorer.compute_business_impact(recommendations_df)
            return impact
        except Exception as e:
            logger.error(f"Business impact computation failed: {e}")
            return self.scorer._zero_impact()

    def _analyze_trends(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Analyze degradation trends from cycle-level data."""
        # Check for trend data
        trend_path = self.predictions_dir / "rul_training_trends.csv"
        if trend_path.exists():
            try:
                trend_df = pd.read_csv(trend_path)
                return self.scorer.analyze_degradation_trends(trend_df)
            except Exception as e:
                logger.warning(f"Degradation trend analysis failed: {e}")

        logger.info("  No trend data available — skipping degradation analysis")
        return pd.DataFrame()

    def _save_outputs(
        self,
        recommendations_df: pd.DataFrame,
        risk_summary: pd.DataFrame,
    ) -> None:
        """Save all output CSVs."""
        pred_dir = self.output_dir / "predictions"
        save_recommendations_csv(recommendations_df, pred_dir)
        save_risk_summary_csv(risk_summary, pred_dir)

    def _generate_charts(
        self,
        recommendations_df: pd.DataFrame,
        business_impact: Dict[str, float],
        save_to_disk: bool = True,
    ) -> Dict[str, go.Figure]:
        """Generate all dashboard visualizations."""
        charts: Dict[str, go.Figure] = {}
        plots_dir = self.output_dir / "plots"

        try:
            # 1. Priority distribution
            charts["priority_distribution"] = plot_maintenance_priority_distribution(
                recommendations_df
            )

            # 2. Risk distribution
            charts["risk_distribution"] = plot_risk_distribution(recommendations_df)

            # 3. Equipment criticality dashboard
            charts["criticality_dashboard"] = plot_equipment_criticality_dashboard(
                recommendations_df
            )

            # 4. Recommended actions
            charts["action_distribution"] = plot_recommendation_action_chart(
                recommendations_df
            )

            # 5. Urgency heatmap
            charts["urgency_heatmap"] = plot_urgency_heatmap(recommendations_df)

            # 6. Business impact KPIs
            charts["business_impact"] = plot_business_impact_summary(business_impact)

            if save_to_disk:
                for name, fig in charts.items():
                    save_figure(fig, plots_dir, f"maintenance_{name}.html")

            logger.info(f"  Generated {len(charts)} dashboard charts")

        except Exception as e:
            logger.error(f"Chart generation failed: {e}")

        return charts

    def _log_summary(
        self,
        recommendations_df: pd.DataFrame,
        business_impact: Dict[str, float],
    ) -> None:
        """Log a human-readable summary of results."""
        priorities = recommendations_df["maintenance_priority"].value_counts()
        actions = recommendations_df["recommended_action"].value_counts()

        logger.info("")
        logger.info("╔══════════════════════════════════════════════════════════╗")
        logger.info("║   MAINTENANCE RECOMMENDATION SUMMARY                    ║")
        logger.info("╚══════════════════════════════════════════════════════════╝")
        logger.info("")
        logger.info("  Priority Breakdown:")
        for priority, count in priorities.items():
            logger.info(f"    {priority:>10}: {count} engines")

        logger.info("")
        logger.info("  Recommended Actions:")
        for action, count in actions.items():
            logger.info(f"    {action:>35}: {count}")

        logger.info("")
        logger.info("  Business Impact:")
        logger.info(
            f"    Cost Savings:       ${business_impact.get('estimated_cost_savings_usd', 0):>12,.0f}"
        )
        logger.info(
            f"    Downtime Reduction: {business_impact.get('downtime_reduction_pct', 0):>11.1f}%"
        )
        logger.info(
            f"    Fleet Reliability:  {business_impact.get('fleet_reliability_score', 0):>11.1f}%"
        )
        logger.info(
            f"    Failure Prevention: {business_impact.get('failure_prevention_rate', 0):>11.1f}%"
        )

        logger.info("")
        logger.success("Pipeline execution completed successfully! ✅")

    # ── Accessors ────────────────────────────────────────────

    @property
    def recommendations(self) -> pd.DataFrame:
        """Get the latest recommendations DataFrame."""
        return self._results.get("recommendations", pd.DataFrame())

    @property
    def risk_summary(self) -> pd.DataFrame:
        """Get the latest risk summary DataFrame."""
        return self._results.get("risk_summary", pd.DataFrame())

    @property
    def business_impact(self) -> Dict[str, float]:
        """Get the latest business impact metrics."""
        return self._results.get("business_impact", {})

    @property
    def alert_report(self) -> str:
        """Get the latest alert report string."""
        return self._results.get("alert_report", "")

    @property
    def charts(self) -> Dict[str, go.Figure]:
        """Get the latest chart dictionary."""
        return self._results.get("charts", {})
