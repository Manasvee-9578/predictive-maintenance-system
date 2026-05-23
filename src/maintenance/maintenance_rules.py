"""
╔══════════════════════════════════════════════════════════════╗
║   Maintenance Rules — Configurable Decision Logic           ║
╚══════════════════════════════════════════════════════════════╝

Encapsulates the rule-based logic that maps sensor-derived metrics
(anomaly scores, health scores, predicted RUL, degradation trends)
into actionable maintenance recommendations.

Design:
    Each rule threshold is a class attribute so that values can be
    tuned from a config file or environment variables without
    modifying source code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ── Priority & Action Constants ──────────────────────────────────

PRIORITY_LOW = "Low"
PRIORITY_MEDIUM = "Medium"
PRIORITY_HIGH = "High"
PRIORITY_CRITICAL = "Critical"

ACTION_MONITOR = "Continue Monitoring"
ACTION_SCHEDULE = "Schedule Maintenance"
ACTION_INSPECT = "Immediate Inspection Required"
ACTION_REPLACE = "Replace Component Soon"

PRIORITY_ORDER = {
    PRIORITY_LOW: 0,
    PRIORITY_MEDIUM: 1,
    PRIORITY_HIGH: 2,
    PRIORITY_CRITICAL: 3,
}

PRIORITY_COLORS = {
    PRIORITY_LOW: "#2E7D32",
    PRIORITY_MEDIUM: "#F9A825",
    PRIORITY_HIGH: "#EF6C00",
    PRIORITY_CRITICAL: "#C62828",
}

ACTION_COLORS = {
    ACTION_MONITOR: "#2E7D32",
    ACTION_SCHEDULE: "#1565C0",
    ACTION_INSPECT: "#EF6C00",
    ACTION_REPLACE: "#C62828",
}


@dataclass
class RuleThresholds:
    """Configurable thresholds for maintenance decision logic.

    Each group of thresholds controls one dimension of the
    recommendation: RUL-based urgency, anomaly severity,
    health degradation, and degradation rate.
    """

    # ── RUL-based thresholds (cycles) ────────────────────────
    rul_critical: float = 15.0
    rul_high: float = 30.0
    rul_medium: float = 60.0
    rul_low: float = 100.0

    # ── Anomaly score thresholds (0-1 normalized) ────────────
    anomaly_critical: float = 0.80
    anomaly_high: float = 0.55
    anomaly_medium: float = 0.35

    # ── Health score thresholds (0-100) ──────────────────────
    health_critical: float = 40.0
    health_warning: float = 60.0
    health_normal: float = 80.0

    # ── Degradation rate thresholds (per-cycle) ──────────────
    degradation_severe: float = 0.025
    degradation_moderate: float = 0.015
    degradation_mild: float = 0.008

    # ── Confidence weighting ─────────────────────────────────
    low_confidence_boost: float = 0.5  # extra risk when confidence is low
    confidence_threshold: float = 0.50  # below this, confidence is "low"

    # ── Composite weights (must sum to 1.0) ──────────────────
    weight_rul: float = 0.35
    weight_anomaly: float = 0.25
    weight_health: float = 0.25
    weight_degradation: float = 0.15


class MaintenanceRules:
    """Rule engine that converts equipment telemetry into
    maintenance priorities, recommended actions, and alert levels.

    Usage::

        rules = MaintenanceRules()
        priority = rules.classify_priority(rul=12, anomaly=0.85,
                                            health=38, degradation=0.03)
        action = rules.recommend_action(priority, rul=12, anomaly=0.85)
    """

    def __init__(self, thresholds: Optional[RuleThresholds] = None) -> None:
        self.thresholds = thresholds or RuleThresholds()
        logger.info(
            "MaintenanceRules initialized with custom thresholds"
            if thresholds
            else "MaintenanceRules initialized with defaults"
        )

    # ── Single-dimension classifiers ─────────────────────────

    def _rul_priority(self, rul: float) -> str:
        """Classify priority based on predicted RUL alone."""
        t = self.thresholds
        if rul <= t.rul_critical:
            return PRIORITY_CRITICAL
        elif rul <= t.rul_high:
            return PRIORITY_HIGH
        elif rul <= t.rul_medium:
            return PRIORITY_MEDIUM
        return PRIORITY_LOW

    def _anomaly_priority(self, score: float) -> str:
        """Classify priority based on anomaly score alone."""
        t = self.thresholds
        if score >= t.anomaly_critical:
            return PRIORITY_CRITICAL
        elif score >= t.anomaly_high:
            return PRIORITY_HIGH
        elif score >= t.anomaly_medium:
            return PRIORITY_MEDIUM
        return PRIORITY_LOW

    def _health_priority(self, health: float) -> str:
        """Classify priority based on health score alone."""
        t = self.thresholds
        if health <= t.health_critical:
            return PRIORITY_CRITICAL
        elif health <= t.health_warning:
            return PRIORITY_HIGH
        elif health <= t.health_normal:
            return PRIORITY_MEDIUM
        return PRIORITY_LOW

    def _degradation_priority(self, rate: float) -> str:
        """Classify priority based on degradation rate alone."""
        t = self.thresholds
        if rate >= t.degradation_severe:
            return PRIORITY_CRITICAL
        elif rate >= t.degradation_moderate:
            return PRIORITY_HIGH
        elif rate >= t.degradation_mild:
            return PRIORITY_MEDIUM
        return PRIORITY_LOW

    # ── Composite classifier ─────────────────────────────────

    def classify_priority(
        self,
        rul: float,
        anomaly: float,
        health: float,
        degradation: float,
        confidence: float = 1.0,
    ) -> str:
        """Determine overall maintenance priority from all signals.

        The composite score is a weighted sum of individual signal
        severities (0-3) with a confidence penalty for uncertain
        predictions.

        Args:
            rul:         Predicted remaining useful life (cycles)
            anomaly:     Normalized anomaly score (0-1)
            health:      Equipment health score (0-100)
            degradation: Degradation rate per cycle
            confidence:  Prediction confidence (0-1)

        Returns:
            One of ``PRIORITY_LOW``, ``PRIORITY_MEDIUM``,
            ``PRIORITY_HIGH``, ``PRIORITY_CRITICAL``
        """
        scores = {
            "rul": PRIORITY_ORDER[self._rul_priority(rul)],
            "anomaly": PRIORITY_ORDER[self._anomaly_priority(anomaly)],
            "health": PRIORITY_ORDER[self._health_priority(health)],
            "degradation": PRIORITY_ORDER[self._degradation_priority(degradation)],
        }

        t = self.thresholds
        composite = (
            scores["rul"] * t.weight_rul
            + scores["anomaly"] * t.weight_anomaly
            + scores["health"] * t.weight_health
            + scores["degradation"] * t.weight_degradation
        )

        # Boost risk for low-confidence predictions
        if confidence < t.confidence_threshold:
            composite += t.low_confidence_boost

        # Map composite back to priority
        if composite >= 2.5:
            return PRIORITY_CRITICAL
        elif composite >= 1.8:
            return PRIORITY_HIGH
        elif composite >= 1.0:
            return PRIORITY_MEDIUM
        return PRIORITY_LOW

    # ── Action recommendation ────────────────────────────────

    def recommend_action(
        self,
        priority: str,
        rul: float,
        anomaly: float,
        health: float = 100.0,
    ) -> str:
        """Select the recommended maintenance action.

        The action depends primarily on the priority class with
        tie-breaking logic from individual metrics.

        Args:
            priority: Composite maintenance priority
            rul:      Predicted remaining useful life
            anomaly:  Normalized anomaly score
            health:   Health score (0-100)

        Returns:
            Recommended action string
        """
        if priority == PRIORITY_CRITICAL:
            if (
                rul <= self.thresholds.rul_critical
                and anomaly >= self.thresholds.anomaly_high
            ):
                return ACTION_REPLACE
            return ACTION_INSPECT

        if priority == PRIORITY_HIGH:
            if rul <= self.thresholds.rul_high:
                return ACTION_REPLACE
            return ACTION_INSPECT

        if priority == PRIORITY_MEDIUM:
            return ACTION_SCHEDULE

        return ACTION_MONITOR

    # ── Failure risk estimation ──────────────────────────────

    def estimate_failure_risk(
        self,
        rul: float,
        anomaly: float,
        health: float,
        degradation: float,
    ) -> float:
        """Compute a 0-100% failure risk percentage.

        Combines inverse RUL proximity, anomaly severity, inverse
        health, and degradation intensity.

        Returns:
            Failure risk percentage (0.0 – 100.0)
        """
        from configs.settings import Settings

        max_rul = Settings.MAX_RUL

        # Inverse RUL: closer to 0 → higher risk
        rul_risk = max(0.0, 1.0 - (rul / max_rul)) * 100

        # Anomaly directly maps to risk
        anomaly_risk = min(anomaly, 1.0) * 100

        # Inverse health
        health_risk = max(0.0, (100.0 - health))

        # Degradation rate (cap at severe threshold × 2)
        max_deg = self.thresholds.degradation_severe * 2
        deg_risk = min(degradation / max_deg, 1.0) * 100

        t = self.thresholds
        composite_risk = (
            rul_risk * t.weight_rul
            + anomaly_risk * t.weight_anomaly
            + health_risk * t.weight_health
            + deg_risk * t.weight_degradation
        )

        return round(np.clip(composite_risk, 0.0, 100.0), 2)

    # ── Urgency scoring ──────────────────────────────────────

    def compute_urgency_score(
        self,
        rul: float,
        anomaly: float,
        health: float,
        degradation: float,
    ) -> float:
        """Compute a 0-10 urgency score for maintenance scheduling.

        Higher values indicate more urgent need for intervention.

        Returns:
            Urgency score (0.0 – 10.0)
        """
        from configs.settings import Settings

        max_rul = Settings.MAX_RUL

        rul_urgency = max(0.0, 1.0 - (rul / max_rul)) * 10
        anomaly_urgency = min(anomaly, 1.0) * 10
        health_urgency = max(0.0, (100.0 - health) / 100.0) * 10
        deg_urgency = (
            min(degradation / (self.thresholds.degradation_severe * 2), 1.0) * 10
        )

        t = self.thresholds
        urgency = (
            rul_urgency * t.weight_rul
            + anomaly_urgency * t.weight_anomaly
            + health_urgency * t.weight_health
            + deg_urgency * t.weight_degradation
        )

        return round(np.clip(urgency, 0.0, 10.0), 2)

    # ── Vectorized batch processing ──────────────────────────

    def classify_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all rules to every row of a merged equipment DataFrame.

        Expected columns:
            ``predicted_rul``, ``mean_anomaly_score`` (or
            ``anomaly_score_norm``), ``health_score``,
            ``degradation_rate``, and optionally ``confidence``.

        Returns:
            DataFrame with new columns:
            ``maintenance_priority``, ``recommended_action``,
            ``failure_risk_pct``, ``urgency_score``
        """
        result = df.copy()

        # Resolve column names flexibly
        rul_col = "predicted_rul"
        anomaly_col = (
            "mean_anomaly_score"
            if "mean_anomaly_score" in result.columns
            else "anomaly_score_norm"
        )
        health_col = "health_score"
        deg_col = "degradation_rate"
        conf_col = "confidence" if "confidence" in result.columns else None

        priorities: List[str] = []
        actions: List[str] = []
        risks: List[float] = []
        urgencies: List[float] = []

        for _, row in result.iterrows():
            rul = float(row.get(rul_col, 125))
            anomaly = float(row.get(anomaly_col, 0))
            health = float(row.get(health_col, 100))
            degradation = float(row.get(deg_col, 0))
            confidence = float(row[conf_col]) if conf_col else 1.0

            priority = self.classify_priority(
                rul, anomaly, health, degradation, confidence
            )
            action = self.recommend_action(priority, rul, anomaly, health)
            risk = self.estimate_failure_risk(rul, anomaly, health, degradation)
            urgency = self.compute_urgency_score(rul, anomaly, health, degradation)

            priorities.append(priority)
            actions.append(action)
            risks.append(risk)
            urgencies.append(urgency)

        result["maintenance_priority"] = priorities
        result["recommended_action"] = actions
        result["failure_risk_pct"] = risks
        result["urgency_score"] = urgencies

        logger.info(
            f"Classified {len(result)} equipment records — "
            f"Critical: {priorities.count(PRIORITY_CRITICAL)}, "
            f"High: {priorities.count(PRIORITY_HIGH)}, "
            f"Medium: {priorities.count(PRIORITY_MEDIUM)}, "
            f"Low: {priorities.count(PRIORITY_LOW)}"
        )

        return result
