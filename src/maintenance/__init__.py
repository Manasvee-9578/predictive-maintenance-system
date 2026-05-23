"""
Maintenance Intelligence package — rule-based risk scoring,
recommendation generation, and dashboard-ready reporting.

Core classes:
    - ``MaintenanceRecommendationEngine`` — end-to-end pipeline
    - ``MaintenanceRules``               — configurable decision rules
    - ``RiskScorer``                      — composite risk computation

Reusable utilities:
    - ``maintenance_utils`` — I/O, formatting, alert generation
"""

from src.maintenance.maintenance_rules import MaintenanceRules
from src.maintenance.risk_scoring import RiskScorer
from src.maintenance.recommendation_engine import MaintenanceRecommendationEngine
from src.maintenance.maintenance_utils import (
    load_prediction_data,
    format_recommendation_card,
    generate_alert_message,
    save_recommendations_csv,
    save_risk_summary_csv,
)

__all__ = [
    "MaintenanceRecommendationEngine",
    "MaintenanceRules",
    "RiskScorer",
    "load_prediction_data",
    "format_recommendation_card",
    "generate_alert_message",
    "save_recommendations_csv",
    "save_risk_summary_csv",
]
