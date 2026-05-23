"""
╔══════════════════════════════════════════════════════════════╗
║   Model Evaluator — Metrics & Performance Analysis          ║
╚══════════════════════════════════════════════════════════════╝

Computes evaluation metrics for RUL prediction models:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- Custom scoring function (asymmetric penalty)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

from configs.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelEvaluator:
    """Evaluate RUL prediction models with standard and custom metrics."""

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² (Coefficient of Determination)."""
        return r2_score(y_true, y_pred)

    @staticmethod
    def scoring_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        NASA-style asymmetric scoring function.

        Penalizes late predictions (under-estimating RUL) more heavily than
        early predictions, since late maintenance is more dangerous.

        S = Σ exp(-d/13) - 1   if d < 0 (late prediction)
        S = Σ exp(d/10) - 1    if d >= 0 (early prediction)
        """
        d = y_pred - y_true
        score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
        return np.sum(score)

    def compute_all_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"
    ) -> dict:
        """Compute all evaluation metrics for a model."""
        metrics = {
            "model": model_name,
            "rmse": self.rmse(y_true, y_pred),
            "mae": self.mae(y_true, y_pred),
            "r2": self.r2(y_true, y_pred),
            "scoring_fn": self.scoring_function(y_true, y_pred),
        }

        logger.info(f"{'─' * 40}")
        logger.info(f"  {model_name} Evaluation Results")
        logger.info(f"{'─' * 40}")
        logger.info(f"  RMSE:       {metrics['rmse']:.4f}")
        logger.info(f"  MAE:        {metrics['mae']:.4f}")
        logger.info(f"  R²:         {metrics['r2']:.4f}")
        logger.info(f"  Score (S):  {metrics['scoring_fn']:.2f}")
        logger.info(f"{'─' * 40}")

        return metrics

    def evaluate_all(self, test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate all saved models on test data.

        Returns:
            DataFrame with comparison of all model metrics
        """
        logger.info("Evaluating all trained models...")

        # Placeholder — actual evaluation requires loading models and running predictions
        # This method is designed to be extended during model training
        results = []

        # Save results
        output_path = Settings.OUTPUT_DIR / "reports" / "model_comparison.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Evaluation results saved to {output_path}")
            return results_df

        logger.warning("No models evaluated — train models first")
        return pd.DataFrame()

    def generate_report(self, results: list, output_path: Path = None):
        """Generate a formatted evaluation report."""
        output_path = (
            output_path or Settings.OUTPUT_DIR / "reports" / "evaluation_report.txt"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("  Model Evaluation Report\n")
            f.write("=" * 60 + "\n\n")

            for r in results:
                f.write(f"Model: {r['model']}\n")
                f.write(f"  RMSE:  {r['rmse']:.4f}\n")
                f.write(f"  MAE:   {r['mae']:.4f}\n")
                f.write(f"  R²:    {r['r2']:.4f}\n")
                f.write(f"  Score: {r['scoring_fn']:.2f}\n")
                f.write("-" * 40 + "\n")

        logger.info(f"Evaluation report saved to {output_path}")
