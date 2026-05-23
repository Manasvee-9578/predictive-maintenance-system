"""
╔══════════════════════════════════════════════════════════════╗
║   Data Pipeline — End-to-End Preprocessing Orchestrator     ║
╚══════════════════════════════════════════════════════════════╝

Orchestrates the full preprocessing workflow:
  1. Load raw data
  2. Validate integrity
  3. Advanced feature engineering
  4. Generate feature summary report
  5. Save processed data to outputs/processed/
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
import pandas as pd
from pathlib import Path
from datetime import datetime

from configs.settings import Settings
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.feature_engineering import FeatureEngineer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataPipeline:
    """End-to-end data preprocessing pipeline."""

    def __init__(self):
        self.loader = DataLoader()
        self.engineer = FeatureEngineer()

    def run(self, save: bool = True) -> tuple:
        """
        Execute the full preprocessing pipeline.

        Returns:
            tuple: (train_df, test_df, rul_df) — processed DataFrames
        """
        logger.info("Starting data preprocessing pipeline...")

        # Step 1: Load raw data
        train_df, test_df, rul_df = self.loader.load_all()

        # Step 2: Validate
        self.loader.validate_data(train_df, "train")
        self.loader.validate_data(test_df, "test")

        # Step 3: Advanced feature engineering
        train_df = self.engineer.transform_train(train_df)
        test_df = self.engineer.transform_test(test_df)

        # Step 4: Save processed data & report
        if save:
            self._save_processed(train_df, test_df, rul_df)
            self._save_feature_report(train_df, test_df)

        logger.success("Data pipeline completed successfully ✅")
        return train_df, test_df, rul_df

    def _get_output_dir(self) -> Path:
        """Return the outputs/processed/ directory, creating if needed."""
        output_dir = Settings.OUTPUT_DIR / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _save_processed(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        rul_df: pd.DataFrame,
    ):
        """Save processed DataFrames to outputs/processed/."""
        output_dir = self._get_output_dir()

        train_path = output_dir / "train_processed.csv"
        test_path = output_dir / "test_processed.csv"
        rul_path = output_dir / "rul_processed.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        rul_df.to_csv(rul_path, index=True)

        logger.info(f"Processed data saved to {output_dir}")
        logger.info(
            f"  ▸ {train_path.name}  ({train_df.shape[0]:,} × {train_df.shape[1]})"
        )
        logger.info(
            f"  ▸ {test_path.name}   ({test_df.shape[0]:,} × {test_df.shape[1]})"
        )
        logger.info(f"  ▸ {rul_path.name}     ({len(rul_df)} engines)")

        # Also persist to data/processed/ for backward compatibility
        compat_dir = Settings.PROCESSED_DIR
        compat_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(compat_dir / "train_processed.csv", index=False)
        test_df.to_csv(compat_dir / "test_processed.csv", index=False)
        rul_df.to_csv(compat_dir / "rul_processed.csv", index=True)

    def _save_feature_report(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        """Generate and save a feature engineering summary report."""
        output_dir = self._get_output_dir()
        report_path = output_dir / "feature_engineering_report.txt"

        # Categorize features
        base_sensor = [
            c
            for c in train_df.columns
            if c.startswith("sensor_") and "_" not in c.split("sensor_")[1]
        ]
        rolling_feats = [
            c for c in train_df.columns if "_roll_mean_" in c or "_roll_std_" in c
        ]
        lag_feats = [c for c in train_df.columns if "_lag_" in c]
        trend_feats = [c for c in train_df.columns if "_trend_slope_" in c]
        ema_feats = [c for c in train_df.columns if "_ema_" in c]
        op_norm_feats = [c for c in train_df.columns if "_op_norm" in c]
        dev_feats = [c for c in train_df.columns if "_dev_from_init" in c]
        engine_feats = [c for c in train_df.columns if "_engine_" in c]
        cycle_feats = [
            c
            for c in train_df.columns
            if c.startswith("cycle_")
            or c
            in [
                "remaining_life_frac",
                "health_index",
                "degradation_rate",
                "late_life_flag",
            ]
        ]

        lines = [
            "=" * 70,
            "  FEATURE ENGINEERING REPORT",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
            f"  Train shape : {train_df.shape[0]:>8,} rows × {train_df.shape[1]:>4} columns",
            f"  Test shape  : {test_df.shape[0]:>8,} rows × {test_df.shape[1]:>4} columns",
            "",
            "─" * 70,
            "  FEATURE CATEGORIES",
            "─" * 70,
            f"  Base sensors                     : {len(base_sensor):>4}",
            f"  Rolling statistics (mean + std)   : {len(rolling_feats):>4}",
            f"  Lag features                     : {len(lag_feats):>4}",
            f"  Trend slopes                     : {len(trend_feats):>4}",
            f"  Exponential moving averages      : {len(ema_feats):>4}",
            f"  Operating condition normalised   : {len(op_norm_feats):>4}",
            f"  Deviation from initial           : {len(dev_feats):>4}",
            f"  Engine-wise grouped              : {len(engine_feats):>4}",
            f"  Cycle / degradation              : {len(cycle_feats):>4}",
            "",
            "─" * 70,
            "  COMPLETE COLUMN LIST",
            "─" * 70,
        ]
        for i, col in enumerate(train_df.columns, 1):
            lines.append(f"  {i:>4}. {col}")

        lines.extend(["", "=" * 70])
        report_text = "\n".join(lines)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info(f"Feature report saved to {report_path}")


if __name__ == "__main__":
    print("\nData pipeline completed successfully.\n")
