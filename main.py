"""
╔══════════════════════════════════════════════════════════════╗
║   Predictive Maintenance & Intelligent RUL Forecasting      ║
║   Main Application Entry Point                              ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python main.py                  # Run full pipeline
    python main.py --preprocess     # Run preprocessing only
    python main.py --train          # Run training only
    python main.py --evaluate       # Run evaluation only
    python main.py --dashboard      # Launch Streamlit dashboard
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.settings import Settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predictive Maintenance & RUL Forecasting Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  Run full pipeline
  python main.py --preprocess     Preprocess data only
  python main.py --train          Train models only
  python main.py --evaluate       Evaluate models only
  python main.py --dashboard      Launch Streamlit dashboard
        """,
    )
    parser.add_argument(
        "--preprocess", action="store_true", help="Run preprocessing pipeline"
    )
    parser.add_argument("--train", action="store_true", help="Train ML/DL models")
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate trained models"
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="Launch Streamlit dashboard"
    )
    parser.add_argument(
        "--recommend", action="store_true", help="Generate maintenance recommendations"
    )
    return parser.parse_args()


def run_preprocessing():
    """Execute the data preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Data Preprocessing & Feature Engineering")
    logger.info("=" * 60)

    from src.preprocessing.data_pipeline import DataPipeline

    pipeline = DataPipeline()
    train_df, test_df, rul_df = pipeline.run()

    logger.success(
        f"Preprocessing complete — Train: {train_df.shape}, Test: {test_df.shape}"
    )
    return train_df, test_df, rul_df


def run_anomaly_detection(train_df):
    """Execute anomaly detection on training data."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Anomaly Detection")
    logger.info("=" * 60)

    from src.anomaly_detection.detector_pipeline import DetectorPipeline

    detector = DetectorPipeline()
    anomaly_results = detector.run(train_df)

    logger.success(
        f"Anomaly detection complete — {len(anomaly_results)} results generated"
    )
    return anomaly_results


def run_training(train_df, test_df, rul_df):
    """Train RUL prediction models."""
    logger.info("=" * 60)
    logger.info("STAGE 3: Model Training")
    logger.info("=" * 60)

    from src.rul_prediction.model_trainer import ModelTrainer

    trainer = ModelTrainer()
    trained_models = trainer.train_all(train_df, test_df, rul_df)

    logger.success(f"Training complete — {len(trained_models)} models trained")
    return trained_models


def run_evaluation(test_df, rul_df):
    """Evaluate trained models."""
    logger.info("=" * 60)
    logger.info("STAGE 4: Model Evaluation")
    logger.info("=" * 60)

    from src.rul_prediction.model_evaluator import ModelEvaluator

    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all(test_df, rul_df)

    logger.success("Evaluation complete — results saved to outputs/reports/")
    return results


def run_maintenance_recommendations():
    """Generate intelligent maintenance recommendations."""
    logger.info("=" * 60)
    logger.info("STAGE 5: Maintenance Recommendations")
    logger.info("=" * 60)

    from src.maintenance.recommendation_engine import MaintenanceRecommendationEngine

    engine = MaintenanceRecommendationEngine()
    results = engine.run()

    logger.success(
        f"Maintenance recommendations generated for {len(engine.recommendations)} engines"
    )
    return results


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    logger.info("=" * 60)
    logger.info("Launching Streamlit Dashboard")
    logger.info("=" * 60)

    os.system(
        f"streamlit run src/dashboard/dashboard.py --server.port {Settings.STREAMLIT_PORT}"
    )


def main():
    """Main application orchestrator."""
    args = parse_args()
    settings = Settings()

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║   Predictive Maintenance & RUL Forecasting Platform     ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"Environment: {settings.APP_ENV} | Debug: {settings.DEBUG}")

    # If no specific flag is set, run the full pipeline
    run_all = not any(
        [args.preprocess, args.train, args.evaluate, args.recommend, args.dashboard]
    )

    try:
        if args.dashboard:
            launch_dashboard()
            return

        if args.preprocess or run_all:
            train_df, test_df, rul_df = run_preprocessing()

        if args.train or run_all:
            if "train_df" not in locals():
                train_df, test_df, rul_df = run_preprocessing()
            run_anomaly_detection(train_df)
            run_training(train_df, test_df, rul_df)

        if args.evaluate or run_all:
            if "test_df" not in locals():
                _, test_df, rul_df = run_preprocessing()
            run_evaluation(test_df, rul_df)

        if args.recommend or run_all:
            run_maintenance_recommendations()

        logger.success("Pipeline execution completed successfully! ✅")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
