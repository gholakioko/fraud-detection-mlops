"""
Model Training Pipeline
======================

This script handles the complete model training pipeline for fraud detection.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.data_processor import FraudDataProcessor
from models.fraud_detector import FraudDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_fraud_model(
    data_path: str,
    model_type: str = 'random_forest',
    output_dir: str = 'models/trained',
    test_size: float = 0.2
):
    """
    Complete training pipeline for fraud detection model.
    
    Args:
        data_path (str): Path to training data CSV
        model_type (str): Type of model to train
        output_dir (str): Directory to save trained model
        test_size (float): Proportion of data for testing
    """
    logger.info("Starting fraud detection model training pipeline...")
    
    # Initialize components
    data_processor = FraudDataProcessor()
    fraud_detector = FraudDetector(model_type=model_type)
    
    # Load and process data
    logger.info(f"Loading data from {data_path}")
    df = data_processor.load_data(data_path)
    
    # Clean data
    df_clean = data_processor.clean_data(df)
    
    # Feature engineering
    df_processed = data_processor.feature_engineering(df_clean)
    
    # Prepare features
    X, y = data_processor.prepare_features(df_processed)
    
    # Train model
    results = fraud_detector.train(X, y, test_size=test_size)
    
    # Save model and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"fraud_detector_{model_type}_{timestamp}.pkl"
    results_filename = f"training_results_{model_type}_{timestamp}.json"
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, model_filename)
    results_path = os.path.join(output_dir, results_filename)
    
    # Save model
    fraud_detector.save_model(model_path)
    
    # Save training results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log results
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Model Performance:")
    logger.info(f"  - Accuracy: {results['accuracy']:.4f}")
    logger.info(f"  - Precision: {results['precision']:.4f}")
    logger.info(f"  - Recall: {results['recall']:.4f}")
    logger.info(f"  - F1-Score: {results['f1_score']:.4f}")
    
    if 'auc_score' in results:
        logger.info(f"  - AUC Score: {results['auc_score']:.4f}")
    
    return model_path, results_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to training data CSV file"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "logistic_regression", "isolation_forest"],
        help="Type of model to train"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/trained",
        help="Directory to save trained model"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing"
    )
    
    args = parser.parse_args()
    
    train_fraud_model(
        data_path=args.data_path,
        model_type=args.model_type,
        output_dir=args.output_dir,
        test_size=args.test_size
    )
