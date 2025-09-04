"""
Utility Functions for Fraud Detection MLOps
==========================================

Common helper functions used across the project.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level (str): Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fraud_detection.log')
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def save_metrics(metrics: Dict[str, Any], output_path: str):
    """
    Save model metrics to JSON file.
    
    Args:
        metrics (Dict[str, Any]): Metrics dictionary
        output_path (str): Path to save metrics
    """
    # Add timestamp
    metrics['timestamp'] = datetime.now().isoformat()
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def generate_sample_data(n_samples: int = 1000, fraud_rate: float = 0.1) -> pd.DataFrame:
    """
    Generate sample fraud detection dataset for testing.
    
    Args:
        n_samples (int): Number of samples to generate
        fraud_rate (float): Proportion of fraudulent transactions
        
    Returns:
        pd.DataFrame: Sample dataset
    """
    np.random.seed(42)
    
    # Generate features
    data = {
        'amount': np.random.lognormal(3, 1, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'customer_age': np.random.randint(18, 80, n_samples),
        'merchant_category': np.random.choice(
            ['grocery', 'gas', 'restaurant', 'online', 'retail'], 
            n_samples
        ),
        'transaction_type': np.random.choice(
            ['purchase', 'withdrawal', 'transfer'], 
            n_samples
        )
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add derived features
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['log_amount'] = np.log1p(df['amount'])
    
    # Generate fraud labels (simplified logic)
    fraud_probability = (
        (df['amount'] > df['amount'].quantile(0.95)).astype(float) * 0.3 +
        (df['hour'].isin([2, 3, 4])).astype(float) * 0.2 +
        (df['is_weekend']).astype(float) * 0.1 +
        np.random.random(n_samples) * 0.4
    )
    
    # Adjust to desired fraud rate
    fraud_threshold = np.quantile(fraud_probability, 1 - fraud_rate)
    df['is_fraud'] = (fraud_probability >= fraud_threshold).astype(int)
    
    return df


def validate_data_schema(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame has required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (List[str]): List of required column names
        
    Returns:
        bool: True if all required columns are present
    """
    missing_columns = set(required_columns) - set(df.columns)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True


def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate comprehensive model evaluation metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_prob (np.ndarray, optional): Prediction probabilities
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics['auc_score'] = roc_auc_score(y_true, y_prob)
    
    # Confusion matrix as nested dict for JSON serialization
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = {
        'true_negative': int(cm[0, 0]),
        'false_positive': int(cm[0, 1]),
        'false_negative': int(cm[1, 0]),
        'true_positive': int(cm[1, 1])
    }
    
    return metrics


def create_directory_structure(base_path: str):
    """
    Create the standard MLOps directory structure.
    
    Args:
        base_path (str): Base directory path
    """
    directories = [
        'data/raw',
        'data/processed',
        'data/external',
        'models/trained',
        'models/artifacts',
        'notebooks',
        'src/data',
        'src/models',
        'src/training',
        'src/serving',
        'src/utils',
        'infra/k8s',
        'infra/docker',
        'infra/ci-cd',
        'charts'
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
        
        # Create .gitkeep for empty directories
        gitkeep_path = os.path.join(dir_path, '.gitkeep')
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, 'w') as f:
                f.write('')


if __name__ == "__main__":
    # Generate sample data for testing
    sample_df = generate_sample_data(1000, 0.1)
    sample_df.to_csv('data/raw/sample_fraud_data.csv', index=False)
    print("Sample data generated and saved to data/raw/sample_fraud_data.csv")
