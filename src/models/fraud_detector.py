"""
Fraud Detection Model
====================

This module contains the main fraud detection model implementation.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetector:
    """
    Main fraud detection model class supporting multiple algorithms.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize fraud detector with specified model type.
        
        Args:
            model_type (str): Type of model ('random_forest', 'logistic_regression', 'isolation_forest')
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        elif model_type == 'isolation_forest':
            self.model = IsolationForest(
                contamination=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the fraud detection model.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target vector
            test_size (float): Proportion of data to use for testing
            
        Returns:
            Dict[str, Any]: Training metrics and results
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        if self.model_type == 'isolation_forest':
            # Isolation Forest is unsupervised, train only on non-fraud data
            X_train_normal = X_train[y_train == 0]
            self.model.fit(X_train_normal)
        else:
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Evaluate model
        results = self.evaluate(X_test, y_test)
        logger.info("Model training completed")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model_type == 'isolation_forest':
            # Convert outlier labels (-1, 1) to (1, 0) for fraud detection
            predictions = self.model.predict(X)
            return (predictions == -1).astype(int)
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model_type == 'isolation_forest':
            # Use decision function for anomaly scores
            scores = self.model.decision_function(X)
            # Normalize to probabilities (approximate)
            probabilities = (scores - scores.min()) / (scores.max() - scores.min())
            return np.column_stack([1 - probabilities, probabilities])
        else:
            return self.model.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        predictions = self.predict(X_test)
        
        # Calculate metrics
        report = classification_report(y_test, predictions, output_dict=True)
        
        results = {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist()
        }
        
        # Add AUC score if model supports probability predictions
        if self.model_type != 'isolation_forest':
            probabilities = self.predict_proba(X_test)[:, 1]
            results['auc_score'] = roc_auc_score(y_test, probabilities)
        
        return results
    
    def save_model(self, file_path: str):
        """
        Save the trained model to disk.
        
        Args:
            file_path (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }, file_path)
        
        logger.info(f"Model saved to {file_path}")
    
    def load_model(self, file_path: str):
        """
        Load a trained model from disk.
        
        Args:
            file_path (str): Path to load the model from
        """
        model_data = joblib.load(file_path)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {file_path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores (for tree-based models).
        
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            return dict(enumerate(self.model.feature_importances_))
        else:
            logger.warning(f"Feature importance not available for {self.model_type}")
            return {}
