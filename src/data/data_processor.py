"""
Data Processing Module for Fraud Detection
==========================================

This module handles data ingestion, cleaning, and preprocessing for fraud detection models.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDataProcessor:
    """
    A comprehensive data processor for fraud detection datasets.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and outliers.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("Starting data cleaning...")
        
        # Handle missing values
        df = df.dropna()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        logger.info(f"Data cleaned. Final shape: {df.shape}")
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        logger.info("Starting feature engineering...")
        
        # Add time-based features if timestamp exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Add transaction amount ratios if applicable
        if 'amount' in df.columns:
            df['log_amount'] = np.log1p(df['amount'])
            df['amount_z_score'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        
        logger.info("Feature engineering completed")
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'is_fraud') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for model training.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target column
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and target arrays
        """
        logger.info("Preparing features for model training...")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = self.label_encoder.fit_transform(X[col])
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Features prepared. Shape: {X_scaled.shape}")
        return X_scaled, y.values
