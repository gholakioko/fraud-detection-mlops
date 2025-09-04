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
        
        # Handle credit card dataset specifics
        if 'Time' in df.columns:
            # Convert time from seconds to hours and extract time-based features
            df['Time_hours'] = df['Time'] / 3600
            df['Time_hour_of_day'] = (df['Time'] / 3600) % 24
            df['Time_day'] = (df['Time'] / 3600 / 24).astype(int)
            
            # Create cyclical features for hour of day
            df['hour_sin'] = np.sin(2 * np.pi * df['Time_hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['Time_hour_of_day'] / 24)
        
        # Add transaction amount features if applicable
        if 'Amount' in df.columns:
            # Log transform for amount (add 1 to handle 0 values)
            df['log_amount'] = np.log1p(df['Amount'])
            
            # Amount categories
            df['amount_cat'] = pd.cut(df['Amount'], 
                                    bins=[0, 10, 50, 200, 1000, float('inf')],
                                    labels=['very_small', 'small', 'medium', 'large', 'very_large'])
            
            # Amount statistics
            df['amount_z_score'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
            df['is_small_amount'] = (df['Amount'] <= 10).astype(int)
            df['is_large_amount'] = (df['Amount'] >= 1000).astype(int)
        
        # Interaction features with V1-V28 (most important PCA components)
        if all(f'V{i}' in df.columns for i in range(1, 29)):
            # Sum of absolute values of V components
            v_cols = [f'V{i}' for i in range(1, 29)]
            df['V_sum_abs'] = df[v_cols].abs().sum(axis=1)
            df['V_mean_abs'] = df[v_cols].abs().mean(axis=1)
            df['V_std'] = df[v_cols].std(axis=1)
            
            # Create interaction features with amount and key V components
            if 'Amount' in df.columns:
                for v_col in ['V1', 'V2', 'V3', 'V4']:  # Most predictive V components typically
                    df[f'{v_col}_amount_ratio'] = df[v_col] / (df['Amount'] + 1)
        
        logger.info("Feature engineering completed")
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'Class') -> Tuple[np.ndarray, np.ndarray]:
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
        
        # Encode categorical variables (including category dtype from pd.cut)
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            if X[col].dtype == 'category':
                # For categorical columns from pd.cut, convert to string first
                X[col] = X[col].astype(str)
            X[col] = self.label_encoder.fit_transform(X[col])
        
        # Ensure all columns are numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # Check for any NaN values after conversion
        if X.isnull().any().any():
            logger.warning("Found NaN values after numeric conversion, filling with 0")
            X = X.fillna(0)
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Features prepared. Shape: {X_scaled.shape}")
        return X_scaled, y.values
