"""
FastAPI Fraud Detection Service
==============================

REST API service for fraud detection predictions.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.fraud_detector import FraudDetector
from data.data_processor import FraudDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="ML-powered fraud detection service",
    version="1.0.0"
)

# Global variables for model and processor
fraud_detector = None
data_processor = None


class TransactionData(BaseModel):
    """Transaction data model for predictions."""
    Time: float = 0.0
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    Amount: float = 0.0
    

class PredictionResponse(BaseModel):
    """Response model for fraud predictions."""
    is_fraud: int
    fraud_probability: float
    confidence: str
    

class BatchTransactionData(BaseModel):
    """Batch transaction data for multiple predictions."""
    transactions: List[TransactionData]


@app.on_event("startup")
async def startup_event():
    """Load model and data processor on startup."""
    global fraud_detector, data_processor
    
    logger.info("Starting Fraud Detection API...")
    
    # Load trained model (you can set this via environment variable)
    model_path = os.getenv("MODEL_PATH", "models/trained/fraud_detector_random_forest_latest.pkl")
    processor_path = os.getenv("PROCESSOR_PATH", model_path.replace("fraud_detector", "data_processor"))
    
    # Initialize fraud detector
    fraud_detector = FraudDetector()
    
    # Load model
    if os.path.exists(model_path):
        try:
            fraud_detector.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
    else:
        logger.warning(f"Model file not found at {model_path}")
    
    # Load data processor
    if os.path.exists(processor_path):
        try:
            import pickle
            with open(processor_path, 'rb') as f:
                data_processor = pickle.load(f)
            logger.info(f"Data processor loaded successfully from {processor_path}")
        except Exception as e:
            logger.error(f"Failed to load data processor: {str(e)}")
            # Fallback to new processor
            data_processor = FraudDataProcessor()
    else:
        logger.warning(f"Data processor file not found at {processor_path}. Using new processor.")
        data_processor = FraudDataProcessor()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Fraud Detection API is running", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Detailed health check."""
    global fraud_detector
    
    health_status = {
        "api_status": "healthy",
        "model_loaded": fraud_detector is not None and fraud_detector.is_trained,
        "model_type": fraud_detector.model_type if fraud_detector else None
    }
    
    return health_status


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionData):
    """
    Predict fraud for a single transaction.
    
    Args:
        transaction: Transaction data
        
    Returns:
        Fraud prediction with probability
    """
    global fraud_detector, data_processor
    
    if not fraud_detector or not fraud_detector.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded or trained")
    
    try:
        # Convert transaction to dataframe
        transaction_dict = transaction.dict()
        df = pd.DataFrame([transaction_dict])
        
        # Apply the same feature engineering as during training
        df_processed = data_processor.feature_engineering(df)
        
        # Handle categorical features if they exist (from pd.cut in feature engineering)
        categorical_columns = df_processed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            if df_processed[col].dtype == 'category':
                df_processed[col] = df_processed[col].astype(str)
            # Use the same encoder as during training (simple approach)
            df_processed[col] = pd.Categorical(df_processed[col], 
                                             categories=['very_small', 'small', 'medium', 'large', 'very_large']).codes
        
        # Convert to numeric and handle any NaNs
        df_processed = df_processed.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Use the saved scaler from the data processor
        X_scaled = data_processor.scaler.transform(df_processed)
        
        # Make prediction
        prediction = fraud_detector.predict(X_scaled)[0]
        
        # Get probability if available
        if hasattr(fraud_detector.model, 'predict_proba'):
            probabilities = fraud_detector.predict_proba(X_scaled)[0]
            fraud_prob = float(probabilities[1])
        else:
            fraud_prob = float(prediction)
        
        # Determine confidence level
        if fraud_prob > 0.8:
            confidence = "high"
        elif fraud_prob > 0.5:
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictionResponse(
            is_fraud=int(prediction),
            fraud_probability=fraud_prob,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_fraud_batch(batch_data: BatchTransactionData):
    """
    Predict fraud for multiple transactions.
    
    Args:
        batch_data: Batch of transaction data
        
    Returns:
        List of fraud predictions
    """
    global fraud_detector, data_processor
    
    if not fraud_detector or not fraud_detector.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded or trained")
    
    try:
        predictions = []
        
        for transaction in batch_data.transactions:
            # Reuse single prediction logic
            result = await predict_fraud(transaction)
            predictions.append(result.dict())
        
        return {"predictions": predictions, "count": len(predictions)}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    global fraud_detector
    
    if not fraud_detector:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_type": fraud_detector.model_type,
        "is_trained": fraud_detector.is_trained,
        "supports_probability": hasattr(fraud_detector.model, 'predict_proba')
    }
    
    # Add feature importance if available
    try:
        feature_importance = fraud_detector.get_feature_importance()
        if feature_importance:
            info["feature_importance"] = feature_importance
    except Exception as e:
        logger.warning(f"Could not get feature importance: {str(e)}")
    
    return info


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
