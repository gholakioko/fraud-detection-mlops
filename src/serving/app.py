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
    amount: float
    merchant_category: str = "online"
    hour: int = 12
    day_of_week: int = 1
    is_weekend: int = 0
    customer_age: int = 35
    transaction_type: str = "purchase"
    

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
    
    # Initialize components
    fraud_detector = FraudDetector()
    data_processor = FraudDataProcessor()
    
    # Load trained model (you can set this via environment variable)
    model_path = os.getenv("MODEL_PATH", "models/trained/fraud_detector_random_forest_latest.pkl")
    
    if os.path.exists(model_path):
        try:
            fraud_detector.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            # Continue without model for health checks
    else:
        logger.warning(f"Model file not found at {model_path}")


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
        
        # Process data (basic preprocessing for serving)
        # Note: In production, you'd want more sophisticated preprocessing
        categorical_cols = ['merchant_category', 'transaction_type']
        for col in categorical_cols:
            if col in df.columns:
                # Simple label encoding for demo (in production, use fitted encoders)
                unique_values = df[col].unique()
                df[col] = pd.Categorical(df[col]).codes
        
        # Make prediction
        X = df.values
        prediction = fraud_detector.predict(X)[0]
        
        # Get probability if available
        if hasattr(fraud_detector.model, 'predict_proba'):
            probabilities = fraud_detector.predict_proba(X)[0]
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
