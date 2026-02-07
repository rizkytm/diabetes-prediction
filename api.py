# -*- coding: utf-8 -*-
"""
FastAPI REST API for Diabetes Prediction

This module provides REST API endpoints for diabetes prediction
with automatic request/response validation and OpenAPI documentation.

Author: Claude Code
Date: 2026-02-07
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import os
from typing import List
import logging

# Import schemas
from src.schemas import (
    DiabetesInput,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    BatchPredictionInput,
    BatchPredictionResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="REST API for predicting diabetes risk using machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
pipeline = None
model = None
model_type = None


# ==================== STARTUP & SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    global pipeline, model, model_type

    try:
        logger.info("[INFO] Loading models...")

        # Load preprocessing pipeline
        pipeline_path = 'models/preprocessing_pipeline.pkl'
        if not os.path.exists(pipeline_path):
            logger.error(f"[ERROR] Preprocessing pipeline not found: {pipeline_path}")
            raise FileNotFoundError(f"Preprocessing pipeline not found: {pipeline_path}")

        pipeline = joblib.load(pipeline_path)
        logger.info(f"[OK] Preprocessing pipeline loaded from: {pipeline_path}")

        # Load trained model
        model_path = 'models/best_model.pkl'
        if not os.path.exists(model_path):
            logger.error(f"[ERROR] Model not found: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)
        model_type = type(model).__name__
        logger.info(f"[OK] Model loaded from: {model_path}")
        logger.info(f"[INFO] Model type: {model_type}")

        logger.info("[OK] All models loaded successfully")

    except Exception as e:
        logger.error(f"[ERROR] Failed to load models: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("[INFO] Shutting down API...")


# ==================== HELPER FUNCTIONS ====================

def get_risk_level(probability: float) -> str:
    """
    Determine risk level based on probability.

    Parameters
    ----------
    probability : float
        Probability of having diabetes

    Returns
    -------
    risk_level : str
        Risk category
    """
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.5:
        return "Moderate Risk"
    elif probability < 0.7:
        return "High Risk"
    else:
        return "Very High Risk"


def get_confidence(probability: float) -> str:
    """
    Determine confidence level based on probability.

    Parameters
    ----------
    probability : float
        Probability of having diabetes

    Returns
    -------
    confidence : str
        Confidence level
    """
    if probability < 0.3 or probability > 0.7:
        return "High"
    elif probability < 0.4 or probability > 0.6:
        return "Medium"
    else:
        return "Low"


def get_recommendations(input_data: dict) -> List[str]:
    """
    Generate health recommendations based on input parameters.

    Parameters
    ----------
    input_data : dict
        Dictionary of feature values

    Returns
    -------
    recommendations : list
        List of health recommendations
    """
    recommendations = []

    # Glucose-based recommendations
    glucose = input_data.get('Glucose', 0)
    if glucose > 140:
        recommendations.append("[!] Glucose level elevated - Consider reducing sugar intake")
    elif glucose > 125:
        recommendations.append("[!] Glucose level above normal - Monitor blood sugar regularly")

    # BMI-based recommendations
    bmi = input_data.get('BMI', 0)
    if bmi >= 30:
        recommendations.append("[!] BMI indicates obesity - Weight management recommended")
    elif bmi >= 25:
        recommendations.append("[!] BMI indicates overweight - Consider healthy weight loss")

    # Blood pressure recommendations
    blood_pressure = input_data.get('BloodPressure', 0)
    if blood_pressure > 90:
        recommendations.append("[!] Blood pressure elevated - Regular monitoring advised")

    # Age-based recommendations
    age = input_data.get('Age', 0)
    if age > 45:
        recommendations.append("[!] Age over 45 - Regular diabetes screening recommended")

    # General recommendations
    if len(recommendations) == 0:
        recommendations.append("[OK] All parameters within normal range - Keep it up!")

    return recommendations


def predict_single(
    input_data: DiabetesInput,
    pipeline,
    model
) -> PredictionResponse:
    """
    Make prediction for a single input.

    Parameters
    ----------
    input_data : DiabetesInput
        Validated input data
    pipeline : sklearn.pipeline.Pipeline
        Preprocessing pipeline
    model : sklearn estimator
        Trained model

    Returns
    -------
    response : PredictionResponse
        Prediction response
    """
    # Convert to DataFrame with correct column names
    input_dict = {
        'Pregnancies': input_data.pregnancies,
        'Glucose': input_data.glucose,
        'BloodPressure': input_data.blood_pressure,
        'SkinThickness': input_data.skin_thickness,
        'Insulin': input_data.insulin,
        'BMI': input_data.bmi,
        'DiabetesPedigreeFunction': input_data.diabetes_pedigree_function,
        'Age': input_data.age
    }

    input_df = pd.DataFrame([input_dict])

    # Preprocess
    input_processed = pipeline.transform(input_df)

    # Predict
    prediction = int(model.predict(input_processed)[0])

    # Get probability
    if hasattr(model, 'predict_proba'):
        probability = float(model.predict_proba(input_processed)[0, 1])
    else:
        probability = float(prediction)

    # Generate response
    response = PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        risk_level=get_risk_level(probability),
        confidence=get_confidence(probability),
        model_used=model_type,
        recommendations=get_recommendations(input_dict)
    )

    # Log prediction
    logger.info(f"[PREDICTION] Risk: {response.risk_level}, Probability: {response.probability:.4f}")

    return response


# ==================== ENDPOINTS ====================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Diabetes Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info"
        }
    }


@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint.

    Returns API status and model loading information.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=(model is not None and pipeline is not None),
        api_version="1.0.0",
        mlflow_tracking=False
    )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(input_data: DiabetesInput):
    """
    Predict diabetes risk for a single patient.

    This endpoint takes patient medical parameters and returns
    diabetes prediction with probability and recommendations.
    """
    if model is None or pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please check server logs."
        )

    try:
        response = predict_single(input_data, pipeline, model)
        return response

    except Exception as e:
        logger.error(f"[ERROR] Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Predict diabetes risk for multiple patients.

    This endpoint takes a list of patient data and returns
    predictions for all patients.
    """
    if model is None or pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please check server logs."
        )

    try:
        predictions = []
        probabilities = []

        for patient_data in batch_input.patients:
            response = predict_single(patient_data, pipeline, model)
            predictions.append(response)
            probabilities.append(response.probability)

        # Calculate summary statistics
        high_risk_count = sum(1 for p in predictions if p.prediction == 1)
        avg_probability = np.mean(probabilities)

        summary = {
            "total_patients": len(predictions),
            "high_risk_count": high_risk_count,
            "low_risk_count": len(predictions) - high_risk_count,
            "average_probability": round(avg_probability, 4)
        }

        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )

    except Exception as e:
        logger.error(f"[ERROR] Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model-info", response_model=ModelInfoResponse, status_code=status.HTTP_200_OK)
async def get_model_info():
    """
    Get information about the loaded model.

    Returns model type, features, and metadata.
    """
    if model is None or pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please check server logs."
        )

    features = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    return ModelInfoResponse(
        model_type=model_type,
        model_version="1.0.0",
        feature_count=len(features),
        features=features,
        target_metric="Recall",
        training_date="2026-02-07"
    )


# ==================== ERROR HANDLERS ====================

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    """Handle file not found errors."""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Model files not found. Please train the model first."}
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)}
    )


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
