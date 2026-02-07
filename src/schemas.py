# -*- coding: utf-8 -*-
"""
API Schemas for Diabetes Prediction

This module contains Pydantic models for request/response validation
in the FastAPI application.

Author: Claude Code
Date: 2026-02-07
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
import numpy as np


class DiabetesInput(BaseModel):
    """
    Input schema for diabetes prediction.

    All fields are validated based on medical knowledge and dataset constraints.
    """

    pregnancies: int = Field(
        ...,
        ge=0,
        le=20,
        description="Number of times pregnant",
        example=1
    )

    glucose: float = Field(
        ...,
        ge=0,
        le=300,
        description="Plasma glucose concentration after 2 hours in OGTT (mg/dL)",
        example=120
    )

    blood_pressure: float = Field(
        ...,
        ge=0,
        le=200,
        description="Diastolic blood pressure (mm Hg)",
        example=70
    )

    skin_thickness: float = Field(
        ...,
        ge=0,
        le=100,
        description="Triceps skin fold thickness (mm)",
        example=20
    )

    insulin: float = Field(
        ...,
        ge=0,
        le=1000,
        description="2-Hour serum insulin (mu U/ml)",
        example=80
    )

    bmi: float = Field(
        ...,
        ge=0.0,
        le=70.0,
        description="Body mass index (weight in kg / height in m^2)",
        example=32.0
    )

    diabetes_pedigree_function: float = Field(
        ...,
        ge=0.0,
        le=3.0,
        description="Diabetes genetic risk score",
        example=0.5
    )

    age: int = Field(
        ...,
        ge=21,
        le=120,
        description="Age in years",
        example=33
    )

    class Config:
        json_schema_extra = {
            "example": {
                "pregnancies": 1,
                "glucose": 120,
                "blood_pressure": 70,
                "skin_thickness": 20,
                "insulin": 80,
                "bmi": 32.0,
                "diabetes_pedigree_function": 0.5,
                "age": 33
            }
        }

    @validator('glucose')
    def validate_glucose(cls, v):
        """Warn if glucose is unusually high or low."""
        if v > 0 and v < 50:
            raise ValueError("Glucose level too low (possible error)")
        if v > 300:
            raise ValueError("Glucose level too high (possible error)")
        return v

    @validator('bmi')
    def validate_bmi(cls, v):
        """Warn if BMI is outside reasonable range."""
        if v > 0 and v < 10:
            raise ValueError("BMI too low (possible error)")
        if v > 70:
            raise ValueError("BMI too high (possible error)")
        return v


class PredictionResponse(BaseModel):
    """
    Response schema for diabetes prediction.
    """

    prediction: int = Field(
        ...,
        ge=0,
        le=1,
        description="Predicted class (0=No Diabetes, 1=Diabetes)"
    )

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of having diabetes"
    )

    risk_level: str = Field(
        ...,
        description="Risk category: Low, Moderate, High, or Very High"
    )

    confidence: str = Field(
        ...,
        description="Confidence level based on probability"
    )

    model_used: str = Field(
        ...,
        description="Name of the model used for prediction"
    )

    recommendations: List[str] = Field(
        ...,
        description="Health recommendations based on input parameters"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 0,
                "probability": 0.23,
                "risk_level": "Low Risk",
                "confidence": "High",
                "model_used": "LogisticRegression",
                "recommendations": [
                    "All parameters within normal range - Keep it up!"
                ]
            }
        }


class HealthResponse(BaseModel):
    """
    Response schema for health check endpoint.
    """

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    api_version: str = Field(..., description="API version")
    mlflow_tracking: bool = Field(..., description="MLflow tracking status")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "api_version": "1.0.0",
                "mlflow_tracking": False
            }
        }


class ModelInfoResponse(BaseModel):
    """
    Response schema for model information endpoint.
    """

    model_type: str = Field(..., description="Type of model")
    model_version: str = Field(..., description="Model version")
    feature_count: int = Field(..., description="Number of features")
    features: List[str] = Field(..., description="List of feature names")
    target_metric: str = Field(..., description="Metric optimized for")
    training_date: Optional[str] = Field(None, description="When model was trained")

    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "LogisticRegression",
                "model_version": "1.0.0",
                "feature_count": 8,
                "features": [
                    "Pregnancies",
                    "Glucose",
                    "BloodPressure",
                    "SkinThickness",
                    "Insulin",
                    "BMI",
                    "DiabetesPedigreeFunction",
                    "Age"
                ],
                "target_metric": "Recall",
                "training_date": "2026-02-07"
            }
        }


class BatchPredictionInput(BaseModel):
    """
    Input schema for batch prediction.
    """

    patients: List[DiabetesInput] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of patient data for batch prediction"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "patients": [
                    {
                        "pregnancies": 1,
                        "glucose": 120,
                        "blood_pressure": 70,
                        "skin_thickness": 20,
                        "insulin": 80,
                        "bmi": 32.0,
                        "diabetes_pedigree_function": 0.5,
                        "age": 33
                    },
                    {
                        "pregnancies": 0,
                        "glucose": 140,
                        "blood_pressure": 80,
                        "skin_thickness": 25,
                        "insulin": 100,
                        "bmi": 35.0,
                        "diabetes_pedigree_function": 0.6,
                        "age": 45
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Response schema for batch prediction.
    """

    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions for each patient"
    )

    summary: dict = Field(
        ...,
        description="Summary statistics of batch predictions"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [],
                "summary": {
                    "total_patients": 2,
                    "high_risk_count": 1,
                    "average_probability": 0.45
                }
            }
        }
