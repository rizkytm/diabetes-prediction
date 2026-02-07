"""
Diabetes Prediction MLOps Package

This package contains modules for diabetes prediction including:
- processing.py: Data preprocessing pipeline
- training.py: Model training with MLflow tracking
- schemas.py: API request/response schemas for FastAPI
"""

__version__ = "1.0.0"
__author__ = "Claude Code"

from src.processing import (
    ZeroToNanTransformer,
    create_preprocessing_pipeline,
    load_preprocessing_pipeline,
    save_preprocessing_pipeline,
)
from src.training import (
    load_data,
    run_training_pipeline,
    setup_mlflow,
    split_data,
    train_logistic_regression,
    train_random_forest,
)

__all__ = [
    "ZeroToNanTransformer",
    "create_preprocessing_pipeline",
    "save_preprocessing_pipeline",
    "load_preprocessing_pipeline",
    "setup_mlflow",
    "load_data",
    "split_data",
    "train_logistic_regression",
    "train_random_forest",
    "run_training_pipeline",
]

# Export schemas for API usage
from src.schemas import PredictionResponse  # noqa: F401
from src.schemas import (  # noqa: F401
    BatchPredictionInput,
    BatchPredictionResponse,
    DiabetesInput,
    HealthResponse,
    ModelInfoResponse,
)

__all__.extend(
    [
        "DiabetesInput",
        "PredictionResponse",
        "HealthResponse",
        "ModelInfoResponse",
        "BatchPredictionInput",
        "BatchPredictionResponse",
    ]
)
