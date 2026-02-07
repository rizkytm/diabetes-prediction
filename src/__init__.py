"""
Diabetes Prediction MLOps Package

This package contains modules for diabetes prediction including:
- processing.py: Data preprocessing pipeline
- training.py: Model training with MLflow tracking
"""

__version__ = "1.0.0"
__author__ = "Claude Code"

from src.processing import (
    ZeroToNanTransformer,
    create_preprocessing_pipeline,
    save_preprocessing_pipeline,
    load_preprocessing_pipeline
)

from src.training import (
    setup_mlflow,
    load_data,
    split_data,
    train_logistic_regression,
    train_random_forest,
    run_training_pipeline
)

__all__ = [
    'ZeroToNanTransformer',
    'create_preprocessing_pipeline',
    'save_preprocessing_pipeline',
    'load_preprocessing_pipeline',
    'setup_mlflow',
    'load_data',
    'split_data',
    'train_logistic_regression',
    'train_random_forest',
    'run_training_pipeline'
]
