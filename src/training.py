# -*- coding: utf-8 -*-
"""
Training Module for Diabetes Prediction

This module contains functions for training machine learning models
with MLflow experiment tracking.

Author: Claude Code
Date: 2026-02-07
"""

import os
from typing import Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Import preprocessing pipeline
try:
    from src.processing import create_preprocessing_pipeline
except ImportError:
    # For running as standalone script
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.processing import create_preprocessing_pipeline


# ==================== MLFLOW SETUP ====================


def setup_mlflow(
    tracking_uri: str = "MLruns", experiment_name: str = "Diabetes_Prediction"
) -> None:
    """
    Configure MLflow tracking.

    Parameters
    ----------
    tracking_uri : str, default='MLruns'
        Path or URI for MLflow tracking
    experiment_name : str, default='Diabetes_Prediction'
        Name of the MLflow experiment
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"[OK] MLflow configured: {experiment_name}")
    print(f"   Tracking URI: {mlflow.get_tracking_uri()}")


# ==================== DATA LOADING ====================


def load_data(filepath: str = "data/diabetes.csv") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load diabetes dataset and separate features and target.

    Parameters
    ----------
    filepath : str, default='data/diabetes.csv'
        Path to the dataset

    Returns
    -------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    """
    df = pd.read_csv(filepath)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    print(f"[OK] Data loaded from: {filepath}")
    print(f"   Shape: {df.shape}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Class distribution: {y.value_counts().to_dict()}")

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple:
    """
    Split data into train and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    test_size : float, default=0.2
        Proportion of data for testing
    random_state : int, default=42
        Random seed for reproducibility
    stratify : bool, default=True
        Whether to stratify the split (maintain class ratio)

    Returns
    -------
    X_train, X_test, y_train, y_test : Tuple of DataFrames/Series
        Split data
    """
    split_params = {"test_size": test_size, "random_state": random_state}

    if stratify:
        split_params["stratify"] = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_params)

    print(f"[OK] Data split:")  # noqa: F541
    print(f"   Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_test, y_train, y_test


# ==================== MODEL EVALUATION ====================


def evaluate_model(
    model: object,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate model and return metrics.

    Parameters
    ----------
    model : sklearn estimator
        Trained model
    X_train : np.ndarray
        Preprocessed training features
    X_test : np.ndarray
        Preprocessed test features
    y_train : pd.Series
        Training target
    y_test : pd.Series
        Test target

    Returns
    -------
    metrics : dict
        Dictionary containing all metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_train_proba = y_train_pred
        y_test_proba = y_test_pred

    # Calculate metrics
    metrics = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
        "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
        "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
        "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
        "train_roc_auc": roc_auc_score(y_train, y_train_proba),
        "test_roc_auc": roc_auc_score(y_test, y_test_proba),
    }

    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """
    Print metrics in a formatted way.

    Parameters
    ----------
    metrics : dict
        Dictionary of metrics
    model_name : str, default="Model"
        Name of the model
    """
    print(f"\n{'='*70}")  # noqa: F541
    print(f"METRICS: {model_name}")
    print(f"{'='*70}")  # noqa: F541  # noqa: F541

    print(f"\nTRAINING METRICS:")  # noqa: F541
    print(f"   Accuracy:  {metrics['train_accuracy']:.4f}")
    print(f"   Precision: {metrics['train_precision']:.4f}")
    print(f"   Recall:    {metrics['train_recall']:.4f}")
    print(f"   F1-Score:  {metrics['train_f1']:.4f}")
    print(f"   ROC-AUC:   {metrics['train_roc_auc']:.4f}")

    print(f"\nTEST METRICS:")  # noqa: F541
    print(f"   Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"   Precision: {metrics['test_precision']:.4f}")
    print(f"   Recall:    {metrics['test_recall']:.4f}  *** MOST IMPORTANT")
    print(f"   F1-Score:  {metrics['test_f1']:.4f}")
    print(f"   ROC-AUC:   {metrics['test_roc_auc']:.4f}")

    # Check for overfitting
    acc_diff = metrics["train_accuracy"] - metrics["test_accuracy"]
    if acc_diff > 0.1:
        print(f"\n[!] OVERFITTING: Gap = {acc_diff:.4f}")
    elif acc_diff < -0.05:
        print(f"\n[!] UNDERFITTING: Test > Train")  # noqa: F541
    else:
        print(f"\n[OK] GOOD FIT: Gap = {acc_diff:.4f}")

    print(f"{'='*70}")  # noqa: F541


# ==================== MODEL TRAINING ====================


def train_logistic_regression(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    run_name: str = "Logistic_Regression",
    **model_params,
) -> Tuple[object, Dict[str, float]]:
    """
    Train Logistic Regression with MLflow tracking.

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed training features
    X_test : np.ndarray
        Preprocessed test features
    y_train : pd.Series
        Training target
    y_test : pd.Series
        Test target
    run_name : str, default="Logistic_Regression"
        Name for MLflow run
    **model_params
        Additional parameters for LogisticRegression

    Returns
    -------
    model : LogisticRegression
        Trained model
    metrics : dict
        Evaluation metrics
    """
    print(f"\n[TRAINING] {run_name}")
    print(f"{'='*70}")  # noqa: F541

    # Default parameters
    default_params = {
        "class_weight": "balanced",
        "random_state": 42,
        "solver": "liblinear",
        "max_iter": 1000,
    }
    params = {**default_params, **model_params}

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Create and train model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Print results
        print_metrics(metrics, run_name)

        run_id = mlflow.active_run().info.run_id
        print(f"\n[OK] MLflow Run ID: {run_id}")

    return model, metrics


def train_random_forest(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    run_name: str = "Random_Forest",
    **model_params,
) -> Tuple[object, Dict[str, float]]:
    """
    Train Random Forest with MLflow tracking.

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed training features
    X_test : np.ndarray
        Preprocessed test features
    y_train : pd.Series
        Training target
    y_test : pd.Series
        Test target
    run_name : str, default="Random_Forest"
        Name for MLflow run
    **model_params
        Additional parameters for RandomForestClassifier

    Returns
    -------
    model : RandomForestClassifier
        Trained model
    metrics : dict
        Evaluation metrics
    """
    print(f"\n[TRAINING] {run_name}")
    print(f"{'='*70}")  # noqa: F541

    # Default parameters
    default_params = {
        "n_estimators": 100,
        "class_weight": "balanced",
        "random_state": 42,
        "max_depth": 10,
        "min_samples_split": 5,
        "n_jobs": -1,
    }
    params = {**default_params, **model_params}

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Create and train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Print results
        print_metrics(metrics, run_name)

        # Feature importance
        feature_importance = pd.DataFrame(
            {
                "feature": [f"feature_{i}" for i in range(model.n_features_in_)],
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print(f"\n[INFO] Top 5 Important Features:")  # noqa: F541
        for idx, row in feature_importance.head(5).iterrows():
            bar = "#" * int(row["importance"] * 50)
            print(f"   {row['feature']:<15} {row['importance']:.4f}  {bar}")

        run_id = mlflow.active_run().info.run_id
        print(f"\n[OK] MLflow Run ID: {run_id}")

    return model, metrics


# ==================== SAVE & LOAD MODELS ====================


def save_model(model: object, filepath: str, model_name: str = "model") -> None:
    """
    Save trained model to disk.

    Parameters
    ----------
    model : sklearn estimator
        Trained model
    filepath : str
        Path to save the model
    model_name : str, default="model"
        Name of the model for logging
    """
    joblib.dump(model, filepath)
    print(f"[OK] {model_name} saved to: {filepath}")


def load_model(filepath: str) -> object:
    """
    Load trained model from disk.

    Parameters
    ----------
    filepath : str
        Path to the saved model

    Returns
    -------
    model : sklearn estimator
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"[OK] Model loaded from: {filepath}")
    return model


# ==================== MAIN TRAINING PIPELINE ====================


def run_training_pipeline(
    data_path: str = "data/diabetes.csv",
    models_to_train: list = ["logistic_regression", "random_forest"],
    test_size: float = 0.2,
    random_state: int = 42,
    save_models: bool = True,
    models_dir: str = "models",
) -> Dict[str, Tuple[object, Dict[str, float]]]:
    """
    Complete training pipeline: load data, preprocess, train models, evaluate.

    Parameters
    ----------
    data_path : str, default='data/diabetes.csv'
        Path to dataset
    models_to_train : list, default=['logistic_regression', 'random_forest']
        List of models to train
    test_size : float, default=0.2
        Proportion of data for testing
    random_state : int, default=42
        Random seed
    save_models : bool, default=True
        Whether to save trained models
    models_dir : str, default='models'
        Directory to save models

    Returns
    -------
    results : dict
        Dictionary mapping model names to (model, metrics) tuples
    """
    print("=" * 70)
    print("DIABETES PREDICTION - TRAINING PIPELINE")
    print("=" * 70)

    # Setup MLflow
    setup_mlflow()

    # Load data
    X, y = load_data(data_path)

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state, stratify=True
    )

    # Create preprocessing pipeline
    print(f"\n[INFO] Creating preprocessing pipeline...")  # noqa: F541
    preprocessing_pipeline = create_preprocessing_pipeline()
    preprocessing_pipeline.fit(X_train)

    # Save preprocessing pipeline
    if save_models:
        os.makedirs(models_dir, exist_ok=True)
        prep_path = os.path.join(models_dir, "preprocessing_pipeline.pkl")
        save_model(preprocessing_pipeline, prep_path, "Preprocessing pipeline")

    # Apply preprocessing
    X_train_processed = preprocessing_pipeline.transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    print(f"\n[OK] Preprocessing complete:")  # noqa: F541
    print(f"   Train shape: {X_train_processed.shape}")
    print(f"   Test shape:  {X_test_processed.shape}")

    # Train models
    results = {}

    if "logistic_regression" in models_to_train:
        lr_model, lr_metrics = train_logistic_regression(
            X_train_processed, X_test_processed, y_train, y_test
        )
        results["logistic_regression"] = (lr_model, lr_metrics)

        if save_models:
            lr_path = os.path.join(models_dir, "logistic_regression_model.pkl")
            save_model(lr_model, lr_path, "Logistic Regression")

    if "random_forest" in models_to_train:
        rf_model, rf_metrics = train_random_forest(
            X_train_processed, X_test_processed, y_train, y_test
        )
        results["random_forest"] = (rf_model, rf_metrics)

        if save_models:
            rf_path = os.path.join(models_dir, "random_forest_model.pkl")
            save_model(rf_model, rf_path, "Random Forest")

    # Select best model (based on recall)
    print(f"\n{'='*70}")
    print("MODEL SELECTION")
    print(f"{'='*70}")  # noqa: F541

    best_model_name = max(results.keys(), key=lambda k: results[k][1]["test_recall"])
    best_model, best_metrics = results[best_model_name]

    print(f"\n[OK] Best Model: {best_model_name}")
    print(f"   Test Recall: {best_metrics['test_recall']:.4f}")
    print(f"   Test F1-Score: {best_metrics['test_f1']:.4f}")
    print(f"   Test ROC-AUC: {best_metrics['test_roc_auc']:.4f}")

    # Save best model
    if save_models:
        best_path = os.path.join(models_dir, "best_model.pkl")
        save_model(best_model, best_path, "Best Model")

    print(f"\n{'='*70}")
    print("TRAINING PIPELINE COMPLETE")
    print(f"{'='*70}")  # noqa: F541

    print(f"\n[INFO] View MLflow Dashboard:")  # noqa: F541
    print(f"   mlflow ui")  # noqa: F541
    print(f"   Open: http://localhost:5000")  # noqa: F541

    return results


# ==================== MAIN ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train diabetes prediction models")
    parser.add_argument(
        "--data", type=str, default="data/diabetes.csv", help="Path to dataset"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "random_forest"],
        choices=["logistic_regression", "random_forest"],
        help="Models to train",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion (default: 0.2)",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save models")

    args = parser.parse_args()

    # Run training pipeline
    run_training_pipeline(
        data_path=args.data,
        models_to_train=args.models,
        test_size=args.test_size,
        save_models=not args.no_save,
    )
