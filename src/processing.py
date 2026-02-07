"""
Preprocessing Pipeline Module for Diabetes Prediction

This module contains custom transformers and preprocessing pipeline
for handling missing values and scaling features.

Author: Claude Code
Date: 2026-02-07
"""

from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class ZeroToNanTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to convert invalid zeros to NaN in medical columns.

    Certain medical columns cannot have zero values in living patients.
    This transformer identifies and converts those zeros to NaN for subsequent
    imputation.

    Medical columns that shouldn't have zeros:
    - Glucose: Plasma glucose concentration (0 would mean coma/death)
    - BloodPressure: Diastolic blood pressure (0 means no heartbeat)
    - SkinThickness: Triceps skin fold thickness
    - Insulin: 2-Hour serum insulin
    - BMI: Body mass index (0 means no body mass)

    NOTE: Pregnancies=0 is VALID (means no pregnancies)

    Parameters
    ----------
    medical_columns : list, default=None
        List of column names where zeros should be converted to NaN.
        If None, uses default medical columns.

    Attributes
    ----------
    medical_columns : list
        List of medical column names to transform.

    Examples
    --------
    >>> transformer = ZeroToNanTransformer()
    >>> X_transformed = transformer.fit_transform(X)
    >>> # Now zeros in medical columns are NaN
    """

    def __init__(self, medical_columns: Optional[List[str]] = None):
        """
        Initialize the ZeroToNanTransformer.

        Parameters
        ----------
        medical_columns : list, default=None
            List of column names where zeros should be converted to NaN.
            If None, uses default medical columns.
        """
        if medical_columns is None:
            self.medical_columns = [
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
            ]
        else:
            self.medical_columns = medical_columns

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ):
        """
        Fit method (required for sklearn pipeline compatibility).

        This transformer doesn't need to learn anything from the data,
        but fit method is required for sklearn compatibility.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input features
        y : pd.DataFrame, np.ndarray, or None, default=None
            Target variable (not used)

        Returns
        -------
        self : ZeroToNanTransformer
            Returns self for method chaining
        """
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Transform zeros to NaN in medical columns.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input features

        Returns
        -------
        X_transformed : pd.DataFrame
            Transformed features with zeros converted to NaN in medical columns

        Raises
        ------
        ValueError
            If medical columns are not found in the input data
        """
        # Make a copy to avoid modifying original data
        X_transformed = X.copy()

        # Convert to DataFrame if numpy array
        if not isinstance(X_transformed, pd.DataFrame):
            if not hasattr(self, "feature_names_in_"):
                raise ValueError(
                    "Cannot transform numpy array without feature names. "
                    "Either provide a DataFrame or fit on a DataFrame first."
                )
            X_transformed = pd.DataFrame(X_transformed, columns=self.feature_names_in_)

        # Validate that medical columns exist
        missing_cols = set(self.medical_columns) - set(X_transformed.columns)
        if missing_cols:
            raise ValueError(
                f"Medical columns not found in data: {missing_cols}. "
                f"Available columns: {X_transformed.columns.tolist()}"
            )

        # Store feature names for future transformations
        self.feature_names_in_ = X_transformed.columns.tolist()

        # Replace zeros with NaN in medical columns
        for col in self.medical_columns:
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].replace(0, np.nan)

        return X_transformed

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input features
        y : pd.DataFrame, np.ndarray, or None, default=None
            Target variable (not used)

        Returns
        -------
        X_transformed : pd.DataFrame
            Transformed features with zeros converted to NaN
        """
        return self.fit(X, y).transform(X)


def create_preprocessing_pipeline(
    imputer_strategy: str = "median", scaler: str = "standard"
) -> Pipeline:
    """
    Create a complete preprocessing pipeline for diabetes prediction.

    The pipeline performs:
    1. Convert invalid zeros to NaN (ZeroToNanTransformer)
    2. Impute missing values (SimpleImputer)
    3. Scale features (StandardScaler)

    Parameters
    ----------
    imputer_strategy : str, default='median'
        The imputation strategy:
        - 'median': Replace with median (robust to outliers)
        - 'mean': Replace with mean
        - 'most_frequent': Replace with most frequent value
        - 'constant': Replace with a constant value

    scaler : str, default='standard'
        The scaling method:
        - 'standard': StandardScaler (mean=0, std=1)
        - 'minmax': MinMaxScaler (scale to [0, 1])
        - 'robust': RobustScaler (scale using quantiles)
        - None: No scaling

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Configured preprocessing pipeline

    Examples
    --------
    >>> from src.processing import create_preprocessing_pipeline
    >>> pipeline = create_preprocessing_pipeline()
    >>> X_processed = pipeline.fit_transform(X)
    >>> # Pipeline is ready for training or inference
    """
    # Validate imputer strategy
    valid_strategies = ["mean", "median", "most_frequent", "constant"]
    if imputer_strategy not in valid_strategies:
        raise ValueError(
            f"Invalid imputer_strategy: {imputer_strategy}. " f"Must be one of {valid_strategies}"
        )

    # Build pipeline steps
    pipeline_steps = [
        ("zero_to_nan", ZeroToNanTransformer()),
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
    ]

    # Add scaler if specified
    if scaler == "standard":
        from sklearn.preprocessing import StandardScaler

        pipeline_steps.append(("scaler", StandardScaler()))
    elif scaler == "minmax":
        from sklearn.preprocessing import MinMaxScaler

        pipeline_steps.append(("scaler", MinMaxScaler()))
    elif scaler == "robust":
        from sklearn.preprocessing import RobustScaler

        pipeline_steps.append(("scaler", RobustScaler()))
    elif scaler is not None:
        raise ValueError(
            f"Invalid scaler: {scaler}. Must be 'standard', 'minmax', 'robust', or None"
        )

    # Create pipeline
    pipeline = Pipeline(pipeline_steps)

    return pipeline


def save_preprocessing_pipeline(
    pipeline: Pipeline, filepath: str = "models/preprocessing_pipeline.pkl"
) -> None:
    """
    Save preprocessing pipeline to disk.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted preprocessing pipeline to save
    filepath : str, default='models/preprocessing_pipeline.pkl'
        Path where to save the pipeline

    Examples
    --------
    >>> from src.processing import create_preprocessing_pipeline, save_preprocessing_pipeline
    >>> pipeline = create_preprocessing_pipeline()
    >>> pipeline.fit(X_train)
    >>> save_preprocessing_pipeline(pipeline, 'models/my_pipeline.pkl')
    """
    joblib.dump(pipeline, filepath)
    print(f" Preprocessing pipeline saved to: {filepath}")


def load_preprocessing_pipeline(
    filepath: str = "models/preprocessing_pipeline.pkl",
) -> Pipeline:
    """
    Load preprocessing pipeline from disk.

    Parameters
    ----------
    filepath : str, default='models/preprocessing_pipeline.pkl'
        Path to the saved pipeline

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Loaded preprocessing pipeline

    Examples
    --------
    >>> from src.processing import load_preprocessing_pipeline
    >>> pipeline = load_preprocessing_pipeline()
    >>> X_processed = pipeline.transform(X)
    """
    pipeline = joblib.load(filepath)
    print(f" Preprocessing pipeline loaded from: {filepath}")
    return pipeline


# Example usage when run as script
if __name__ == "__main__":
    print("Preprocessing Module Loaded Successfully")
    print("\nAvailable components:")
    print("  1. ZeroToNanTransformer - Custom transformer class")
    print("  2. create_preprocessing_pipeline() - Function to create pipeline")
    print("  3. save_preprocessing_pipeline() - Function to save pipeline")
    print("  4. load_preprocessing_pipeline() - Function to load pipeline")
    print("\nExample usage:")
    print("  from src.processing import create_preprocessing_pipeline")
    print("  pipeline = create_preprocessing_pipeline()")
    print("  X_processed = pipeline.fit_transform(X)")
