# Diabetes Prediction - Project Roadmap

This document outlines the remaining priorities and future improvements for the Diabetes Prediction MLOps project.

**Current Status:** ✅ **Priority 2 (Production Readiness) - COMPLETED**

---

## 📊 Overall Progress

| Priority | Status | Progress |
|----------|--------|----------|
| Priority 1: Model Improvements | ⏳ Pending | 0% |
| Priority 2: Production Readiness | ✅ Complete | 100% |
| Priority 3: Monitoring & Maintenance | ⏳ Pending | 0% |
| Priority 4: Testing | ⏳ Pending | 0% |
| Priority 5: Advanced Features | ⏳ Pending | 0% |
| Priority 6: Documentation | ✅ Complete | 100% |

---

## 🎯 Priority 1: Model Improvements

**Goal:** Enhance model performance and prediction accuracy

### 1.1 Hyperparameter Tuning

**Current State:** Using default hyperparameters

**Proposed Implementation:**

```python
# Create: notebooks/hyperparameter_tuning.ipynb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

# Logistic Regression Hyperparameters
lr_params = {
    'C': uniform(0.001, 100),
    'solver': ['liblinear', 'saga'],
    'penalty': ['l1', 'l2'],
    'max_iter': [500, 1000, 2000]
}

# Random Forest Hyperparameters
rf_params = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None]
}

# Randomized Search (faster)
lr_random = RandomizedSearchCV(
    LogisticRegression(class_weight='balanced'),
    param_distributions=lr_params,
    n_iter=50,
    cv=5,
    scoring='recall',
    n_jobs=-1,
    random_state=42
)

# Grid Search (thorough)
rf_grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid=rf_params,
    cv=5,
    scoring='recall',
    n_jobs=-1
)
```

**Benefits:**
- Improved model performance
- Better generalization
- Optimized trade-offs between bias and variance

**Expected Time:** 1-2 days

**Files to Create:**
- `notebooks/hyperparameter_tuning.ipynb`
- `src/tuning.py` (refactored production code)

**MLflow Integration:**
- Log each hyperparameter combination
- Compare runs in MLflow UI
- Select best run automatically

---

### 1.2 Advanced Class Imbalance Handling

**Current State:** Using `class_weight='balanced'`

**Proposed Implementation:**

```python
# Create: src/imbalance_handling.py
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Option 1: SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(
    sampling_strategy='auto',  # or 0.5 for specific ratio
    random_state=42,
    k_neighbors=5
)

X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
print(f"Original: {Counter(y_train)}")
print(f"After SMOTE: {Counter(y_train_smote)}")

# Option 2: SMOTE + Tomek links (cleaner decision boundary)
smote_tomek = SMOTETomek(
    smote=SMOTE(sampling_strategy=0.5),
    tomek=RandomUnderSampler(sampling_strategy=0.8),
    random_state=42
)

# Option 3: ADASYN (Adaptive Synthetic Sampling)
adasyn = ADASYN(
    sampling_strategy='auto',
    n_neighbors=5,
    random_state=42
)
```

**Comparison Table:**

| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| class_weight | Simple, fast | Limited improvement | Baseline |
| SMOTE | Effective, popular | Can create noise | General purpose |
| SMOTETomek | Cleaner boundaries | Slower | When overlap is high |
| ADASYN | Adaptive | Can overfit | Complex distributions |

**Benefits:**
- Better handling of minority class
- Improved Recall metric
- More balanced predictions

**Expected Time:** 1 day

---

### 1.3 Model Ensemble

**Current State:** Single best model

**Proposed Implementation:**

```python
# Create: src/ensemble.py
from sklearn.ensemble import (
    VotingClassifier,
    StackingClassifier,
    BaggingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 1. Voting Ensemble (hard/soft voting)
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(class_weight='balanced', random_state=42)),
        ('rf', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )),
        ('xgb', XGBClassifier(
            scale_pos_weight=2,  # handle imbalance
            random_state=42
        ))
    ],
    voting='soft',  # use probabilities
    n_jobs=-1
)

# 2. Stacking Ensemble (meta-learner)
stacking_clf = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(class_weight='balanced', random_state=42)),
        ('rf', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )),
        ('xgb', XGBClassifier(
            scale_pos_weight=2,
            random_state=42
        ))
    ],
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

# 3. Bagging (bootstrap aggregating)
bagging_clf = BaggingClassifier(
    base_estimator=LogisticRegression(class_weight='balanced'),
    n_estimators=50,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)
```

**Benefits:**
- Better predictive performance
- Reduced overfitting
- More robust predictions

**Expected Time:** 1-2 days

---

### 1.4 Additional Algorithms

**Current State:** Logistic Regression & Random Forest

**Proposed Implementation:**

```python
# Add to: src/training.py

# 1. XGBoost
def train_xgboost(X_train, X_test, y_train, y_test):
    from xgboost import XGBClassifier

    model = XGBClassifier(
        scale_pos_weight=2,  # handle class imbalance
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    with mlflow.start_run(run_name="XGBoost"):
        mlflow.log_param("model_type", "XGBoost")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        # ... logging code

    return model, metrics

# 2. LightGBM
def train_lightgbm(X_train, X_test, y_train, y_test):
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        class_weight='balanced',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    with mlflow.start_run(run_name="LightGBM"):
        mlflow.log_param("model_type", "LightGBM")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        # ... logging code

    return model, metrics

# 3. Support Vector Machine (SVM)
def train_svm(X_train, X_test, y_train, y_test):
    from sklearn.svm import SVC

    model = SVC(
        class_weight='balanced',
        probability=True,  # needed for predict_proba
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42
    )

    with mlflow.start_run(run_name="SVM"):
        mlflow.log_param("model_type", "SVM")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        # ... logging code

    return model, metrics
```

**Algorithm Comparison:**

| Algorithm | Strengths | Weaknesses | Expected Recall |
|-----------|-----------|------------|-----------------|
| Logistic Regression | Interpretable, fast | Linear assumptions | 70-80% |
| Random Forest | Handles non-linear | Can overfit | 75-85% |
| XGBoost | High accuracy | Complex tuning | 80-90% |
| LightGBM | Fast, accurate | Can overfit | 80-90% |
| SVM | Good for small data | Slow on large data | 75-85% |

**Expected Time:** 2-3 days

**Dependencies:**
```txt
# Add to requirements.txt
xgboost==2.1.0
lightgbm==4.5.0
imbalanced-learn==0.12.3
```

---

## 🔧 Priority 3: Monitoring & Maintenance

**Goal:** Ensure model reliability and performance in production

### 3.1 Model Monitoring

**Proposed Implementation:**

```python
# Create: src/monitoring.py
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

class ModelMonitor:
    """Monitor model performance and data drift"""

    def __init__(self, model_name: str, threshold: float = 0.1):
        self.model_name = model_name
        self.threshold = threshold
        self.baseline_stats = None

    def calculate_baseline(self, X_train: pd.DataFrame):
        """Calculate baseline statistics from training data"""
        self.baseline_stats = {
            'mean': X_train.mean(),
            'std': X_train.std(),
            'min': X_train.min(),
            'max': X_train.max()
        }

    def check_data_drift(self, X_new: pd.DataFrame) -> dict:
        """
        Detect data drift using Kolmogorov-Smirnov test

        Returns dict with drift statistics for each feature
        """
        drift_results = {}

        for column in X_new.columns:
            # KS test for numerical features
            statistic, p_value = stats.ks_2samp(
                self.baseline_stats['mean'][column].values,
                X_new[column].values
            )

            drift_results[column] = {
                'statistic': statistic,
                'p_value': p_value,
                'drift_detected': p_value < 0.05  # 5% significance
            }

        return drift_results

    def check_prediction_drift(
        self,
        predictions_recent: np.ndarray,
        predictions_baseline: np.ndarray
    ) -> dict:
        """Monitor prediction distribution drift"""
        # Chi-square test for categorical predictions
        from scipy.stats import chisquare

        observed = np.bincount(predictions_recent)
        expected = np.bincount(predictions_baseline)

        # Normalize
        observed = observed / observed.sum()
        expected = expected / expected.sum()

        chi2_stat, p_value = chisquare(f_observed, expected)

        return {
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'drift_detected': p_value < 0.05
        }

    def log_to_mlflow(self, metrics: dict):
        """Log monitoring metrics to MLflow"""
        with mlflow.start_run(nested=True):
            mlflow.log_metrics(metrics)
            mlflow.set_tag("monitoring_type", "drift_detection")

# Usage in production
# monitor = ModelMonitor("diabetes_prediction_v1")
# monitor.calculate_baseline(X_train)
#
# # For new predictions
# drift_report = monitor.check_data_drift(X_new)
# if drift_report['Glucose']['drift_detected']:
#     alert_team("Glucose distribution has shifted!")
```

**Features:**
- Data drift detection (KS test)
- Prediction drift monitoring
- Feature importance tracking
- Performance degradation alerts

**Expected Time:** 2-3 days

---

### 3.2 Logging System

**Current State:** Basic logging in API

**Proposed Implementation:**

```python
# Create: src/logger.py
import logging
import json
from datetime import datetime
from pathlib import Path

class PredictionLogger:
    """Log all predictions for audit trail and analysis"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger("prediction_logger")
        self.logger.setLevel(logging.INFO)

        # File handler
        log_file = self.log_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # JSON formatter
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def log_prediction(
        self,
        input_data: dict,
        prediction: int,
        probability: float,
        model_version: str,
        timestamp: datetime = None
    ):
        """Log a single prediction"""

        log_entry = {
            'timestamp': timestamp or datetime.now().isoformat(),
            'model_version': model_version,
            'input': input_data,
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': self._get_risk_level(probability)
        }

        self.logger.info(json.dumps(log_entry))

        # Also log to separate file for analysis
        self._append_to_csv(log_entry)

    def _get_risk_level(self, probability: float) -> str:
        if probability < 0.3:
            return "Low"
        elif probability < 0.5:
            return "Moderate"
        elif probability < 0.7:
            return "High"
        else:
            return "Very High"

    def _append_to_csv(self, log_entry: dict):
        """Append to CSV for easy analysis"""
        import pandas as pd

        csv_file = self.log_dir / "predictions.csv"

        # Flatten log entry
        flat_entry = {
            'timestamp': log_entry['timestamp'],
            'model_version': log_entry['model_version'],
            'prediction': log_entry['prediction'],
            'probability': log_entry['probability'],
            'risk_level': log_entry['risk_level']
        }

        # Add input features
        flat_entry.update({f"input_{k}": v for k, v in log_entry['input'].items()})

        # Append to CSV
        df = pd.DataFrame([flat_entry])

        if csv_file.exists():
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, mode='w', header=True, index=False)

    def get_statistics(self, days: int = 7) -> dict:
        """Get prediction statistics for last N days"""
        import pandas as pd

        csv_file = self.log_dir / "predictions.csv"

        if not csv_file.exists():
            return {}

        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Filter last N days
        cutoff = datetime.now() - timedelta(days=days)
        df_recent = df[df['timestamp'] > cutoff]

        return {
            'total_predictions': len(df_recent),
            'positive_cases': int(df_recent['prediction'].sum()),
            'positivity_rate': float(df_recent['prediction'].mean()),
            'avg_probability': float(df_recent['probability'].mean()),
            'risk_distribution': df_recent['risk_level'].value_counts().to_dict()
        }

# Integrate with API
# logger = PredictionLogger()
# logger.log_prediction(input_data, prediction, probability, "v1.0.0")
```

**Benefits:**
- Complete audit trail
- Easy analysis and reporting
- Regulatory compliance
- Debugging capability

**Expected Time:** 1-2 days

---

## 🧪 Priority 4: Testing

**Goal:** Ensure code quality and prevent regressions

### 4.1 Unit Tests

**Proposed Implementation:**

```python
# Create: tests/test_processing.py
import pytest
import pandas as pd
import numpy as np
from src.processing import (
    ZeroToNanTransformer,
    create_preprocessing_pipeline
)

class TestZeroToNanTransformer:
    """Test ZeroToNanTransformer"""

    def test_initialization(self):
        """Test transformer initialization"""
        transformer = ZeroToNanTransformer()
        assert transformer.medical_columns == [
            'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'
        ]

    def test_fit_returns_self(self):
        """Test fit method returns self"""
        transformer = ZeroToNanTransformer()
        df = pd.DataFrame({'Glucose': [0, 120], 'Age': [25, 30]})
        result = transformer.fit(df)
        assert result is transformer

    def test_transform_converts_zeros_to_nan(self):
        """Test zeros are converted to NaN in medical columns"""
        transformer = ZeroToNanTransformer()
        df = pd.DataFrame({
            'Glucose': [0, 120, 0],
            'BloodPressure': [0, 80, 70],
            'Age': [0, 25, 30]  # Not medical, should not change
        })

        result = transformer.fit_transform(df)

        assert result['Glucose'].isna().sum() == 2  # Two zeros
        assert result['BloodPressure'].isna().sum() == 1  # One zero
        assert result['Age'].isna().sum() == 0  # No change

    def test_transform_with_numpy_array(self):
        """Test transform works with numpy array after fit"""
        transformer = ZeroToNanTransformer()
        df = pd.DataFrame({
            'Glucose': [0, 120],
            'BloodPressure': [0, 80],
            'Age': [25, 30]
        })

        transformer.fit(df)
        result = transformer.transform(df.values)

        assert isinstance(result, pd.DataFrame)
        assert result['Glucose'].isna().sum() == 1

    def test_invalid_medical_columns_raises_error(self):
        """Test error when medical columns not found"""
        transformer = ZeroToNanTransformer()
        df = pd.DataFrame({'WrongColumn': [1, 2, 3]})

        with pytest.raises(ValueError, match="Medical columns not found"):
            transformer.fit_transform(df)

class TestPreprocessingPipeline:
    """Test preprocessing pipeline"""

    def test_pipeline_creation(self):
        """Test pipeline can be created"""
        pipeline = create_preprocessing_pipeline()
        assert pipeline is not None
        assert len(pipeline.steps) == 3  # zero_to_nan, imputer, scaler

    def test_pipeline_transform(self):
        """Test pipeline transforms data correctly"""
        pipeline = create_preprocessing_pipeline()

        df = pd.DataFrame({
            'Glucose': [0, 120, 140],
            'BloodPressure': [0, 80, 70],
            'SkinThickness': [0, 20, 25],
            'Insulin': [0, 80, 90],
            'BMI': [0, 25.0, 30.0],
            'Age': [25, 30, 35]
        })

        pipeline.fit(df)
        result = pipeline.transform(df)

        # Check no NaN values
        assert not np.isnan(result).any()

        # Check scaled (approximately mean=0, std=1)
        assert np.abs(result.mean()).max() < 1.0
```

```python
# Create: tests/test_training.py
import pytest
import numpy as np
from src.training import load_data, split_data
from sklearn.datasets import make_classification

class TestDataLoading:
    """Test data loading functions"""

    def test_load_data(self):
        """Test data loads correctly"""
        # For this test, we need actual data file
        # Or create mock data
        pass

    def test_split_data(self):
        """Test train/test split"""
        X, y = make_classification(
            n_samples=100,
            n_features=8,
            n_informative=5,
            random_state=42
        )
        X = pd.DataFrame(X, columns=[
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ])
        y = pd.Series(y)

        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42
        )

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

class TestModelTraining:
    """Test model training functions"""

    @pytest.mark.slow
    def test_train_logistic_regression(self):
        """Test logistic regression training"""
        # Create sample data
        X_train, X_test, y_train, y_test = self._get_sample_data()

        model, metrics = train_logistic_regression(
            X_train, X_test, y_train, y_test
        )

        assert model is not None
        assert 'test_recall' in metrics
        assert metrics['test_recall'] >= 0.0
        assert metrics['test_recall'] <= 1.0

    def _get_sample_data(self):
        """Helper to create sample data"""
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=200,
            n_features=8,
            n_informative=5,
            random_state=42
        )
        X_train, X_test = X[:160], X[160:]
        y_train, y_test = y[:160], y[160:]

        return X_train, X_test, y_train, y_test
```

```python
# Create: tests/test_api.py
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

class TestAPIEndpoints:
    """Test FastAPI endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data

    def test_predict_endpoint_valid_input(self):
        """Test prediction with valid input"""
        input_data = {
            "pregnancies": 1,
            "glucose": 120,
            "blood_pressure": 70,
            "skin_thickness": 20,
            "insulin": 80,
            "bmi": 32.0,
            "diabetes_pedigree_function": 0.5,
            "age": 33
        }

        response = client.post("/predict", json=input_data)
        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "risk_level" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1

    def test_predict_endpoint_invalid_glucose(self):
        """Test prediction with invalid glucose"""
        input_data = {
            "pregnancies": 1,
            "glucose": 500,  # Invalid: > 300
            "blood_pressure": 70,
            "skin_thickness": 20,
            "insulin": 80,
            "bmi": 32.0,
            "diabetes_pedigree_function": 0.5,
            "age": 33
        }

        response = client.post("/predict", json=input_data)
        assert response.status_code == 422  # Validation error

    def test_model_info_endpoint(self):
        """Test model information endpoint"""
        response = client.get("/model-info")
        assert response.status_code == 200

        data = response.json()
        assert "model_type" in data
        assert "feature_count" in data
        assert data["feature_count"] == 8
```

**Test Coverage Goal:** > 80%

**Expected Time:** 2-3 days

---

### 4.2 Integration Tests

```python
# Create: tests/test_integration.py
import pytest
import pandas as pd
from src.processing import create_preprocessing_pipeline
from src.training import train_logistic_regression
import joblib
import os

class TestIntegration:
    """Integration tests for end-to-end workflows"""

    def test_full_training_pipeline(self):
        """Test complete training pipeline"""
        # Load data
        X, y = load_data()

        # Split
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Create pipeline
        pipeline = create_preprocessing_pipeline()
        pipeline.fit(X_train)

        # Transform
        X_train_processed = pipeline.transform(X_train)
        X_test_processed = pipeline.transform(X_test)

        # Train
        model, metrics = train_logistic_regression(
            X_train_processed, X_test_processed, y_train, y_test
        )

        # Assertions
        assert model is not None
        assert metrics['test_recall'] > 0.5  # Minimum acceptable recall
        assert X_train_processed.shape[1] == 8  # All features present

    def test_model_persistence(self):
        """Test saving and loading models"""
        from src.training import save_model, load_model

        # Train a simple model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit([[1], [2]], [0, 1])

        # Save
        save_path = "tests/test_model.pkl"
        save_model(model, save_path)

        # Load
        loaded_model = load_model(save_path)

        # Check prediction matches
        assert (model.predict([[1]]) == loaded_model.predict([[1]])).all()

        # Cleanup
        os.remove(save_path)

    def test_prediction_pipeline(self):
        """Test prediction from raw input to output"""
        # This would test the complete flow used in app.py and api.py
        pass
```

**Expected Time:** 1-2 days

---

## 🚀 Priority 5: Advanced Features

**Goal:** Add sophisticated capabilities for better insights

### 5.1 Feature Engineering

**Proposed Implementation:**

```python
# Create: src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create domain-specific features for diabetes prediction"""

    def __init__(self):
        self.feature_names_in_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """Learn from data"""
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data with new features"""
        X_new = X.copy()

        # 1. BMI * Age interaction
        X_new['BMI_Age_Interaction'] = X_new['BMI'] * X_new['Age']

        # 2. Glucose to Insulin ratio (if both > 0)
        X_new['Glucose_Insulin_Ratio'] = np.where(
            (X_new['Insulin'] > 0) & (X_new['Glucose'] > 0),
            X_new['Glucose'] / X_new['Insulin'],
            0
        )

        # 3. Age groups
        X_new['Age_Group'] = pd.cut(
            X_new['Age'],
            bins=[0, 30, 45, 60, 100],
            labels=['Young', 'Middle', 'Senior', 'Elderly']
        ).cat.codes  # Convert to numerical

        # 4. BMI categories
        X_new['BMI_Category'] = pd.cut(
            X_new['BMI'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        ).cat.codes

        # 5. Pregnancy risk groups
        X_new['Pregnancy_Risk'] = pd.cut(
            X_new['Pregnancies'],
            bins=[-1, 0, 2, 5, 20],
            labels=['None', 'Low', 'Medium', 'High']
        ).cat.codes

        # 6. Metabolic syndrome indicator
        # (High glucose OR high BMI OR high BP)
        X_new['Metabolic_Syndrome_Risk'] = (
            (X_new['Glucose'] > 100).astype(int) +
            (X_new['BMI'] > 25).astype(int) +
            (X_new['BloodPressure'] > 80).astype(int)
        ) / 3  # Normalize to 0-1

        # 7. Standardized glucose (by age)
        X_new['Glucose_Age_Std'] = X_new['Glucose'] / (X_new['Age'] + 1)

        return X_new

    def get_feature_names_out(self):
        """Return names of generated features"""
        return [
            'BMI_Age_Interaction',
            'Glucose_Insulin_Ratio',
            'Age_Group',
            'BMI_Category',
            'Pregnancy_Risk',
            'Metabolic_Syndrome_Risk',
            'Glucose_Age_Std'
        ]
```

**Benefits:**
- Capture non-linear relationships
- Domain knowledge injection
- Better model performance

**Expected Time:** 1-2 days

---

### 5.2 Model Explainability (XAI)

**Proposed Implementation:**

```python
# Create: src/explainability.py
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

class ModelExplainer:
    """Generate explanations for model predictions"""

    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.explainer = None

    def setup_explainer(self, X_background):
        """Setup SHAP explainer"""
        # Preprocess background data
        X_background_processed = self.preprocessor.transform(X_background)

        # Choose appropriate explainer based on model type
        model_type = type(self.model).__name__

        if model_type in ['RandomForestClassifier', 'XGBClassifier']:
            self.explainer = shap.TreeExplainer(self.model)
        elif model_type == 'LogisticRegression':
            self.explainer = shap.LinearExplainer(
                self.model,
                X_background_processed
            )
        else:
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                X_background_processed
            )

    def explain_single_prediction(
        self,
        input_data: dict,
        save_path: str = None
    ) -> dict:
        """
        Generate explanation for single prediction

        Returns:
            dict with SHAP values and interpretation
        """
        # Convert to DataFrame and preprocess
        input_df = pd.DataFrame([input_data])
        input_processed = self.preprocessor.transform(input_df)

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(input_processed)

        # Feature importance for this prediction
        feature_importance = pd.DataFrame({
            'feature': input_data.keys(),
            'value': list(input_data.values()),
            'shap_value': shap_values[0][:, 1]  # For class 1 (diabetes)
        }).sort_values('shap_value', key=abs, ascending=False)

        # Generate interpretation
        top_positive = feature_importance.iloc[0]['feature']
        top_negative = feature_importance.iloc[-1]['feature']

        interpretation = {
            'prediction': 'Diabetes' if self.model.predict(input_processed)[0] == 1 else 'No Diabetes',
            'probability': float(self.model.predict_proba(input_processed)[0, 1]),
            'top_risk_factor': top_positive,
            'top_protective_factor': top_negative,
            'feature_importance': feature_importance.to_dict('records')
        }

        # Generate plot if save_path provided
        if save_path:
            self._plot_single_explanation(
                shap_values, input_processed, save_path
            )

        return interpretation

    def explain_global_model(
        self,
        X_sample: pd.DataFrame,
        save_path: str = None
    ) -> dict:
        """
        Generate global model explanation

        Returns:
            dict with overall feature importance
        """
        # Preprocess
        X_processed = self.preprocessor.transform(X_sample)

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_processed)

        # Mean absolute SHAP values for each feature
        mean_shap = np.abs(shap_values[0]).mean(axis=0)

        feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)

        # Generate summary plot
        if save_path:
            self._plot_global_explanation(
                shap_values, X_sample, X_processed, save_path
            )

        return {
            'feature_importance': feature_importance.to_dict('records'),
            'top_feature': feature_importance.iloc[0]['feature'],
            'total_features': len(feature_importance)
        }

    def _plot_single_explanation(self, shap_values, input_data, save_path):
        """Generate force plot for single prediction"""
        shap.initjs()

        plot = shap.force_plot(
            self.explainer.expected_value[1],
            shap_values[0][:, 1],
            input_data[0],
            feature_names=self.preprocessor.get_feature_names_out(),
            show=False
        )

        # Save
        shap.save_html(save_path, plot)

    def _plot_global_explanation(self, shap_values, X_original, X_processed, save_path):
        """Generate summary plot for global explanation"""
        plt.figure()

        shap.summary_plot(
            shap_values[0],
            X_processed,
            feature_names=self.preprocessor.get_feature_names_out(),
            show=False
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# Integration with API
# @app.post("/explain")
# async def explain_prediction(input_data: DiabetesInput):
#     explainer = ModelExplainer(model, pipeline)
#     explainer.setup_explainer(X_train)  # Setup once
#
#     explanation = explainer.explain_single_prediction(
#         input_data.dict(),
#         save_path="explanations/latest.html"
#     )
#
#     return explanation
```

**Benefits:**
- Understand model decisions
- Build trust with users
- Regulatory compliance (GDPR Article 15)
- Debugging

**Expected Time:** 2-3 days

**Dependencies:**
```txt
# Add to requirements.txt
shap==0.46.0
```

---

### 5.3 A/B Testing Framework

**Proposed Implementation:**

```python
# Create: src/ab_testing.py
import pandas as pd
import numpy as np
from typing import Dict, List
import mlflow

class ABTestFramework:
    """Framework for A/B testing different models"""

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.models = {}
        self.results = {}

    def add_model(self, model_name: str, model, preprocessor):
        """Add a model to the test"""
        self.models[model_name] = {
            'model': model,
            'preprocessor': preprocessor
        }

    def run_ab_test(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Run A/B test across all models

        Returns comparison DataFrame
        """
        results_list = []

        for model_name, model_data in self.models.items():
            model = model_data['model']
            preprocessor = model_data['preprocessor']

            # Preprocess
            X_test_processed = preprocessor.transform(X_test)

            # Predict
            y_pred = model.predict(X_test_processed)
            y_proba = model.predict_proba(X_test_processed)[:, 1]

            # Calculate metrics
            from sklearn.metrics import (
                accuracy_score, precision_score,
                recall_score, f1_score, roc_auc_score
            )

            metrics = {
                'model': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_proba)
            }

            results_list.append(metrics)

            # Log to MLflow
            with mlflow.start_run(run_name=f"AB_Test_{model_name}"):
                mlflow.log_params({"test_name": self.test_name})
                mlflow.log_metrics(metrics)
                mlflow.set_tag("ab_test", self.test_name)

        self.results = pd.DataFrame(results_list)
        return self.results

    def select_winner(self, metric: str = 'recall') -> str:
        """
        Select winning model based on specified metric

        Args:
            metric: Metric to compare ('recall', 'f1', 'roc_auc', etc.)
        """
        if self.results is None or len(self.results) == 0:
            raise ValueError("No test results available. Run run_ab_test first.")

        # Find model with highest metric
        winner = self.results.loc[self.results[metric].idxmax(), 'model']

        print(f"🏆 Winner: {winner} with {metric} = {self.results.loc[self.results[metric].idxmax(), metric]:.4f}")

        return winner

    def generate_report(self, save_path: str = None) -> dict:
        """Generate A/B test report"""
        if self.results is None or len(self.results) == 0:
            raise ValueError("No test results available.")

        report = {
            'test_name': self.test_name,
            'num_models': len(self.models),
            'results_table': self.results.to_dict('records'),
            'best_by_metric': {}
        }

        # Find best model for each metric
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in metrics:
            best_model = self.results.loc[self.results[metric].idxmax(), 'model']
            best_score = self.results[metric].max()
            report['best_by_metric'][metric] = {
                'model': best_model,
                'score': float(best_score)
            }

        # Save to file if path provided
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)

        return report

# Usage example:
# ab_test = ABTestFramework("Model_Comparison_v1")
# ab_test.add_model("Logistic_Regression", lr_model, pipeline)
# ab_test.add_model("Random_Forest", rf_model, pipeline)
# ab_test.add_model("XGBoost", xgb_model, pipeline)
#
# results = ab_test.run_ab_test(X_test, y_test)
# print(results)
#
# winner = ab_test.select_winner('recall')
# report = ab_test.generate_report('ab_test_report.json')
```

**Benefits:**
- Data-driven model selection
- Easy comparison
- Automated deployment decisions

**Expected Time:** 2 days

---

## 📊 Priority Comparison

| Priority | Impact | Effort | Time | Dependencies |
|----------|--------|--------|------|--------------|
| 1. Model Improvements | High | High | 1-2 weeks | None |
| 2. Production Readiness | High | Medium | ✅ Done | None |
| 3. Monitoring | High | Medium | 1 week | Priority 2 |
| 4. Testing | High | Medium | 1 week | None |
| 5. Advanced Features | Medium | High | 1-2 weeks | Priority 1 |

---

## 🎯 Recommended Implementation Order

### Phase 1: Foundation (Week 1-2)
1. **Priority 4: Testing** (Unit & Integration)
   - Essential for code quality
   - Prevents regressions
   - Easy to implement incrementally

### Phase 2: Model Enhancement (Week 3-4)
2. **Priority 1.1: Hyperparameter Tuning**
   - Immediate performance gain
   - Builds on existing code

3. **Priority 1.2: SMOTE for Class Imbalance**
   - Addresses known issue
   - Significant recall improvement

### Phase 3: Production Readiness (Week 5-6)
4. **Priority 3.1: Model Monitoring**
   - Critical for production
   - Enables data drift detection

5. **Priority 3.2: Enhanced Logging**
   - Audit trail
   - Debugging capability

### Phase 4: Advanced Features (Week 7-8)
6. **Priority 1.3: Model Ensemble**
   - Performance improvement
   - More robust predictions

7. **Priority 5.2: Model Explainability**
   - User trust
   - Regulatory compliance

### Phase 5: Innovation (Week 9-10)
8. **Priority 1.4: Additional Algorithms (XGBoost, LightGBM)**
   - State-of-the-art performance

9. **Priority 5.1: Feature Engineering**
   - Domain knowledge
   - Custom features

10. **Priority 5.3: A/B Testing Framework**
    - Continuous improvement
    - Easy comparison

---

## 💡 Quick Wins (< 1 day each)

1. **Add confidence intervals** to predictions
2. **Add example inputs** in Streamlit sidebar
3. **Create prediction export** feature (download CSV)
4. **Add model comparison** page in Streamlit
5. **Create batch prediction** UI
6. **Add prediction history** view
7. **Create model performance dashboard**

---

## 📋 Additional Ideas

### Ideas for Future Consideration

1. **Multi-language Support**: Translate UI to Indonesian/other languages
2. **Mobile App**: React Native or Flutter app
3. **Email Notifications**: Alert users for high-risk predictions
4. **Doctor Consultation Integration**: Book appointments directly
5. **Patient History Tracking**: Track predictions over time
6. **Educational Content**: Show diabetes information based on results
7. **Community Features**: Forum or support groups
8. **Integration with Health APIs**: Apple HealthKit, Google Fit
9. **Federated Learning**: Train across multiple hospitals
10. **Real-time Monitoring**: Live dashboard of predictions

---

## 🎓 Learning Resources

For implementing these priorities:

**Hyperparameter Tuning:**
- Scikit-learn GridSearchCV documentation
- "Optuna" framework for advanced optimization

**Class Imbalance:**
- Imbalanced-learn library documentation
- SMOTE research papers

**Ensemble Methods:**
- "Ensemble Methods in Machine Learning" - Zhou Zhihua
- Kaggle ensemble tutorials

**SHAP/XAI:**
- SHAP documentation: https://shap.readthedocs.io/
- "Interpretable Machine Learning" - Christoph Molnar

**Testing:**
- Pytest documentation
- "Test-Driven Development with Python" - Harry Percival

---

## 📝 Template for New Features

When implementing any of these priorities, follow this template:

```markdown
## [Feature Name]

### Description
[Brief description of the feature]

### Files to Create/Modify
- `path/to/file1.py`
- `path/to/file2.py`

### Implementation Steps
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Testing Strategy
- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing

### Documentation
- [ ] Update README.md
- [ ] Add examples
- [ ] Update API docs

### MLflow Integration
- [ ] Log parameters
- [ ] Log metrics
- [ ] Save model

### Success Criteria
- [ ] Meets performance target
- [ ] All tests pass
- [ ] Documentation complete
```

---

## 🚀 Getting Started

To start implementing any priority:

```bash
# Create new branch
git checkout -b feature/[priority-name]

# Create necessary files
mkdir -p src/[feature-name]
touch src/[feature-name].py
touch tests/test_[feature-name].py

# Run tests
pytest tests/

# When done, create PR
git push origin feature/[priority-name]
```

---

**Last Updated:** 2026-02-07

**Maintainer:** Claude Code

**Status:** 📝 Planning Phase
