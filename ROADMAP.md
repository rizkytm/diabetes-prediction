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
| Priority 7: Workflow Orchestration | ⏳ Future | 0% |

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
| 7. Workflow Orchestration | Medium | High | 1-2 weeks | Priority 2, 3 |

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

### Phase 6: Orchestration & Automation (Week 11-12) - **Optional**
11. **Priority 7: Workflow Orchestration**
   - Start with GitHub Actions scheduled workflows
   - Migrate to Prefect if needed (when >5 workflows)
   - Consider Apache Airflow for enterprise-scale
   - Implement automated retraining DAG
   - Implement batch prediction DAG
   - Implement model monitoring DAG

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

**Class Imalance:**
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

**Apache Airflow & Workflow Orchestration:**
- Apache Airflow documentation: https://airflow.apache.org/docs/
- "Data Pipelines with Apache Airflow" - Bas Harenslak & Julian Rut
- Prefect documentation: https://docs.prefect.io/
- "Orchestration Patterns for ML" - Various blog posts
- Airflow by Example: https://airflowbyexample.org/

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

## 🎛️ Priority 7: Workflow Orchestration

**Goal:** Automate ML workflows and pipelines at scale

### When to Implement This Priority

**Trigger Points:**
- ✅ System is in production with active users
- ✅ Need for scheduled/recurring jobs
- ✅ Multiple data pipelines to manage
- ✅ Complex workflow dependencies
- ✅ Need for robust monitoring and alerting

**Prerequisites:**
- Priority 2 (Production Readiness) complete
- Priority 3 (Monitoring & Maintenance) in place
- Regular model retraining requirements
- Batch prediction needs

---

### 7.1 Apache Airflow Integration

**Current State:** Using GitHub Actions for CI/CD only

**Proposed Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions                            │
│  - CI/CD (lint, test, build Docker images)                  │
│  - Deploy on push to main                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Apache Airflow                              │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Retraining DAG   │  │ Batch Prediction │                │
│  │ - Weekly         │  │ - Daily          │                │
│  │ - MLflow tracking│  │ - Alert high risk│                │
│  └──────────────────┘  └──────────────────┘                │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Monitoring DAG   │  │ Data Pipeline DAG│                │
│  │ - Data drift     │  │ - ETL            │                │
│  │ - Performance    │  │ - Validation     │                │
│  └──────────────────┘  └──────────────────┘                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   MLflow Server                              │
│  - Experiment tracking                                      │
│  - Model registry                                           │
│  - Metrics visualization                                    │
└─────────────────────────────────────────────────────────────┘
```

**Proposed Implementation:**

#### DAG 1: Model Retraining Pipeline

```python
# Create: dags/01_model_retraining.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.slack.operators.slack import SlackAPIPostOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'diabetes_model_retraining',
    default_args=default_args,
    description='Weekly automated model retraining with quality checks',
    schedule_interval='@weekly',  # Every Sunday at midnight
    catchup=False,
    tags=['ml', 'retraining', 'diabetes', 'weekly'],
)

def fetch_latest_data(**context):
    """Fetch latest training data from database or API"""
    import pandas as pd
    from datetime import datetime

    # TODO: Connect to your data source
    # Example: Query database for new records since last training

    # For now, just log
    print(f"[{datetime.now()}] Fetching latest data...")

    # Update data/diabetes.csv
    # TODO: Replace with actual data fetch logic
    # df = pd.read_sql("SELECT * FROM patients WHERE date > last_training_date", con)
    # df.to_csv('data/diabetes.csv', index=False)

    return {
        'rows_fetched': 768,  # Example
        'last_date': datetime.now().isoformat()
    }

def train_and_evaluate_models(**context):
    """Train all models with MLflow tracking"""
    import sys
    sys.path.append('/opt/airflow')

    from src.training import run_training_pipeline
    import mlflow

    # Set MLflow tracking
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Diabetes_Prediction_Auto")

    # Run training pipeline
    results = run_training_pipeline(
        data_path='data/diabetes.csv',
        models_to_train=['logistic_regression', 'random_forest'],
        save_models=True,
        models_dir='models'
    )

    # Extract metrics
    rf_metrics = results['random_forest'][1]
    lr_metrics = results['logistic_regression'][1]

    # Quality checks
    if rf_metrics['test_recall'] < 0.70:
        raise ValueError(f"Random Forest recall below threshold: {rf_metrics['test_recall']:.4f}")

    if rf_metrics['test_recall'] - rf_metrics['train_recall'] < -0.15:
        raise ValueError("Severe overfitting detected!")

    # Push metrics to XCom
    context['task_instance'].xcom_push(key='rf_recall', value=rf_metrics['test_recall'])
    context['task_instance'].xcom_push(key='lr_recall', value=lr_metrics['test_recall'])

    print(f"✅ Training complete!")
    print(f"   Random Forest Recall: {rf_metrics['test_recall']:.4f}")
    print(f"   Logistic Regression Recall: {lr_metrics['test_recall']:.4f}")

    return results

def compare_with_baseline(**context):
    """Compare new model with current production model"""
    task_instance = context['task_instance']
    rf_recall = task_instance.xcom_pull(task_ids='train_models', key='rf_recall')

    # TODO: Load baseline metrics from database or MLflow
    baseline_recall = 0.75  # Example

    improvement = rf_recall - baseline_recall

    if improvement > 0.02:  # 2% improvement threshold
        print(f"✅ Model improved by {improvement:.2%}")
        return {'deploy': True, 'improvement': improvement}
    elif improvement > -0.02:
        print(f"ℹ️ Model performance stable ({improvement:.2%})")
        return {'deploy': False, 'improvement': improvement}
    else:
        print(f"⚠️ Model degraded by {abs(improvement):.2%}")
        return {'deploy': False, 'improvement': improvement}

def deploy_if_better(**context):
    """Deploy new model if performance improved"""
    task_instance = context['task_instance']
    comparison = task_instance.xcom_pull(task_ids='compare_baseline', key='return_value')

    if comparison['deploy']:
        # TODO: Deploy to production
        # - Copy to production models directory
        # - Update API to use new model
        # - Log deployment to MLflow

        print(f"🚀 Deploying new model to production!")

        # Notify team
        SlackAPIPostOperator(
            task_id='notify_deployment',
            slack_conn_id='slack_default',
            text=f"🎉 New model deployed with {comparison['improvement']:.2%} improvement!",
            channel='#ml-deployments',
            username='Airflow ML Bot',
            dag=dag
        ).execute(context=context)
    else:
        print(f"⏭️ Skipping deployment - no significant improvement")

def send_training_report(**context):
    """Send summary report of training run"""
    task_instance = context['task_instance']
    comparison = task_instance.xcom_pull(task_ids='compare_baseline', key='return_value')

    report = f"""
    📊 *Model Retraining Report*

    *Date:* {datetime.now().strftime('%Y-%m-%d %H:%M')}
    *Improvement:* {comparison['improvement']:.2%}
    *Status:* {'✅ Deployed' if comparison['deploy'] else '⏭️ Not deployed'}

    Check MLflow for details: http://mlflow:5000
    """

    SlackAPIPostOperator(
        task_id='send_report',
        slack_conn_id='slack_default',
        text=report,
        channel='#ml-reports',
        username='Airflow ML Bot',
        dag=dag
    ).execute(context=context)

# Define tasks
fetch_task = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_latest_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_models',
    python_callable=train_and_evaluate_models,
    dag=dag,
)

compare_task = PythonOperator(
    task_id='compare_baseline',
    python_callable=compare_with_baseline,
    dag=dag,
)

deploy_task = PythonOperator(
    task_id='deploy_if_better',
    python_callable=deploy_if_better,
    dag=dag,
)

report_task = PythonOperator(
    task_id='send_report',
    python_callable=send_training_report,
    dag=dag,
)

# Set dependencies
fetch_task >> train_task >> compare_task >> deploy_task >> report_task
```

#### DAG 2: Batch Prediction Pipeline

```python
# Create: dags/02_batch_prediction.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.slack.operators.slack import SlackAPIPostOperator
from datetime import datetime, timedelta
import pandas as pd

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'diabetes_batch_predictions',
    default_args=default_args,
    description='Daily batch predictions for new patients',
    schedule_interval='0 6 * * *',  # 6 AM daily
    catchup=False,
    tags=['batch', 'prediction', 'daily'],
)

def fetch_patients_needing_prediction(**context):
    """Fetch patients who need prediction"""
    import pandas as pd

    # TODO: Query database for new patients
    # SELECT * FROM patients WHERE prediction IS NULL AND created_at > yesterday

    # Example: Simulate fetching
    patients = pd.DataFrame({
        'patient_id': [1, 2, 3, 4, 5],
        'Pregnancies': [1, 0, 2, 0, 4],
        'Glucose': [120, 140, 110, 180, 95],
        'BloodPressure': [70, 80, 72, 90, 68],
        'SkinThickness': [20, 25, 18, 30, 15],
        'Insulin': [80, 100, 75, 150, 60],
        'BMI': [32.0, 35.0, 28.0, 42.0, 26.0],
        'DiabetesPedigreeFunction': [0.5, 0.6, 0.4, 0.8, 0.3],
        'Age': [33, 45, 28, 55, 24]
    })

    # Save to temp file
    patients.to_csv('/tmp/batch_patients.csv', index=False)

    print(f"Fetched {len(patients)} patients for prediction")
    return len(patients)

def run_batch_predictions(**context):
    """Run predictions on batch"""
    import sys
    sys.path.append('/opt/airflow')

    import joblib
    from src.processing import load_preprocessing_pipeline

    # Load model and pipeline
    model = joblib.load('models/best_model.pkl')
    pipeline = load_preprocessing_pipeline('models/preprocessing_pipeline.pkl')

    # Load patients
    patients = pd.read_csv('/tmp/batch_patients.csv')
    patient_ids = patients['patient_id'].values if 'patient_id' in patients.columns else None

    # Remove patient_id for prediction
    if patient_ids is not None:
        X = patients.drop('patient_id', axis=1)
    else:
        X = patients

    # Preprocess
    X_processed = pipeline.transform(X)

    # Predict
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)[:, 1]

    # Add results
    patients['prediction'] = predictions
    patients['probability'] = probabilities
    patients['risk_level'] = pd.cut(
        patients['probability'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Low', 'Moderate', 'High', 'Very High']
    )

    # Save results
    patients.to_csv('/tmp/batch_predictions.csv', index=False)

    high_risk_count = len(patients[patients['probability'] > 0.7])

    print(f"✅ Batch prediction complete: {len(patients)} predictions")
    print(f"   High risk patients: {high_risk_count}")

    context['task_instance'].xcom_push(key='high_risk_count', value=high_risk_count)
    context['task_instance'].xcom_push(key='total_predictions', value=len(patients))

    return len(patients)

def save_predictions_to_db(**context):
    """Save predictions to database"""
    import pandas as pd

    predictions = pd.read_csv('/tmp/batch_predictions.csv')

    # TODO: Insert into database
    # UPDATE patients SET prediction = ?, probability = ?, risk_level = ? WHERE patient_id = ?

    print(f"💾 Saved {len(predictions)} predictions to database")

    return len(predictions)

def alert_high_risk_patients(**context):
    """Send alerts for high-risk patients"""
    task_instance = context['task_instance']
    high_risk_count = task_instance.xcom_pull(task_ids='run_predictions', key='high_risk_count')

    if high_risk_count > 0:
        alert = f"""
        ⚠️ *High Risk Alert*

        {high_risk_count} high-risk patients detected in today's batch prediction.

        Please review and take appropriate action.
        """

        SlackAPIPostOperator(
            task_id='alert_high_risk',
            slack_conn_id='slack_default',
            text=alert,
            channel='#health-alerts',
            username='Airflow Health Bot',
            dag=dag
        ).execute(context=context)

        print(f"⚠️ Sent alert for {high_risk_count} high-risk patients")
    else:
        print("✅ No high-risk patients detected")

# Define tasks
fetch_task = PythonOperator(
    task_id='fetch_patients',
    python_callable=fetch_patients_needing_prediction,
    dag=dag,
)

predict_task = PythonOperator(
    task_id='run_predictions',
    python_callable=run_batch_predictions,
    dag=dag,
)

save_task = PythonOperator(
    task_id='save_to_db',
    python_callable=save_predictions_to_db,
    dag=dag,
)

alert_task = PythonOperator(
    task_id='alert_high_risk',
    python_callable=alert_high_risk_patients,
    dag=dag,
)

# Dependencies
fetch_task >> predict_task >> save_task >> alert_task
```

#### DAG 3: Model Monitoring & Data Drift

```python
# Create: dags/03_model_monitoring.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'diabetes_model_monitoring',
    default_args=default_args,
    description='Daily model monitoring and data drift detection',
    schedule_interval='@daily',
    catchup=False,
    tags=['monitoring', 'drift', 'daily'],
)

def check_data_drift(**context):
    """Check for data distribution drift"""
    import sys
    sys.path.append('/opt/airflow')

    from src.monitoring import ModelMonitor
    import pandas as pd
    from scipy import stats

    monitor = ModelMonitor('diabetes_prediction_v1')

    # Load baseline (should be saved from training)
    # TODO: Load from MLflow or file
    X_train_baseline = pd.read_csv('data/X_train_baseline.csv')
    monitor.calculate_baseline(X_train_baseline)

    # Load recent data (last 24 hours)
    # TODO: Fetch from database
    X_recent = pd.read_csv('data/recent_predictions.csv')

    # Check drift
    drift_results = monitor.check_data_drift(X_recent)

    # Count drifted features
    drifted_count = sum([1 for v in drift_results.values() if v['drift_detected']])

    context['task_instance'].xcom_push(key='drifted_features', value=drifted_count)

    if drifted_count > 0:
        print(f"⚠️ Data drift detected in {drifted_count} features!")
        for feature, result in drift_results.items():
            if result['drift_detected']:
                print(f"   - {feature}: p-value={result['p_value']:.4f}")
    else:
        print("✅ No significant data drift detected")

    return drift_results

def check_prediction_drift(**context):
    """Check prediction distribution drift"""
    import sys
    sys.path.append('/opt/airflow')

    from src.monitoring import ModelMonitor
    import pandas as pd
    import numpy as np

    monitor = ModelMonitor('diabetes_prediction_v1')

    # Load baseline predictions
    baseline_predictions = np.load('models/baseline_predictions.npy')

    # Load recent predictions
    recent_predictions = pd.read_csv('data/recent_predictions.csv')['prediction'].values

    # Check drift
    drift_result = monitor.check_prediction_drift(recent_predictions, baseline_predictions)

    if drift_result['drift_detected']:
        print(f"⚠️ Prediction drift detected! p-value={drift_result['p_value']:.4f}")
    else:
        print("✅ No prediction drift detected")

    return drift_result

def check_model_performance(**context):
    """Check if model performance degraded"""
    # TODO: Calculate metrics on recent predictions with ground truth
    # Compare with baseline metrics

    # For now, simulate
    current_recall = 0.78
    baseline_recall = 0.80

    degradation = baseline_recall - current_recall

    if degradation > 0.05:  # 5% degradation threshold
        print(f"⚠️ Model performance degraded by {degradation:.2%}")
        return {'alert': True, 'degradation': degradation}
    else:
        print(f"✅ Model performance stable (degradation: {degradation:.2%})")
        return {'alert': False, 'degradation': degradation}

def generate_monitoring_report(**context):
    """Generate daily monitoring dashboard"""
    task_instance = context['task_instance']
    drifted_features = task_instance.xcom_pull(task_ids='check_data_drift', key='drifted_features')
    perf_result = task_instance.xcom_pull(task_ids='check_performance', key='return_value')

    report = {
        'date': datetime.now().isoformat(),
        'data_drift_detected': drifted_features > 0,
        'drifted_feature_count': drifted_features,
        'performance_degraded': perf_result['alert'],
        'degradation_pct': perf_result['degradation'],
        'overall_status': 'HEALTHY' if not (drifted_features > 0 or perf_result['alert']) else 'WARNING'
    }

    # TODO: Save to database or dashboard
    import json
    with open('/tmp/monitoring_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"📊 Monitoring report generated: {report['overall_status']}")

    return report

def alert_if_needed(**context):
    """Send alert if monitoring detected issues"""
    task_instance = context['task_instance']
    report = task_instance.xcom_pull(task_ids='generate_report', key='return_value')

    if report['overall_status'] == 'WARNING':
        from airflow.providers.slack.operators.slack import SlackAPIPostOperator

        alert = f"""
        ⚠️ *Model Monitoring Alert*

        *Date:* {report['date']}
        *Status:* {report['overall_status']}

        *Issues:*
        - Data drift: {'Yes' if report['data_drift_detected'] else 'No'} ({report['drifted_feature_count']} features)
        - Performance: {'Degraded' if report['performance_degraded'] else 'Stable'} ({report['degradation_pct']:.2%})

        Please investigate.
        """

        SlackAPIPostOperator(
            task_id='send_alert',
            slack_conn_id='slack_default',
            text=alert,
            channel='#ml-monitoring',
            dag=dag
        ).execute(context=context)

        print("⚠️ Alert sent to team")
    else:
        print("✅ No alerts needed - system healthy")

# Define tasks
drift_task = PythonOperator(
    task_id='check_data_drift',
    python_callable=check_data_drift,
    dag=dag,
)

pred_drift_task = PythonOperator(
    task_id='check_prediction_drift',
    python_callable=check_prediction_drift,
    dag=dag,
)

performance_task = PythonOperator(
    task_id='check_performance',
    python_callable=check_model_performance,
    dag=dag,
)

report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_monitoring_report,
    dag=dag,
)

alert_task = PythonOperator(
    task_id='alert_if_needed',
    python_callable=alert_if_needed,
    dag=dag,
)

# Dependencies
[drift_task, pred_drift_task, performance_task] >> report_task >> alert_task
```

---

### 7.2 Docker Compose Setup

**File: docker-compose-airflow.yml**

```yaml
version: '3.8'

services:
  # PostgreSQL database for Airflow
  postgres:
    image: postgres:13
    container_name: airflow-postgres
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-db:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U airflow"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis for Celery broker
  redis:
    image: redis:7-alpine
    container_name: airflow-redis
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Airflow Webserver
  airflow-webserver:
    image: apache/airflow:2.8.0-python3.12
    container_name: airflow-webserver
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AIRFLOW__CELERY__BROKER_URL: redis://redis:6379/0
      AIRFLOW__CELERY__RESULT_BACKEND: redis://redis:6379/0
      AIRFLOW__CORE__DAGS_FOLDER: /opt/airflow/dags
      AIRFLOW__CORE__PLUGINS_FOLDER: /opt/airflow/plugins
      AIRFLOW__WEBSERVER__WORKER_REFRESH_INTERVAL: 30
      AIRFLOW__WEBSERVER__SECRET_KEY: ${AIRFLOW_SECRET_KEY:-secret_key}
      # Load custom connections from env
      AIRFLOW_CONN_METADATA_DEFAULT: ${AIRFLOW_CONN_METADATA_DEFAULT}
    volumes:
      - ./dags:/opt/airflow/dags
      - ./plugins:/opt/airflow/plugins
      - ./logs:/opt/airflow/logs
      - ../models:/opt/airflow/models  # Share models with main app
      - ../data:/opt/airflow/data      # Share data with main app
      - ./airflow.cfg:/opt/airflow/airflow.cfg
    ports:
      - "8080:8080"
    command: webserver
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Airflow Scheduler
  airflow-scheduler:
    image: apache/airflow:2.8.0-python3.12
    container_name: airflow-scheduler
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AIRFLOW__CELERY__BROKER_URL: redis://redis:6379/0
      AIRFLOW__CELERY__RESULT_BACKEND: redis://redis:6379/0
      AIRFLOW__CORE__DAGS_FOLDER: /opt/airflow/dags
      AIRFLOW__WEBSERVER__SECRET_KEY: ${AIRFLOW_SECRET_KEY:-secret_key}
    volumes:
      - ./dags:/opt/airflow/dags
      - ./plugins:/opt/airflow/plugins
      - ./logs:/opt/airflow/logs
      - ../models:/opt/airflow/models
      - ../data:/opt/airflow/data
      - ./airflow.cfg:/opt/airflow/airflow.cfg
    command: scheduler
    healthcheck:
      test: ["CMD-SHELL", "pgrep -f scheduler"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Airflow Worker
  airflow-worker:
    image: apache/airflow:2.8.0-python3.12
    container_name: airflow-worker
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AIRFLOW__CELERY__BROKER_URL: redis://redis:6379/0
      AIRFLOW__CELERY__RESULT_BACKEND: redis://redis:6379/0
      AIRFLOW__CORE__DAGS_FOLDER: /opt/airflow/dags
      AIRFLOW__WEBSERVER__SECRET_KEY: ${AIRFLOW_SECRET_KEY:-secret_key}
    volumes:
      - ./dags:/opt/airflow/dags
      - ./plugins:/opt/airflow/plugins
      - ./logs:/opt/airflow/logs
      - ../models:/opt/airflow/models
      - ../data:/opt/airflow/data
      - ./airflow.cfg:/opt/airflow/airflow.cfg
    command: celery worker
    healthcheck:
      test: ["CMD-SHELL", "pgrep -f worker"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Airflow CLI (for initialization)
  airflow-init:
    image: apache/airflow:2.8.0-python3.12
    container_name: airflow-init
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AIRFLOW__WEBSERVER__SECRET_KEY: ${AIRFLOW_SECRET_KEY:-secret_key}
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ../models:/opt/airflow/models
      - ../data:/opt/airflow/data
    command: version
    restart: on-failure

volumes:
  postgres-db:
```

**File: airflow.cfg**

```ini
[core]
# Airflow home directory
airflow_home = /opt/airflow

# The executor class to use (CeleryExecutor for distributed execution)
executor = CeleryExecutor

# SQLAlchemy connection string for metadata database
sql_alchemy_conn = postgresql+psycopg2://airflow:airflow@postgres/airflow

# The folder where your airflow pipelines live
dags_folder = /opt/airflow/dags

# The folder where airflow plugins live
plugins_folder = /opt/airflow/plugins

# Hostname of the webserver
web_server_host = 0.0.0.0

# Default timezone
default_timezone = UTC

# Load custom DAG examples
load_examples = False

[logging]
# Base log folder
base_log_folder = /opt/airflow/logs

# Log filename format
log_filename_template = dag_id={{ ti.dag_id }}/run_id={{ ti.run_id }}/task={{ ti.task_id }}

[cli]
# DAGs to show when listing
dag_discovery_safe_mode = False

[scheduler]
# How often to scan DAGs for changes
dag_dir_list_interval = 30

[celery]
# Celery broker URL (Redis)
broker_url = redis://redis:6379/0

# Celery result backend (Redis)
result_backend = redis://redis:6379/0

[webserver]
# Expose config switch
expose_config = True

# RBAC
rbac = True
```

---

### 7.3 Requirements and Setup

**File: airflow-requirements.txt**

```txt
# Airflow core
apache-airflow==2.8.0
apache-airflow-providers-postgres==5.10.0
apache-airflow-providers-slack==8.6.0
apache-airflow-providers-http==4.11.0
apache-airflow-providers-cncf-kubernetes==7.4.0

# Additional DAG dependencies
psycopg2-binary==2.9.9
redis==5.0.1
pandas==2.2.0
scikit-learn==1.4.0
mlflow==2.18.0

# Monitoring
scipy==1.12.0

# Database (example - adjust as needed)
psycopg2==2.9.9
sqlalchemy==2.0.29
```

**Setup Script:**

```bash
#!/bin/bash
# setup-airflow.sh

echo "Setting up Apache Airflow..."

# Create directories
mkdir -p airflow/dags
mkdir -p airflow/plugins
mkdir -p airflow/logs
mkdir -p airflow/config

# Copy DAGs
cp dags/*.py airflow/dags/

# Copy configuration
cp airflow.cfg airflow/config/

# Build and start
docker-compose -f docker-compose-airflow.yml up -d

echo "Waiting for Airflow to start..."
sleep 30

# Initialize database
docker exec airflow-init airflow db migrate

# Create admin user
docker exec airflow-init airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

echo "✅ Airflow setup complete!"
echo "   Webserver: http://localhost:8080"
echo "   Username: admin"
echo "   Password: admin"
```

---

### 7.4 Alternatives to Airflow

| Tool | Best For | Complexity | Setup Time |
|------|----------|------------|------------|
| **GitHub Actions** | Simple scheduled jobs | Low | ✅ Already have |
| **Apache Airflow** | Complex workflows, enterprise | High | 1-2 days |
| **Prefect** | Modern, code-first | Medium | 1 day |
| **Cron Jobs** | Basic scheduling | Very Low | 1 hour |
| **Kubeflow Pipelines** | K8s-native ML | Very High | 3-5 days |

#### Alternative 1: GitHub Actions Scheduled (Recommended for Now)

```yaml
# .github/workflows/scheduled-retraining.yml
name: Scheduled Model Retraining

on:
  schedule:
    - cron: '0 2 * * 0'  # 2 AM every Sunday
  workflow_dispatch:

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Train models
        run: python src/training.py

      - name: Notify on failure
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: '⚠️ Model Retraining Failed',
              body: 'Check the [workflow logs](' + context.server_url + '/' + context.repo.owner + '/' + context.repo.repo + '/actions/runs/' + context.runId + ') for details.'
            })
```

#### Alternative 2: Prefect (Modern, Simpler)

```python
# flows/retraining_flow.py
from prefect import flow, task
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import IntervalSchedule
from datetime import timedelta

@task
def fetch_data():
    """Fetch latest data"""
    # Your logic here
    pass

@task
def train_models():
    """Train models"""
    from src.training import run_training_pipeline
    return run_training_pipeline()

@task
def evaluate_and_deploy(results):
    """Evaluate and deploy if better"""
    # Your logic here
    pass

@flow(name="diabetes-model-retraining")
def retraining_flow():
    data = fetch_data()
    models = train_models()
    deploy = evaluate_and_deploy(models)
    return deploy

# Deploy with weekly schedule
Deployment.build_from_flow(
    flow=retraining_flow,
    name="weekly-model-retraining",
    schedule=IntervalSchedule(timedelta(hours=168)),  # Weekly
    work_queue_name="ml-pipeline",
)
```

---

### 7.5 Decision Framework: When to Use What

```python
# Decision tree for choosing orchestration tool

def choose_orchestration_tool(requirements):
    """
    Helper function to decide which orchestration tool to use
    """

    # Criteria 1: Scale and complexity
    if requirements['num_workflows'] < 5 and requirements['dependencies'] == 'simple':
        if requirements['already_uses_gh_actions']:
            return 'GitHub Actions'
        else:
            return 'Cron Jobs'

    # Criteria 2: Team expertise and infrastructure
    elif requirements['has_k8s'] and requirements['team_knows_k8s']:
        return 'Kubeflow Pipelines'

    # Criteria 3: Need for advanced features
    elif requirements['needs']:
        if 'complex_dependencies' in requirements['needs']:
            if requirements['prefers_modern']:
                return 'Prefect'
            else:
                return 'Apache Airflow'

        elif 'real_time_monitoring' in requirements['needs']:
            return 'Apache Airflow'

        elif 'ui_required' in requirements['needs']:
            return 'Apache Airflow'

    # Criteria 4: Enterprise requirements
    elif requirements['is_enterprise']:
        if requirements['cloud_provider'] == 'gcp':
            return 'Google Cloud Composer'
        elif requirements['cloud_provider'] == 'aws':
            return 'Amazon MWAA'
        elif requirements['cloud_provider'] == 'azure':
            return 'Azure Data Factory'

    # Default
    return 'Apache Airflow'  # Most flexible

# Example usage for this project
current_requirements = {
    'num_workflows': 3,  # retraining, batch prediction, monitoring
    'dependencies': 'medium',
    'already_uses_gh_actions': True,
    'has_k8s': False,
    'team_knows_k8s': False,
    'prefers_modern': False,
    'needs': ['ui_required', 'monitoring', 'alerting'],
    'is_enterprise': False,
    'cloud_provider': None
}

recommendation = choose_orchestration_tool(current_requirements)
print(f"Recommended: {recommendation}")
# Output: Recommended: Apache Airflow (or GitHub Actions for simplicity)
```

---

### 7.6 Migration Path

#### Phase 1: Stay with GitHub Actions (Current - ✅ Recommended)
```yaml
# Simple scheduled workflows
# - Weekly model retraining
# - Daily batch predictions
# - Daily monitoring checks
```
**Timeframe:** Now
**Complexity:** Low
**Cost:** Free

#### Phase 2: Add Prefect (When workflows get complex)
```python
# Code-first, modern alternative
# - Better developer experience
# - Native Python
# - Less infrastructure
```
**Timeframe:** When > 5 workflows
**Complexity:** Medium
**Cost:** Free tier available

#### Phase 3: Migrate to Airflow (Enterprise scale)
```python
# When you need:
# - Advanced DAG dependencies
# - Enterprise features
# - Complex SLA requirements
# - Team already knows Airflow
```
**Timeframe:** When production-scale with many workflows
**Complexity:** High
**Cost:** $$$$ (infrastructure, managed services)

---

### 7.7 Best Practices

#### DO ✅:
1. **Start Simple** - GitHub Actions scheduled workflows
2. **Version Control DAGs** - Keep in git repo
3. **Test Locally** - Use `airflow dags test` before deploying
4. **Monitor DAGs** - Set up alerts for failures
5. **Document Dependencies** - Clear DAG structure
6. **Use XCom** sparingly - Only for small data
7. **Idempotent Tasks** - Tasks should be retryable
8. **Timeouts** - Set reasonable timeouts
9. **Resource Limits** - Set memory/CPU limits
10. **Log Everything** - Structured logs

#### DON'T ❌:
1. **Don't Over-Orchestrate** - Not everything needs Airflow
2. **Don't Hardcode Paths** - Use environment variables
3. **Don't Ignore Errors** - Handle failures gracefully
4. **Don't Create Mega DAGs** - Break into smaller, focused DAGs
5. **Don't Run as Root** - Security risk
6. **Don't Store Secrets in DAGs** - Use Airflow connections
7. **Don't Forget Idempotency** - Tasks should be safe to retry
8. **Don't Overload Workers** - Scale appropriately
9. **Don't Skip Testing** - Test DAGs before deploying
10. **Don't Ignore Deprecated Features** - Keep Airflow updated

---

### 7.8 Monitoring and Maintenance

#### Airflow Health Checks

```python
# dags/airflow_health_check.py
def check_airflow_health():
    """Monitor Airflow instance health"""
    from airflow.api.client.local_client import Client

    client = Client()

    # Check for paused DAGs
    dags = client.list_dags()
    paused_dags = [dag for dag in dags if dag.is_paused]

    # Check for failed tasks
    from airflow.models import DagRun
    failed_runs = DagRun.find(dag_id='diabetes_model_retraining', state='failed')

    # Check worker queue
    from airflow.utils.state import State
    queued_tasks = State.queued

    return {
        'paused_dags': len(paused_dags),
        'failed_runs': len(failed_runs),
        'queued_tasks': len(queued_tasks)
    }
```

---

### Summary

| Priority | Status | Effort | Time to Implement |
|----------|--------|--------|-------------------|
| Priority 7.1: Airflow Integration | ⏳ Future | High | 1-2 weeks |
| Priority 7.2: Docker Compose Setup | ⏳ Future | Medium | 2-3 days |
| Priority 7.3: DAG Development | ⏳ Future | High | 1 week |
| Priority 7.4: Alternatives Setup | ⏳ Future | Low | 1 day |

**Recommended Starting Point:** GitHub Actions scheduled workflows (simpler)

**Migration Trigger:** When you have >5 complex workflows with dependencies

**Expected Time:** 1-2 weeks for full Airflow setup

---

**Last Updated:** 2026-02-07

**Maintainer:** Claude Code

**Status:** 📝 Planning Phase
