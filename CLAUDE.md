# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end MLOps project for diabetes prediction using the Pima Indians Diabetes Dataset. Implements Scikit-learn pipelines for data preprocessing and MLflow for experiment tracking.

**Current State**: ✅ **COMPLETE & PRODUCTION-READY**

All components have been implemented, tested, and deployed successfully:
- ✅ Preprocessing pipeline with custom ZeroToNanTransformer
- ✅ Model training with MLflow tracking (Logistic Regression & Random Forest)
- ✅ Streamlit web application for predictions
- ✅ Complete EDA and documentation

## Development Approach

**Hybrid Development Strategy:**
1. **Exploration Phase**: Used Jupyter notebooks for EDA and preprocessing development
2. **Production Refactor**: Converted working code to modular production modules in `src/`
3. **Benefits**: Rapid experimentation with maintainable, reusable production code

**Key Decision:** Deleted `notebooks/model_experiments.ipynb` after refactoring to prevent confusion. Use `src/training.py` for all model training.

## Common Commands

### Environment Setup
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Model Training and MLflow Tracking
```bash
python src/training.py          # Train model and log to MLflow
mlflow ui                        # Start MLflow dashboard at http://localhost:5000
```

### Running the Web Application
```bash
streamlit run app.py             # Launch Streamlit dashboard
```

## Architecture

### Data Flow Pipeline

Raw Data (`data/diabetes.csv`) → Processing Pipeline (`src/processing.py`) → Model Training (`src/training.py`) → Inference (`app.py`)

**Critical**: The preprocessing pipeline must be saved and reused during inference to ensure consistency. The pipeline handles invalid zeros in medical data (Glucose, BloodPressure, SkinThickness, Insulin, BMI) by converting them to NaN, then applying median imputation and standard scaling.

### Key Components

**`src/processing.py`**
- Defines preprocessing pipeline with custom transformer for medical zeros
- Returns sklearn.pipeline.Pipeline object
- Pattern: Custom transformer → SimpleImputer(median) → StandardScaler

**`src/training.py`**
- Loads data, applies preprocessing pipeline, trains classifier
- MLflow logging: parameters, metrics (Accuracy, Recall, F1-Score), model artifacts
- Saves model to `models/` directory

**`app.py`**
- Streamlit UI for predictions
- Loads saved preprocessing pipeline AND model
- Takes 8 feature inputs, returns prediction with probability

## MLflow Configuration

- **Storage**: `MLruns/` directory (file-based)
- **Log per run**: parameters, metrics (Accuracy, Recall, F1-Score), model artifacts
- **Artifacts**: classifier + preprocessing pipeline (save both!)

## Dataset Details

**Pima Indians Diabetes Dataset**: 768 rows, 9 columns
- **Features**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target**: Outcome (0=No Diabetes, 1=Diabetes)
- **Class imbalance**: 65.1% negative, 34.9% positive

**Critical Data Quality Issues** (must handle in preprocessing):
- Glucose: 5 zeros (invalid)
- BloodPressure: 35 zeros (invalid)
- SkinThickness: 227 zeros (missing)
- Insulin: 374 zeros (missing)
- BMI: 11 zeros (invalid)

## MLOps Considerations

**Metric Priority**: For medical diagnosis, **Recall is critical** (minimize false negatives - missed diabetes cases). Accuracy alone is insufficient due to class imbalance.

**Reproducibility**:
- Always use `random_state` parameter
- Log ALL hyperparameters to MLflow
- Save complete preprocessing pipeline with model
- Never train without MLflow logging enabled

## Environment

- **Python**: 3.9+ (currently 3.12.10)
- **Platform**: macOS (Apple Silicon)
- **Dependencies**: pandas, numpy, scikit-learn, mlflow, streamlit

## Important Notes

**Encoding Policy:**
- All Python files use UTF-8 encoding with `# -*- coding: utf-8 -*-` header
- **CRITICAL**: Use ASCII-only characters in production code (no emoji)
- Replace emoji with text markers: [OK], [INFO], [TRAINING], [!], etc.
- Reason: Prevents `SyntaxError: Non-UTF-8 code` errors during execution
- Affected files: `src/training.py`, `app.py`

**Import Best Practices:**
- Use `from src.processing import ...` for importing preprocessing components
- Use `from src.training import ...` for importing training functions
- This ensures proper module resolution and avoids ZeroToNanTransformer import errors

## Implementation Status

**✅ COMPLETED COMPONENTS:**

**Core Modules:**
- ✅ `requirements.txt` - All dependencies specified with M1-compatible versions
- ✅ `src/processing.py` - Custom ZeroToNanTransformer and preprocessing pipeline
- ✅ `src/training.py` - Model training with MLflow tracking (Logistic Regression & Random Forest)
- ✅ `src/__init__.py` - Package exports for clean imports
- ✅ `app.py` - Streamlit web application with probability visualization

**Notebooks (Exploration & Development):**
- ✅ `notebooks/eda.ipynb` - Complete EDA with detailed interpretations
- ✅ `notebooks/preprocessing_exploration.ipynb` - Preprocessing pipeline development
- ❌ `notebooks/model_experiments.ipynb` - **DELETED** (obsolete, use src/training.py)

**Trained Models (saved in `models/`):**
- ✅ `preprocessing_pipeline.pkl` - Fitted preprocessing pipeline
- ✅ `logistic_regression_model.pkl` - Trained Logistic Regression model
- ✅ `random_forest_model.pkl` - Trained Random Forest model
- ✅ `best_model.pkl` - Best model selected by test recall metric

**Documentation:**
- ✅ `README.md` - Complete project documentation
- ✅ `CLAUDE.md` - This file (guidance for Claude Code)

**READY FOR:**
- Model deployment (Streamlit app already working)
- Containerization (Docker)
- CI/CD pipeline integration
- API deployment (FastAPI/Flask wrapper)
