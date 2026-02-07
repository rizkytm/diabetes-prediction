# -*- coding: utf-8 -*-
"""
Diabetes Prediction Web Application

Streamlit app for diabetes prediction using trained ML models.

Author: Claude Code
Date: 2026-02-07
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon=":",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-positive {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #ff6b6b;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .result-negative {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #51cf66;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .probability-bar {
        height: 3rem;
        border-radius: 0.5rem;
        background: linear-gradient(90deg, #51cf66 0%, #ffd43b 50%, #ff6b6b 100%);
    }
</style>
""", unsafe_allow_html=True)


# ==================== LOAD MODELS ====================

@st.cache_resource
def load_models():
    """
    Load preprocessing pipeline and trained model.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Preprocessing pipeline
    model : sklearn estimator
        Trained model
    """
    try:
        # Load preprocessing pipeline
        pipeline_path = 'models/preprocessing_pipeline.pkl'
        model_path = 'models/best_model.pkl'

        if not os.path.exists(pipeline_path):
            st.error("Preprocessing pipeline not found. Please run training first.")
            st.stop()

        if not os.path.exists(model_path):
            st.error("Model not found. Please run training first.")
            st.stop()

        pipeline = joblib.load(pipeline_path)
        model = joblib.load(model_path)

        return pipeline, model

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()


# ==================== PREDICTION FUNCTION ====================

def predict_diabetes(pipeline, model, input_data):
    """
    Make prediction on input data.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Preprocessing pipeline
    model : sklearn estimator
        Trained model
    input_data : dict
        Dictionary of feature values

    Returns
    -------
    prediction : int
        Predicted class (0 or 1)
    probability : float
        Prediction probability
    """
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess
    input_processed = pipeline.transform(input_df)

    # Predict
    prediction = model.predict(input_processed)[0]

    # Get probability
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(input_processed)[0, 1]
    else:
        probability = float(prediction)

    return prediction, probability


# ==================== MAIN APP ====================

def main():
    """Main application."""

    # Header
    st.markdown('<h1 class="main-header">[Diabetes] Prediction</h1>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Load models
    pipeline, model = load_models()

    # Sidebar - Model Info
    with st.sidebar:
        st.header("[Model Information]")

        # Get model type
        model_type = type(model).__name__
        st.info(f"**Model:** {model_type}")

        st.markdown("""
        **About:**
        This app uses machine learning to predict diabetes risk based on
        medical parameters such as glucose level, BMI, age, etc.

        **Features:**
        - Custom preprocessing pipeline
        - Class imbalance handling
        - Median imputation for missing values
        - Standardized features

        **Target Metric:**
        - Optimized for Recall (minimize false negatives)
        - Ensures early detection of diabetes cases
        """)

        st.markdown("---")
        st.markdown("""
        **Disclaimer:**
        This prediction is for informational purposes only.
        Please consult a healthcare professional for medical advice.
        """)

    # Main content
    st.header("[Patient Information]")

    st.markdown("""
    Please enter the following medical parameters. All values should be
    from recent laboratory tests and physical examinations.
    """)

    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Basic Info")

            pregnancies = st.number_input(
                "Number of Pregnancies",
                min_value=0,
                max_value=20,
                value=1,
                help="Number of times pregnant"
            )

            glucose = st.number_input(
                "Glucose Level (mg/dL)",
                min_value=0,
                max_value=300,
                value=120,
                help="Plasma glucose concentration after 2 hours in OGTT"
            )

            blood_pressure = st.number_input(
                "Blood Pressure (mm Hg)",
                min_value=0,
                max_value=200,
                value=70,
                help="Diastolic blood pressure"
            )

            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0,
                max_value=100,
                value=20,
                help="Triceps skin fold thickness"
            )

        with col2:
            st.subheader("Advanced Parameters")

            insulin = st.number_input(
                "Insulin Level (mu U/ml)",
                min_value=0,
                max_value=1000,
                value=80,
                help="2-Hour serum insulin"
            )

            bmi = st.number_input(
                "BMI (Body Mass Index)",
                min_value=0.0,
                max_value=70.0,
                value=32.0,
                step=0.1,
                help="Body mass index (weight in kg / height in m^2)"
            )

            dpf = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.5,
                step=0.01,
                help="Diabetes genetic risk score"
            )

            age = st.number_input(
                "Age (years)",
                min_value=21,
                max_value=100,
                value=33,
                help="Age in years"
            )

        # Submit button
        submit = st.form_submit_button("[Predict] Diabetes Risk",
                                       use_container_width=True)

    # Prediction
    if submit:
        # Prepare input data
        input_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }

        # Make prediction
        prediction, probability = predict_diabetes(pipeline, model, input_data)

        # Display results
        st.markdown("---")
        st.header("[Prediction Results]")

        # Prediction
        if prediction == 1:
            st.markdown('<div class="result-positive">[!] High Risk of Diabetes</div>',
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-negative">[OK] Low Risk of Diabetes</div>',
                       unsafe_allow_html=True)

        # Probability
        st.subheader("Probability")
        probability_percent = probability * 100

        st.markdown(f"""
        <div class="probability-bar"></div>
        **Risk Level: {probability_percent:.1f}%**
        """, unsafe_allow_html=True)

        # Progress bar
        st.progress(probability)

        # Detailed interpretation
        st.markdown("---")
        st.subheader("[Interpretation]")

        if probability < 0.3:
            interpretation = """
            **Low Risk** [Green]

            Your diabetes risk is relatively low based on the parameters provided.
            Continue maintaining a healthy lifestyle with regular exercise and balanced diet.
            """
        elif probability < 0.5:
            interpretation = """
            **Moderate Risk** [Yellow]

            You have some risk factors for diabetes. Consider:
            - Regular health check-ups
            - Monitoring blood sugar levels
            - Maintaining healthy weight
            - Regular physical activity
            """
        elif probability < 0.7:
            interpretation = """
            **High Risk** [Orange]

            Your diabetes risk is elevated. We strongly recommend:
            - Consult a healthcare professional
            - Get proper diabetes screening
            - Review lifestyle factors
            - Discuss prevention strategies
            """
        else:
            interpretation = """
            **Very High Risk** [Red]

            Your risk profile indicates high likelihood of diabetes.
            **Please consult a healthcare professional as soon as possible**
            for proper evaluation and diagnosis.
            """

        st.info(interpretation)

        # Recommendations
        st.markdown("---")
        st.subheader("[Health Recommendations]")

        recommendations = []

        # Glucose-based recommendations
        if glucose > 140:
            recommendations.append("[!] **Glucose level elevated** - Consider reducing sugar intake")
        elif glucose > 125:
            recommendations.append("[!] **Glucose level above normal** - Monitor blood sugar regularly")

        # BMI-based recommendations
        if bmi >= 30:
            recommendations.append("[!] **BMI indicates obesity** - Weight management recommended")
        elif bmi >= 25:
            recommendations.append("[!] **BMI indicates overweight** - Consider healthy weight loss")

        # Blood pressure recommendations
        if blood_pressure > 90:
            recommendations.append("[!] **Blood pressure elevated** - Regular monitoring advised")

        # Age-based recommendations
        if age > 45:
            recommendations.append("[!] **Age over 45** - Regular diabetes screening recommended")

        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("[OK] All parameters within normal range - Keep it up!")

        for rec in recommendations:
            st.markdown(f"- {rec}")

        # Input summary
        with st.expander("[View] Input Parameters"):
            st.write(pd.DataFrame([input_data]).T.rename(columns={0: "Value"}))

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Diabetes Prediction App</strong></p>
        <p>Built with ML Pipeline</p>
        <p>(c) 2026 - MLOps Project</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
