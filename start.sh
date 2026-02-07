#!/bin/bash

# Diabetes Prediction - Quick Start Script
# This script helps you get started quickly with the project

set -e

echo "================================"
echo "Diabetes Prediction MLOps Project"
echo "================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python -m venv venv
    echo "[OK] Virtual environment created"
else
    echo "[1/5] Virtual environment already exists"
fi

echo ""

# Activate virtual environment
echo "[2/5] Activating virtual environment..."
source venv/bin/activate
echo "[OK] Virtual environment activated"

echo ""

# Install dependencies
echo "[3/5] Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "[OK] Dependencies installed"

echo ""

# Check if models exist
if [ ! -f "models/best_model.pkl" ]; then
    echo "[4/5] Models not found. Training models..."
    python src/training.py
    echo "[OK] Models trained and saved"
else
    echo "[4/5] Models already exist"
fi

echo ""

# Ask what to run
echo "================================"
echo "What would you like to start?"
echo "================================"
echo "1) Streamlit Web App (http://localhost:8501)"
echo "2) FastAPI REST API (http://localhost:8000)"
echo "3) Docker Compose (both services)"
echo "4) MLflow Dashboard (http://localhost:5000)"
echo "5) Exit"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "[5/5] Starting Streamlit app..."
        streamlit run app.py
        ;;
    2)
        echo ""
        echo "[5/5] Starting FastAPI server..."
        uvicorn api:app --reload --host 0.0.0.0 --port 8000
        ;;
    3)
        echo ""
        echo "[5/5] Starting Docker Compose..."
        docker-compose up --build
        ;;
    4)
        echo ""
        echo "[5/5] Starting MLflow dashboard..."
        mlflow ui
        ;;
    5)
        echo ""
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo ""
        echo "[ERROR] Invalid choice. Exiting."
        exit 1
        ;;
esac
