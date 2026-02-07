#!/bin/bash

# Script untuk rebuild MLflow container dengan konfigurasi baru
# Ini akan mempercepat startup karena MLflow di-install saat build, bukan runtime

set -e

echo "================================"
echo "Rebuilding MLflow Container"
echo "================================"
echo ""

# Stop existing MLflow container
echo "[1/4] Stopping existing MLflow container..."
docker-compose stop mlflow 2>/dev/null || echo "No existing container to stop"
docker-compose rm -f mlflow 2>/dev/null || echo "No container to remove"

# Remove old image (optional - uncomment if you want to rebuild from scratch)
# echo "[2/4] Removing old image..."
# docker rmi diabetes-prediction-mlflow 2>/dev/null || echo "No old image to remove"

echo ""
echo "[2/4] Building new MLflow image with MLflow pre-installed..."
docker-compose build mlflow

echo ""
echo "[3/4] Starting MLflow container..."
docker-compose up -d mlflow

echo ""
echo "[4/4] Waiting for MLflow to be ready..."
sleep 5

# Check if MLflow is running
if docker ps | grep -q diabetes-mlflow; then
    echo ""
    echo "✅ MLflow container is running!"
    echo ""
    echo "Checking health status..."
    sleep 3

    # Try to access health endpoint
    if curl -s http://localhost:5001/health > /dev/null 2>&1; then
        echo "✅ MLflow is accessible at http://localhost:5001"
    else
        echo "⏳ MLflow is still starting up..."
        echo "   Check logs: docker logs -f diabetes-mlflow"
    fi

    echo ""
    echo "View logs:"
    echo "  docker logs -f diabetes-mlflow"
    echo ""
    echo "Open MLflow UI:"
    echo "  http://localhost:5001"
else
    echo ""
    echo "❌ Failed to start MLflow container"
    echo "   Check logs: docker logs diabetes-mlflow"
    exit 1
fi
