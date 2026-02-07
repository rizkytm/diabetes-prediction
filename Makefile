.PHONY: help install train test clean docker-build docker-up docker-down docker-logs api streamlit mlflow mlflow-rebuild

# Default target
help:
	@echo "Diabetes Prediction MLOps Project - Makefile Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install     - Install dependencies"
	@echo "  make train       - Train models"
	@echo ""
	@echo "Development:"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run code quality checks"
	@echo "  make format      - Format code with black and isort"
	@echo ""
	@echo "Local Deployment:"
	@echo "  make streamlit   - Run Streamlit app (http://localhost:8501)"
	@echo "  make api         - Run FastAPI server (http://localhost:8000)"
	@echo "  make mlflow      - Run MLflow dashboard (http://localhost:5001)"
	@echo ""
	@echo "Docker Deployment:"
	@echo "  make docker-build      - Build Docker images"
	@echo "  make docker-up         - Start all services with docker-compose"
	@echo "  make docker-down       - Stop all services"
	@echo "  make docker-logs       - View logs from all services"
	@echo "  make docker-clean      - Remove all containers and volumes"
	@echo "  make mlflow-rebuild    - Rebuild MLflow container (faster startup)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean       - Clean temporary files"
	@echo "  make backup      - Backup models directory"

# Installation
install:
	@echo "[INFO] Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "[OK] Dependencies installed"

# Training
train:
	@echo "[INFO] Training models..."
	python src/training.py
	@echo "[OK] Training complete"

# Testing
test:
	@echo "[INFO] Running tests..."
	if [ -d "tests" ]; then \
		pytest tests/ -v --cov=src; \
	else \
		echo "[WARN] No tests found. Create tests/ directory first."; \
	fi

# Linting
lint:
	@echo "[INFO] Running code quality checks..."
	flake8 src/ api.py app.py --max-line-length=100 --extend-ignore=E203,W503
	pylint src/ api.py app.py --disable=C0111,R0913,R0914,C0103,R0903,W1309,E0213,R1716,R0912,R0915,R0917,W0102,C0415,R0801,W0707,E0401,W0603 --fail-under=7.5

# Formatting
format:
	@echo "[INFO] Formatting code..."
	black src/ api.py app.py
	isort src/ api.py app.py
	@echo "[OK] Code formatted"

# Clean temporary files
clean:
	@echo "[INFO] Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	@echo "[OK] Cleanup complete"

# Backup models
backup:
	@echo "[INFO] Backing up models..."
	mkdir -p backups
	cp -r models/ backups/models-$$(date +%Y%m%d-%H%M%S)/
	@echo "[OK] Models backed up to backups/"

# Streamlit app
streamlit:
	@echo "[INFO] Starting Streamlit app..."
	streamlit run app.py

# FastAPI server
api:
	@echo "[INFO] Starting FastAPI server..."
	uvicorn api:app --reload --host 0.0.0.0 --port 8000

# MLflow dashboard
mlflow:
	@echo "[INFO] Starting MLflow dashboard..."
	mlflow ui

# Docker build
docker-build:
	@echo "[INFO] Building Docker images..."
	docker-compose build
	@echo "[OK] Docker images built"

# Docker up
docker-up:
	@echo "[INFO] Starting Docker services..."
	docker-compose up -d
	@echo "[OK] Services started:"
	@echo "  - Streamlit: http://localhost:8501"
	@echo "  - API: http://localhost:8000"

# Docker down
docker-down:
	@echo "[INFO] Stopping Docker services..."
	docker-compose down
	@echo "[OK] Services stopped"

# Docker logs
docker-logs:
	docker-compose logs -f

# Docker clean
docker-clean:
	@echo "[INFO] Cleaning Docker resources..."
	docker-compose down -v
	docker system prune -f
	@echo "[OK] Docker resources cleaned"

# Rebuild MLflow container
mlflow-rebuild:
	@echo "[INFO] Rebuilding MLflow container with pre-installed MLflow..."
	./rebuild-mlflow.sh

# Run all checks
check: lint test
	@echo "[OK] All checks passed"
