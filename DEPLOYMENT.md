# Deployment Guide - Diabetes Prediction MLOps Project

This guide provides comprehensive instructions for deploying the Diabetes Prediction application using Docker, FastAPI, and CI/CD pipelines.

---

## Table of Contents

1. [Local Deployment](#local-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Docker Compose Deployment](#docker-compose-deployment)
4. [Cloud Deployment Options](#cloud-deployment-options)
5. [CI/CD Pipeline](#cicd-pipeline)
6. [API Usage](#api-usage)
7. [Monitoring & Maintenance](#monitoring--maintenance)

---

## Local Deployment

### Prerequisites

- Python 3.12+
- pip package manager
- Git

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd diabetes-prediction
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train Model

```bash
python src/training.py
```

This will:
- Train Logistic Regression and Random Forest models
- Log experiments to MLflow
- Save models to `models/` directory

### Step 5: Run Streamlit App

```bash
streamlit run app.py
```

Access at: http://localhost:8501

### Step 6: Run FastAPI Server

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Access API docs at: http://localhost:8000/docs

---

## Docker Deployment

### Build Streamlit Docker Image

```bash
docker build -t diabetes-streamlit:latest .
```

### Run Streamlit Container

```bash
docker run -d \
  --name diabetes-streamlit \
  -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  diabetes-streamlit:latest
```

### Build FastAPI Docker Image

```bash
docker build -f Dockerfile.api -t diabetes-api:latest .
```

### Run FastAPI Container

```bash
docker run -d \
  --name diabetes-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  diabetes-api:latest
```

### Docker Commands

```bash
# View logs
docker logs -f diabetes-streamlit
docker logs -f diabetes-api

# Stop containers
docker stop diabetes-streamlit diabetes-api

# Remove containers
docker rm diabetes-streamlit diabetes-api

# View running containers
docker ps
```

---

## Docker Compose Deployment

### Deploy All Services

```bash
docker-compose up -d
```

This will start:
- Streamlit app on port 8501
- FastAPI on port 8000

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f streamlit-app
docker-compose logs -f api
```

### Stop Services

```bash
docker-compose down
```

### Stop and Remove Volumes

```bash
docker-compose down -v
```

### Scale Services (if needed)

```bash
docker-compose up -d --scale api=3
```

---

## Cloud Deployment Options

### Option 1: Render.com

#### Deploy Streamlit App

1. Push code to GitHub
2. Go to [render.com](https://render.com)
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Build Command**: `""` (empty for Streamlit)
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - **Python Version**: 3.12

#### Deploy FastAPI

1. Create new Web Service
2. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`

### Option 2: Railway.app

1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select repository
4. Railway will auto-detect Python settings
5. Add environment variables if needed

### Option 3: AWS ECS

#### Push to Docker Hub

```bash
# Tag images
docker tag diabetes-streamlit:latest <username>/diabetes-streamlit:latest
docker tag diabetes-api:latest <username>/diabetes-api:latest

# Push to Docker Hub
docker push <username>/diabetes-streamlit:latest
docker push <username>/diabetes-api:latest
```

#### Deploy to ECS

1. Create ECS cluster
2. Create task definitions for both services
3. Configure load balancer
4. Deploy services

### Option 4: Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/<project-id>/diabetes-streamlit

# Deploy to Cloud Run
gcloud run deploy diabetes-streamlit \
  --image gcr.io/<project-id>/diabetes-streamlit \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Option 5: Azure Container Instances

```bash
# Create resource group
az group create --name diabetes-rg --location eastus

# Create container
az container create \
  --resource-group diabetes-rg \
  --name diabetes-streamlit \
  --image <username>/diabetes-streamlit:latest \
  --ports 8501 \
  --cpu 1 \
  --memory 2
```

---

## CI/CD Pipeline

### GitHub Actions Setup

The project includes a GitHub Actions workflow (`.github/workflows/ml-pipeline.yml`) that:

1. **Code Quality Checks**
   - Black (code formatting)
   - isort (import sorting)
   - Flake8 (linting)
   - Pylint (code quality)

2. **Unit Tests**
   - Runs pytest with coverage
   - Uploads coverage to Codecov

3. **Docker Build**
   - Builds Docker images
   - Pushes to Docker Hub

4. **Security Scan**
   - Trivy vulnerability scanner
   - Uploads results to GitHub Security

5. **Deployment**
   - Automated deployment to production
   - Configure deployment target in workflow

### Required Secrets

Add these secrets to your GitHub repository:

```
DOCKER_USERNAME      # Docker Hub username
DOCKER_PASSWORD      # Docker Hub password/access token
RENDER_SERVICE_ID    # Render service ID (optional)
RENDER_API_KEY       # Render API key (optional)
```

### Enable GitHub Actions

1. Go to repository Settings → Secrets and variables → Actions
2. Add required secrets
3. Push to main branch to trigger workflow

---

## API Usage

### Endpoints

#### 1. Root Endpoint

```bash
GET http://localhost:8000/
```

**Response:**
```json
{
  "message": "Diabetes Prediction API",
  "version": "1.0.0",
  "status": "running"
}
```

#### 2. Health Check

```bash
GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "api_version": "1.0.0"
}
```

#### 3. Single Prediction

```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "pregnancies": 1,
  "glucose": 120,
  "blood_pressure": 70,
  "skin_thickness": 20,
  "insulin": 80,
  "bmi": 32.0,
  "diabetes_pedigree_function": 0.5,
  "age": 33
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.23,
  "risk_level": "Low Risk",
  "confidence": "High",
  "model_used": "LogisticRegression",
  "recommendations": [
    "[OK] All parameters within normal range - Keep it up!"
  ]
}
```

#### 4. Batch Prediction

```bash
POST http://localhost:8000/predict/batch
Content-Type: application/json

{
  "patients": [
    {
      "pregnancies": 1,
      "glucose": 120,
      "blood_pressure": 70,
      "skin_thickness": 20,
      "insulin": 80,
      "bmi": 32.0,
      "diabetes_pedigree_function": 0.5,
      "age": 33
    }
  ]
}
```

#### 5. Model Information

```bash
GET http://localhost:8000/model-info
```

### Using cURL

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 1,
    "glucose": 120,
    "blood_pressure": 70,
    "skin_thickness": 20,
    "insulin": 80,
    "bmi": 32.0,
    "diabetes_pedigree_function": 0.5,
    "age": 33
  }'
```

### Using Python Requests

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "pregnancies": 1,
    "glucose": 120,
    "blood_pressure": 70,
    "skin_thickness": 20,
    "insulin": 80,
    "bmi": 32.0,
    "diabetes_pedigree_function": 0.5,
    "age": 33
}

response = requests.post(url, json=data)
result = response.json()
print(result)
```

### Interactive API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Monitoring & Maintenance

### Health Checks

```bash
# Check if services are running
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
```

### View Logs

```bash
# Docker logs
docker logs -f diabetes-api
docker logs -f diabetes-streamlit

# Docker Compose logs
docker-compose logs -f
```

### Monitor Resources

```bash
# Container stats
docker stats diabetes-api diabetes-streamlit
```

### Update Deployment

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

### Backup Models

```bash
# Backup trained models
cp -r models/ models-backup-$(date +%Y%m%d)/
```

### Retrain Model

```bash
# Activate virtual environment
source venv/bin/activate

# Retrain with new data
python src/training.py --data data/new-dataset.csv

# Restart services to use new model
docker-compose restart
```

---

## Security Considerations

1. **Environment Variables**: Never commit `.env` files
2. **API Keys**: Use GitHub Secrets for CI/CD
3. **CORS**: Configure allowed origins in production
4. **Rate Limiting**: Implement rate limiting for API
5. **HTTPS**: Enable SSL/TLS in production
6. **Input Validation**: All inputs validated via Pydantic

---

## Troubleshooting

### Issue: Models not found

**Solution:**
```bash
# Train models first
python src/training.py

# Check models exist
ls -la models/
```

### Issue: Port already in use

**Solution:**
```bash
# Find process using port
lsof -i :8501
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Issue: Docker build fails

**Solution:**
```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

### Issue: API returns 503 Service Unavailable

**Solution:**
```bash
# Check if models are loaded
docker logs diabetes-api

# Verify models directory is mounted
docker inspect diabetes-api | grep Mounts -A 20
```

---

## Performance Optimization

### Docker Optimization

```bash
# Use multi-stage builds (already implemented)
# Minimize image size (using slim variants)
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1
docker build -t diabetes-streamlit .
```

### API Performance

- Use connection pooling for database
- Implement caching with Redis
- Use async endpoints for I/O operations
- Enable gzip compression

---

## Support & Documentation

- **GitHub Issues**: Report bugs and feature requests
- **API Documentation**: http://localhost:8000/docs
- **Streamlit App**: http://localhost:8501
- **MLflow Dashboard**: http://localhost:5001 (if enabled)

---

## Quick Start Checklist

- [ ] Clone repository
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Train models (`python src/training.py`)
- [ ] Test locally (`streamlit run app.py`)
- [ ] Test API (`uvicorn api:app --reload`)
- [ ] Build Docker images (`docker-compose build`)
- [ ] Run with Docker Compose (`docker-compose up -d`)
- [ ] Set up CI/CD (add GitHub secrets)
- [ ] Deploy to cloud (choose platform)
- [ ] Configure domain and SSL
- [ ] Set up monitoring and alerts

---

**Deployment Status**: Production Ready

**Last Updated**: 2026-02-07
