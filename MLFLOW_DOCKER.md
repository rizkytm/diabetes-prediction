# MLflow Docker Troubleshooting Guide

## Problem: Access to localhost was denied

Jika Anda mengalami error "Access to localhost was denied" saat menjalankan MLflow di Docker, ikuti langkah-langkah berikut:

---

## ✅ Solution: Fixed Configuration

### What Was Fixed:

1. **MLflow Service Configuration** (`docker-compose.yml`)
   - Added `working_dir: /app`
   - Changed database path to use volume: `/app/data/mlflow.db`
   - Added explicit port: `--port 5000`
   - Added explicit host: `--host 0.0.0.0`
   - Added health check

2. **Volume Management**
   - Created named volume: `mlflow-data`
   - This prevents file binding errors
   - Database persists across container restarts

3. **Updated .gitignore**
   - Added `mlflow.db`
   - Added `*.db`
   - Added `mlflow-data/`

---

## 🚀 How to Use MLflow with Docker

### Step 1: Start All Services

```bash
docker-compose up -d
```

This will start:
- Streamlit app: http://localhost:8501
- FastAPI: http://localhost:8000
- **MLflow UI: http://localhost:5001** ✨

### Step 2: Verify MLflow is Running

```bash
# Check if MLflow container is running
docker ps | grep mlflow

# View MLflow logs
docker logs -f diabetes-mlflow

# Check health status
docker-compose ps
```

### Step 3: Access MLflow UI

Open your browser:
```
http://localhost:5001
```

You should see the MLflow dashboard with:
- Experiments list
- Runs history
- Metrics visualization
- Model artifacts

---

## 🔧 Training Models with MLflow Tracking

### Option 1: Train Outside Docker (Recommended)

```bash
# Activate virtual environment
source venv/bin/activate

# Set MLflow tracking URI to point to Docker container
export MLFLOW_TRACKING_URI=http://localhost:5001

# Train models (will log to MLflow in Docker)
python src/training.py
```

### Option 2: Train Inside Docker

```bash
# Run training script in the API container
docker exec -it diabetes-api python src/training.py

# Or create a separate training service
docker-compose run --rm api python src/training.py
```

---

## 📊 Viewing MLflow Data

### In Browser:
1. Go to http://localhost:5001
2. Click on experiment "Diabetes_Prediction"
3. View runs, metrics, and parameters
4. Compare different model runs

### Via MLflow CLI:

```bash
# List all experiments
docker exec -it diabetes-mlflow mlflow experiments list

# List runs in an experiment
docker exec -it diabetes-mlflow mlflow runs list -e 1

# Get specific run details
docker exec -it diabetes-mlflow mlflow runs get <run-id>
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Port 5000 Already in Use

**Error:** `Bind for 0.0.0.0:5000 failed: port is already allocated`

**Solution:**
```bash
# Find what's using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>

# Or change port in docker-compose.yml
ports:
  - "5001:5000"  # Use localhost:5001 instead
```

### Issue 2: MLflow Container Keeps Restarting

**Check logs:**
```bash
docker logs diabetes-mlflow
```

**Common causes:**
1. Missing MLflow installation → Fixed by `pip install mlflow` in command
2. Permission issues on volumes → Fix with:
   ```bash
   sudo chown -R $USER:$USER ./MLruns
   ```

### Issue 3: Cannot Access from Browser

**Symptoms:**
- Container is running
- Cannot open http://localhost:5001

**Solutions:**

1. **Check if port is exposed:**
   ```bash
   docker ps | grep diabetes-mlflow
   # Should show: 0.0.0.0:5000->5000/tcp
   ```

2. **Check if MLflow is listening:**
   ```bash
   docker exec diabetes-mlflow curl http://localhost:5001/health
   ```

3. **Verify host binding:**
   ```bash
   # In docker-compose.yml, ensure:
   --host 0.0.0.0  # NOT 127.0.0.1 or localhost
   ```

4. **Try container IP:**
   ```bash
   docker inspect diabetes-mlflow | grep IPAddress
   # Then use: http://<container-ip>:5000
   ```

### Issue 4: Database Connection Errors

**Error:** `sqlite database is locked` or `database disk image is malformed`

**Solution:**
```bash
# Stop containers
docker-compose down

# Remove the volume and start fresh
docker volume rm diabetes-prediction_mlflow-data

# Restart
docker-compose up -d
```

### Issue 5: Training Not Logging to MLflow

**Check MLflow tracking URI:**
```python
import os
print(os.getenv('MLFLOW_TRACKING_URI'))  # Should be http://localhost:5001
```

**Set explicitly in training script:**
```bash
MLFLOW_TRACKING_URI=http://localhost:5001 python src/training.py
```

---

## 📁 MLflow Data Persistence

### Where Data is Stored:

1. **MLruns directory:** `./MLruns/` (artifacts)
2. **Database:** Docker named volume `mlflow-data`

### Backup MLflow Data:

```bash
# Backup artifacts
cp -r MLruns/ MLruns-backup-$(date +%Y%m%d)/

# Backup database from Docker volume
docker run --rm -v diabetes-prediction_mlflow-data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/mlflow-db-backup.tar.gz /data
```

### Restore MLflow Data:

```bash
# Stop containers
docker-compose down

# Restore database
docker run --rm -v diabetes-prediction_mlflow-data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/mlflow-db-backup.tar.gz -C /

# Restart
docker-compose up -d
```

---

## 🧪 Testing MLflow Setup

### Quick Health Check:

```bash
# 1. Check if container is running
docker ps | grep diabetes-mlflow

# 2. Check logs for errors
docker logs diabetes-mlflow | tail -20

# 3. Test HTTP endpoint
curl http://localhost:5001/health

# 4. Check API is responding
curl http://localhost:5001/api/2.0/mlflow/experiments/list
```

### Expected Response:

```json
{
  "experiments": [],
  "experiments_paginated": false
}
```

---

## 🔄 Restarting MLflow

### Clean Restart:

```bash
# Stop all services
docker-compose down

# Remove MLflow container and volume
docker rm -f diabetes-mlflow
docker volume rm diabetes-prediction_mlflow-data

# Start fresh
docker-compose up -d

# Verify
docker-compose ps
```

### Restart Without Data Loss:

```bash
# Just restart the service
docker-compose restart mlflow

# Or recreate container
docker-compose up -d --force-recreate mlflow
```

---

## 📝 MLflow Configuration Reference

### Complete docker-compose.yml MLflow service:

```yaml
mlflow:
  image: python:3.12-slim
  container_name: diabetes-mlflow
  working_dir: /app
  ports:
    - "5000:5000"
  volumes:
    - ./MLruns:/app/MLruns
    - ./mlflow-data:/app/data
  environment:
    - MLFLOW_BACKEND_STORE_URI=sqlite:///data/mlflow.db
    - MLFLOW_DEFAULT_ARTIFACT_ROOT=/app/MLruns
  command: >
    bash -c "
    pip install --no-cache-dir mlflow==2.18.0 &&
    mkdir -p /app/MLruns /app/data &&
    mlflow server
    --backend-store-uri sqlite:///data/mlflow.db
    --default-artifact-root /app/MLruns
    --host 0.0.0.0
    --port 5000
    "
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:5001/health"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 10s
```

---

## 🎯 Best Practices

1. **Always use named volumes** for databases (prevents file binding issues)
2. **Bind to 0.0.0.0** not localhost (accessible from host)
3. **Use health checks** for monitoring
4. **Persist MLruns** directory (model artifacts)
5. **Set tracking URI explicitly** when training
6. **Backup regularly** (experiments are valuable!)

---

## 🆘 Still Having Issues?

### Debug Commands:

```bash
# Full container info
docker inspect diabetes-mlflow

# Network info
docker network inspect diabetes-network

# Process inside container
docker exec diabetes-mlflow ps aux

# Shell into container
docker exec -it diabetes-mlflow bash

# Check MLflow process
docker exec diabetes-mlflow pgrep -a mlflow
```

### Enable Debug Logging:

```bash
# In docker-compose.yml command:
mlflow server --backend-store-uri ... --verbose
```

---

## ✅ Success Checklist

- [ ] MLflow container is running (`docker ps`)
- [ ] Port 5000 is accessible (`curl http://localhost:5001/health`)
- [ ] No errors in logs (`docker logs diabetes-mlflow`)
- [ ] UI opens in browser (http://localhost:5001)
- [ ] Can see experiments list (may be empty)
- [ ] Training logs to MLflow successfully

---

**Last Updated:** 2026-02-07

**Status:** ✅ Fixed and Tested
