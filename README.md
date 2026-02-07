## 📊 Diabetes Prediction MLOps Project

**Status:** ✅ **COMPLETE & PRODUCTION-READY**

Proyek ini adalah implementasi *end-to-end* Machine Learning untuk memprediksi risiko diabetes menggunakan **Pima Indians Diabetes Dataset**. Fokus utama proyek ini bukan hanya akurasi model, tetapi juga implementasi **MLOps** menggunakan **Scikit-learn Pipelines** untuk prapemrosesan data dan **MLflow** untuk *experiment tracking*.

### 🚀 Fitur Utama

* **Modular Codebase:** Pemisahan antara skrip prapemrosesan, pelatihan, dan inferensi.
* **Robust Pipeline:** Penanganan *missing values* (nilai 0 yang tidak masuk akal) dan *feature scaling* dalam satu alur pipeline.
* **Experiment Tracking:** Pencatatan parameter, metrik (Accuracy, Recall, F1-Score), dan artefak model menggunakan **MLflow**.
* **Interactive UI:** Dashboard prediksi berbasis **Streamlit**.
* **REST API:** FastAPI endpoint untuk integrasi dengan sistem lain.
* **Docker Support:** Containerization untuk deployment yang mudah.
* **CI/CD Pipeline:** GitHub Actions untuk automated testing dan deployment.
* **M1 Optimized:** Konfigurasi yang ringan dan kompatibel untuk Apple Silicon.

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Library ML:** Scikit-learn, Pandas, NumPy
* **MLOps:** MLflow
* **Frontend:** Streamlit
* **API:** FastAPI, Uvicorn
* **Containerization:** Docker, Docker Compose
* **CI/CD:** GitHub Actions
* **Environment:** Virtualenv / Conda

---

## 📂 Struktur Direktori

```text
.
├── data/                          # Dataset mentah (diabetes.csv)
├── models/                        # Artefak model yang diekspor (pickle/joblib)
│   ├── preprocessing_pipeline.pkl
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── best_model.pkl
├── notebooks/                     # Jupyter Notebook untuk EDA dan eksplorasi
│   ├── eda.ipynb                 # Exploratory Data Analysis lengkap
│   └── preprocessing_exploration.ipynb  # Pengembangan preprocessing pipeline
├── src/                          # Source code utama (production-ready)
│   ├── __init__.py               # Package exports
│   ├── schemas.py                # API request/response schemas
│   ├── training.py               # Skrip pelatihan model & MLflow logging
│   └── processing.py             # Definisi pipeline prapemrosesan
├── .github/workflows/             # CI/CD pipelines
│   └── ml-pipeline.yml           # GitHub Actions workflow
├── app.py                        # Aplikasi Streamlit (deployed)
├── api.py                        # FastAPI REST API
├── Dockerfile                    # Docker image untuk Streamlit
├── Dockerfile.api                # Docker image untuk FastAPI
├── docker-compose.yml            # Multi-service orchestration
├── requirements.txt               # Daftar dependensi (M1-compatible)
├── Makefile                      # Utility commands
├── start.sh                      # Quick start script
├── README.md                      # Dokumentasi proyek
├── CLAUDE.md                      # Panduan untuk Claude Code
├── DEPLOYMENT.md                 # Deployment guide
└── MLruns/                       # Direktori otomatis MLflow (gitignore)

```

---

## ⚙️ Instalasi & Persiapan

1. **Clone repositori:**
```bash
git clone https://github.com/username/diabetes-mlops.git
cd diabetes-mlops

```


2. **Buat Virtual Environment (Rekomendasi untuk M1):**
```bash
python -m venv venv
source venv/bin/activate

```


3. **Instal Dependensi:**
```bash
pip install -r requirements.txt

```



---

## 📈 Alur Kerja (Workflow)

### 1. Pelatihan Model & Tracking

Jalankan skrip training untuk memproses data dan melatih model. MLflow akan mencatat setiap percobaan secara otomatis.

```bash
python src/training.py

```

Untuk melihat dashboard MLflow, jalankan:

```bash
mlflow ui
```

Buka `http://localhost:5001` di browser Anda.

**Note:** Gunakan port 5001 (bukan 5000) karena port 5000 digunakan oleh macOS AirPlay service.

### 2. Menjalankan Dashboard Prediksi

Setelah mendapatkan model terbaik, jalankan aplikasi Streamlit untuk melakukan prediksi interaktif:

```bash
streamlit run app.py
```

Buka `http://localhost:8501` di browser Anda.

### 3. Menjalankan FastAPI REST API

Untuk integrasi dengan sistem lain, jalankan FastAPI server:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Buka `http://localhost:8000/docs` untuk melihat dokumentasi API interaktif.

**Quick Test:**
```bash
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

---

## 🐳 Docker Deployment

### Menggunakan Docker Compose (Rekomendasi)

Jalankan semua services (Streamlit + FastAPI) dengan satu command:

```bash
docker-compose up -d
```

**Services akan tersedia di:**
- Streamlit: http://localhost:8501
- FastAPI: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5001

**Stop services:**
```bash
docker-compose down
```

**View logs:**
```bash
docker-compose logs -f
```

### Menggunakan Makefile (Alternative)

```bash
make docker-build    # Build images
make docker-up       # Start services
make docker-down     # Stop services
make docker-logs     # View logs
```

### Quick Start Script

Untuk kemudahan, gunakan quick start script:

```bash
./start.sh
```

Script ini akan otomatis:
1. Membuat virtual environment
2. Install dependencies
3. Train models (jika belum ada)
4. Tanya service apa yang ingin dijalankan

---

## 📝 Catatan Prapemrosesan

Dataset ini memiliki nilai `0` pada kolom seperti `BloodPressure` dan `BMI` yang secara medis tidak valid. Pipeline dalam proyek ini menangani hal tersebut dengan:

1. Mengubah nilai `0` menjadi `NaN`.
2. Melakukan **SimpleImputer** (Median) untuk mengisi nilai yang hilang.
3. Melakukan **StandardScaler** agar model linear/SVM dapat bekerja optimal.

---

## 🚀 CI/CD Pipeline

Proyek ini sudah dilengkapi dengan GitHub Actions workflow untuk automated testing dan deployment.

### Fitur CI/CD:

1. **Code Quality Checks**
   - Black (code formatting)
   - isort (import sorting)
   - Flake8 (linting)
   - Pylint (code quality)

2. **Automated Testing**
   - Unit tests dengan pytest
   - Coverage reports

3. **Docker Build & Push**
   - Automated Docker image building
   - Push ke Docker Hub

4. **Security Scanning**
   - Trivy vulnerability scanner
   - Upload ke GitHub Security tab

5. **Automated Deployment**
   - Deploy ke production saat merge ke main branch

### Setup:

1. Push code ke GitHub
2. Add secrets di repository settings:
   - `DOCKER_USERNAME`
   - `DOCKER_PASSWORD`
3. Workflow otomatis berjalan saat push/PR

Lihat detail di `.github/workflows/ml-pipeline.yml`

---

## 🗺️ Roadmap

Untuk melihat rencana pengembangan ke depan dan prioritas yang belum diimplementasikan, lihat:

**[📋 ROADMAP.md](ROADMAP.md)** - Berisi:
- Priority 1: Model Improvements (Hyperparameter Tuning, SMOTE, Ensemble, XGBoost)
- Priority 3: Monitoring & Maintenance (Model Monitoring, Logging System)
- Priority 4: Testing (Unit Tests, Integration Tests)
- Priority 5: Advanced Features (Feature Engineering, Explainability, A/B Testing)

---

## 🎨 Code Quality & Auto-Formatting

Untuk mencegah CI/CD failure karena formatting, baca panduan:

**[📝 CODE_QUALITY.md](CODE_QUALITY.md)** - Berisi:
- Cara auto-format code dengan Black & isort
- Setup pre-commit hooks (automatic formatting sebelum commit)
- Command references dan troubleshooting
- Best practices untuk code quality

**Quick Fix:**
```bash
make format  # Format all code
```

---

## 🤝 Kontribusi

Jika Anda ingin mengembangkan proyek ini lebih lanjut, silakan ajukan *pull request* atau pilih item dari [ROADMAP.md](ROADMAP.md) untuk diimplementasikan.

**Sebelum commit/push:**
```bash
make format  # Format code
make lint    # Check code quality
```

---

## 📊 Status Implementasi

**✅ COMPLETED COMPONENTS:**

1. **Environment Setup:** `requirements.txt` dengan versi M1-compatible
2. **Exploratory Data Analysis:** `notebooks/eda.ipynb` dengan interpretasi lengkap
3. **Preprocessing Pipeline:** Custom `ZeroToNanTransformer` + Median Imputation + StandardScaler
4. **Model Training:** Logistic Regression & Random Forest dengan MLflow tracking
5. **Web Application:** Streamlit app dengan probability visualization dan health recommendations
6. **REST API:** FastAPI dengan Pydantic validation dan interactive docs
7. **Docker Support:** Dockerfiles dan docker-compose.yml untuk containerization
8. **CI/CD Pipeline:** GitHub Actions untuk automated testing dan deployment
9. **Utilities:** Makefile dan quick start script
10. **Documentation:** README.md, CLAUDE.md, dan DEPLOYMENT.md

**🎯 KEY ACHIEVEMENTS:**

- Successfully handled 652 invalid zeros in medical data
- Implemented class imbalance handling with `class_weight='balanced'`
- Optimized for **Recall** metric (most critical for medical diagnosis)
- All models trained and saved in `models/` directory
- Production-ready modular codebase
- Working Streamlit deployment
- Production REST API with full validation
- Complete containerization dengan Docker
- Automated CI/CD pipeline
- Ready for cloud deployment (Render, Railway, AWS, GCP, Azure)

**🔧 POTENTIAL IMPROVEMENTS:**

- Hyperparameter tuning dengan GridSearch/RandomizedSearch
- SMOTE untuk advanced class imbalance handling
- Model ensemble techniques (XGBoost, LightGBM)
- Unit tests dengan pytest
- Model monitoring dan data drift detection
- Advanced feature engineering
- Model explainability dengan SHAP