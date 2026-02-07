## 📊 Diabetes Prediction MLOps Project

**Status:** ✅ **COMPLETE & PRODUCTION-READY**

Proyek ini adalah implementasi *end-to-end* Machine Learning untuk memprediksi risiko diabetes menggunakan **Pima Indians Diabetes Dataset**. Fokus utama proyek ini bukan hanya akurasi model, tetapi juga implementasi **MLOps** menggunakan **Scikit-learn Pipelines** untuk prapemrosesan data dan **MLflow** untuk *experiment tracking*.

### 🚀 Fitur Utama

* **Modular Codebase:** Pemisahan antara skrip prapemrosesan, pelatihan, dan inferensi.
* **Robust Pipeline:** Penanganan *missing values* (nilai 0 yang tidak masuk akal) dan *feature scaling* dalam satu alur pipeline.
* **Experiment Tracking:** Pencatatan parameter, metrik (Accuracy, Recall, F1-Score), dan artefak model menggunakan **MLflow**.
* **Interactive UI:** Dashboard prediksi berbasis **Streamlit**.
* **M1 Optimized:** Konfigurasi yang ringan dan kompatibel untuk Apple Silicon.

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Library ML:** Scikit-learn, Pandas, NumPy
* **MLOps:** MLflow
* **Frontend:** Streamlit
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
│   ├── training.py               # Skrip pelatihan model & MLflow logging
│   └── processing.py             # Definisi pipeline prapemrosesan
├── app.py                        # Aplikasi Streamlit (deployed)
├── requirements.txt               # Daftar dependensi (M1-compatible)
├── README.md                      # Dokumentasi proyek
├── CLAUDE.md                      # Panduan untuk Claude Code
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

Buka `http://localhost:5000` di browser Anda.

### 2. Menjalankan Dashboard Prediksi

Setelah mendapatkan model terbaik, jalankan aplikasi Streamlit untuk melakukan prediksi interaktif:

```bash
streamlit run app.py

```

---

## 📝 Catatan Prapemrosesan

Dataset ini memiliki nilai `0` pada kolom seperti `BloodPressure` dan `BMI` yang secara medis tidak valid. Pipeline dalam proyek ini menangani hal tersebut dengan:

1. Mengubah nilai `0` menjadi `NaN`.
2. Melakukan **SimpleImputer** (Median) untuk mengisi nilai yang hilang.
3. Melakukan **StandardScaler** agar model linear/SVM dapat bekerja optimal.

---

## 🤝 Kontribusi

Jika Anda ingin mengembangkan proyek ini lebih lanjut (misalnya: menambahkan integrasi Docker atau CI/CD), silakan ajukan *pull request*.

---

## 📊 Status Implementasi

**✅ COMPLETED COMPONENTS:**

1. **Environment Setup:** `requirements.txt` dengan versi M1-compatible
2. **Exploratory Data Analysis:** `notebooks/eda.ipynb` dengan interpretasi lengkap
3. **Preprocessing Pipeline:** Custom `ZeroToNanTransformer` + Median Imputation + StandardScaler
4. **Model Training:** Logistic Regression & Random Forest dengan MLflow tracking
5. **Web Application:** Streamlit app dengan probability visualization dan health recommendations
6. **Documentation:** README.md dan CLAUDE.md

**🎯 KEY ACHIEVEMENTS:**

- Successfully handled 652 invalid zeros in medical data
- Implemented class imbalance handling with `class_weight='balanced'`
- Optimized for **Recall** metric (most critical for medical diagnosis)
- All models trained and saved in `models/` directory
- Production-ready modular codebase
- Working Streamlit deployment

**🔧 POTENTIAL IMPROVEMENTS:**

- Hyperparameter tuning dengan GridSearch/RandomizedSearch
- SMOTE untuk advanced class imbalance handling
- Model ensemble techniques
- Docker containerization
- CI/CD pipeline integration
- API deployment (FastAPI/Flask)