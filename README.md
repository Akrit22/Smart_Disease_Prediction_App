# 🏥 Smart Disease Risk Prediction System

> **BTech CSE (Data Science) — 3rd Year Major Project**  
> **Author:** Akrit Pathania

---

## 📌 Overview

An AI-powered multi-disease risk prediction system that predicts the likelihood of:
- 🩸 **Diabetes**
- ❤️ **Heart Disease**
- ⚖️ **Obesity**
- 🧠 **Mental Health Stress**

Using lifestyle habits, clinical metrics, and behavioural data — all in one unified dashboard.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Models
```bash
python train_models.py
```

### 3. Launch the App
```bash
streamlit run app.py
```

---

## 📊 Datasets

| Dataset | Source | Rows |
|---------|--------|------|
| PIMA Indians Diabetes | Kaggle / UCI | 768 |
| UCI Heart Disease | Kaggle | 921 |
| Obesity Levels | Kaggle | 20,758 |
| OSMI Mental Health Survey | Kaggle | 1,259 |

Place all CSVs in the `data/` folder before training.

---

## 🤖 ML Models

| Disease | Algorithm | Accuracy | AUC-ROC |
|---------|-----------|----------|---------|
| Diabetes | Random Forest (200 trees) | 76.6% | 82.5% |
| Heart Disease | Gradient Boosting | 84.2% | 90.3% |
| Obesity | Random Forest (200 trees) | 97.4% | 99.4% |
| Mental Health | Random Forest (150 trees) | 83.3% | 91.3% |

---

## 🏗️ System Architecture

```
User Input (Streamlit UI)
        ↓
Data Preprocessing (Imputation + Encoding + Scaling)
        ↓
Feature Engineering (Disease-specific pipelines)
        ↓
ML Prediction Engine (4 independent models)
        ↓
Risk Scoring System (Probability calibration)
        ↓
Explainability Layer (Feature importances)
        ↓
Recommendation Engine (Rule-based + risk-driven)
        ↓
Visual Dashboard (Risk cards + charts + history)
```

---

## 📁 Project Structure

```
smart_disease_prediction/
├── app.py                 # Main Streamlit application
├── train_models.py        # ML training pipeline (all 4 models)
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── data/
│   ├── diabetes.csv
│   ├── heart_disease_uci.csv
│   ├── obesity_level.csv
│   └── survey.csv
└── models/                # Auto-generated after training
    ├── diabetes_model.pkl
    ├── diabetes_features.pkl
    ├── heart_model.pkl
    ├── heart_features.pkl
    ├── heart_encoders.pkl
    ├── obesity_model.pkl
    ├── obesity_features.pkl
    ├── obesity_encoders.pkl
    ├── mental_model.pkl
    ├── mental_features.pkl
    └── mental_encoders.pkl
```

---



## ⚠️ Disclaimer

This system is built purely for academic / educational purposes. All predictions are probabilistic ML outputs and are **not a substitute for professional medical advice**.

---

## 📄 License

MIT License — Free to use for academic projects with attribution.
