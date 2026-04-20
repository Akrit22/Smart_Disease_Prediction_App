"""
Smart Disease Risk Prediction System
Train ML models for: Diabetes, Heart Disease, Obesity, Mental Health (Stress)
Author: Akrit Pathania
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DIABETES MODEL
# ─────────────────────────────────────────────────────────────────────────────
def train_diabetes():
    print("\n📊 Training Diabetes Model...")
    df = pd.read_csv(os.path.join(DATA_DIR, "diabetes.csv"))
    
    # Replace zeros that are physiologically impossible with NaN
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced'))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc').mean()
    
    print(f"  ✅ Accuracy: {acc:.3f} | AUC-ROC: {auc:.3f} | CV AUC: {cv:.3f}")
    
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "diabetes_model.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "diabetes_features.pkl"))
    return pipeline, feature_names

# ─────────────────────────────────────────────────────────────────────────────
# 2. HEART DISEASE MODEL
# ─────────────────────────────────────────────────────────────────────────────
def train_heart_disease():
    print("\n❤️  Training Heart Disease Model...")
    df = pd.read_csv(os.path.join(DATA_DIR, "heart_disease_uci.csv"))
    
    # Binary target: num > 0 means disease present
    df['target'] = (df['num'] > 0).astype(int)
    df = df.drop(['id', 'num', 'dataset'], axis=1, errors='ignore')
    
    # Encode categoricals
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'target']
    
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc').mean()
    
    print(f"  ✅ Accuracy: {acc:.3f} | AUC-ROC: {auc:.3f} | CV AUC: {cv:.3f}")
    
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "heart_model.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "heart_features.pkl"))
    joblib.dump(le_dict, os.path.join(MODEL_DIR, "heart_encoders.pkl"))
    return pipeline, feature_names

# ─────────────────────────────────────────────────────────────────────────────
# 3. OBESITY MODEL
# ─────────────────────────────────────────────────────────────────────────────
def train_obesity():
    print("\n⚖️  Training Obesity Model...")
    df = pd.read_csv(os.path.join(DATA_DIR, "obesity_level.csv"))
    
    # Rename last column if messy
    df.columns = [c.strip() for c in df.columns]
    target_col = df.columns[-1]
    df = df.rename(columns={target_col: 'obesity_level'})
    
    # Binary: obese or not (Obesity I, II, III = 1)
    obese_classes = ['Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III',
                     'Overweight_Level_I', 'Overweight_Level_II']
    df['target'] = df['obesity_level'].apply(
        lambda x: 1 if any(o in str(x) for o in ['Obesity', 'Overweight']) else 0
    )
    
    df = df.drop(['id', 'obesity_level'], axis=1, errors='ignore')
    
    # Encode categoricals
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'target']
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc').mean()
    
    print(f"  ✅ Accuracy: {acc:.3f} | AUC-ROC: {auc:.3f} | CV AUC: {cv:.3f}")
    
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "obesity_model.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "obesity_features.pkl"))
    joblib.dump(le_dict, os.path.join(MODEL_DIR, "obesity_encoders.pkl"))
    return pipeline, feature_names

# ─────────────────────────────────────────────────────────────────────────────
# 4. MENTAL HEALTH / STRESS MODEL
# ─────────────────────────────────────────────────────────────────────────────
def train_mental_health():
    print("\n🧠 Training Mental Health / Stress Model...")
    df = pd.read_csv(os.path.join(DATA_DIR, "survey.csv"))
    
    # Target: did they seek treatment?
    df['target'] = (df['treatment'] == 'Yes').astype(int)
    
    # Select useful features
    features = ['Age', 'Gender', 'family_history', 'work_interfere',
                'no_employees', 'remote_work', 'tech_company', 'benefits',
                'care_options', 'wellness_program', 'seek_help', 'anonymity',
                'mental_health_consequence', 'phys_health_consequence']
    
    df = df[features + ['target']].copy()
    
    # Clean age
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df = df[(df['Age'] >= 18) & (df['Age'] <= 80)]
    
    # Clean gender
    df['Gender'] = df['Gender'].str.strip().str.lower()
    df['Gender'] = df['Gender'].apply(
        lambda x: 'Male' if 'm' in str(x) and 'fe' not in str(x) else
                  ('Female' if 'f' in str(x) else 'Other')
    )
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, class_weight='balanced'))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc').mean()
    
    print(f"  ✅ Accuracy: {acc:.3f} | AUC-ROC: {auc:.3f} | CV AUC: {cv:.3f}")
    
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "mental_model.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "mental_features.pkl"))
    joblib.dump(le_dict, os.path.join(MODEL_DIR, "mental_encoders.pkl"))
    return pipeline, feature_names

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🏥 Smart Disease Risk Prediction System - Model Training")
    print("=" * 60)
    
    train_diabetes()
    train_heart_disease()
    train_obesity()
    train_mental_health()
    
    print("\n" + "=" * 60)
    print("✅ All models trained and saved to /models/")
    print("=" * 60)
