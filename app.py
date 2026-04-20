"""
Smart Disease Risk Prediction System
Streamlit App — Main Entry Point
Author: Akrit Pathania | BTech CSE (Data Science) — 3rd Year
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Disease Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main theme */
    .main { background-color: #f0f4f8; }
    
    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2980b9 50%, #6dd5fa 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(30,58,95,0.3);
    }
    .hero-header h1 { font-size: 2.5rem; margin: 0; font-weight: 800; letter-spacing: -1px; }
    .hero-header p { font-size: 1.1rem; margin-top: 0.5rem; opacity: 0.9; }
    
    /* Risk cards */
    .risk-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 5px solid #2980b9;
        transition: transform 0.2s;
    }
    .risk-card:hover { transform: translateY(-2px); }
    .risk-card.high { border-left-color: #e74c3c; }
    .risk-card.medium { border-left-color: #f39c12; }
    .risk-card.low { border-left-color: #27ae60; }
    
    /* Risk badges */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    .badge-high { background: #fde8e8; color: #c0392b; }
    .badge-medium { background: #fef5e4; color: #d35400; }
    .badge-low { background: #e8f8f0; color: #1e8449; }
    
    /* Section headers */
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e3a5f;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2980b9;
    }
    
    /* Metric boxes */
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .metric-box .metric-value { font-size: 2rem; font-weight: 800; }
    .metric-box .metric-label { font-size: 0.85rem; opacity: 0.85; }

    /* Advice cards */
    .advice-card {
        background: #eaf4ff;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #3498db;
    }
    
    /* Sidebar styling */
    .css-1d391kg { background-color: #1e3a5f !important; }
    
    /* Disclaimer */
    .disclaimer {
        background: #fff8e1;
        border: 1px solid #f9ca24;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #7d6608;
        margin-top: 1rem;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #27ae60, #f39c12, #e74c3c);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }
    
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Model Loading ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_models():
    models = {}
    try:
        models['diabetes'] = {
            'pipeline': joblib.load(os.path.join(MODEL_DIR, "diabetes_model.pkl")),
            'features': joblib.load(os.path.join(MODEL_DIR, "diabetes_features.pkl")),
        }
    except Exception as e:
        models['diabetes'] = None

    try:
        models['heart'] = {
            'pipeline': joblib.load(os.path.join(MODEL_DIR, "heart_model.pkl")),
            'features': joblib.load(os.path.join(MODEL_DIR, "heart_features.pkl")),
            'encoders': joblib.load(os.path.join(MODEL_DIR, "heart_encoders.pkl")),
        }
    except:
        models['heart'] = None

    try:
        models['obesity'] = {
            'pipeline': joblib.load(os.path.join(MODEL_DIR, "obesity_model.pkl")),
            'features': joblib.load(os.path.join(MODEL_DIR, "obesity_features.pkl")),
            'encoders': joblib.load(os.path.join(MODEL_DIR, "obesity_encoders.pkl")),
        }
    except:
        models['obesity'] = None

    try:
        models['mental'] = {
            'pipeline': joblib.load(os.path.join(MODEL_DIR, "mental_model.pkl")),
            'features': joblib.load(os.path.join(MODEL_DIR, "mental_features.pkl")),
            'encoders': joblib.load(os.path.join(MODEL_DIR, "mental_encoders.pkl")),
        }
    except:
        models['mental'] = None

    return models

MODELS = load_models()

# ─── Helper Functions ─────────────────────────────────────────────────────────
def get_risk_level(prob):
    if prob >= 0.65:
        return "HIGH", "high", "🔴"
    elif prob >= 0.40:
        return "MODERATE", "medium", "🟡"
    else:
        return "LOW", "low", "🟢"

def get_risk_color(prob):
    if prob >= 0.65: return "#e74c3c"
    elif prob >= 0.40: return "#f39c12"
    else: return "#27ae60"

def predict_diabetes(data):
    m = MODELS['diabetes']
    if not m: return None, None
    features = m['features']
    X = pd.DataFrame([{f: data.get(f, 0) for f in features}])
    prob = m['pipeline'].predict_proba(X)[0][1]
    return prob, m['pipeline']

def predict_heart(data):
    m = MODELS['heart']
    if not m: return None, None
    features = m['features']
    X = pd.DataFrame([{f: data.get(f, 0) for f in features}])
    prob = m['pipeline'].predict_proba(X)[0][1]
    return prob, m['pipeline']

def predict_obesity(data):
    m = MODELS['obesity']
    if not m: return None, None
    features = m['features']
    X = pd.DataFrame([{f: data.get(f, 0) for f in features}])
    prob = m['pipeline'].predict_proba(X)[0][1]
    return prob, m['pipeline']

def predict_mental(data):
    m = MODELS['mental']
    if not m: return None, None
    features = m['features']
    X = pd.DataFrame([{f: data.get(f, 0) for f in features}])
    prob = m['pipeline'].predict_proba(X)[0][1]
    return prob, m['pipeline']

def render_risk_card(disease, emoji, prob, pipeline, feature_names):
    level, css_class, dot = get_risk_level(prob)
    color = get_risk_color(prob)
    pct = int(prob * 100)

    # Feature importances
    imp_html = ""
    try:
        model_step = pipeline.named_steps['model']
        if hasattr(model_step, 'feature_importances_'):
            importances = model_step.feature_importances_
            top_idx = np.argsort(importances)[::-1][:3]
            factors = [f"<li><b>{feature_names[i]}</b> — importance: {importances[i]:.2%}</li>" for i in top_idx]
            imp_html = f"<p style='font-size:0.85rem;color:#555;margin-top:0.8rem;'>🔍 Top contributing factors:<ul>{''.join(factors)}</ul></p>"
    except:
        pass

    st.markdown(f"""
    <div class="risk-card {css_class}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:1.5rem;">{emoji}</span>
                <span style="font-size:1.2rem; font-weight:700; margin-left:0.5rem; color:#1e3a5f;">{disease}</span>
            </div>
            <div>
                <span class="badge badge-{css_class}">{dot} {level} RISK</span>
            </div>
        </div>
        <div style="margin-top:1rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:0.85rem; color:#666;">Risk Probability</span>
                <span style="font-size:1.1rem; font-weight:800; color:{color};">{pct}%</span>
            </div>
            <div style="background:#eee; border-radius:10px; height:10px;">
                <div style="width:{pct}%; background:{color}; height:10px; border-radius:10px; transition: width 1s;"></div>
            </div>
        </div>
        {imp_html}
    </div>
    """, unsafe_allow_html=True)

def get_recommendations(results):
    recs = []
    if results.get('diabetes', 0) >= 0.5:
        recs.extend([
            "🥗 Reduce sugar and refined carbohydrate intake significantly",
            "🚶 Aim for 30 minutes of moderate exercise at least 5 days/week",
            "⚖️ Work towards maintaining a healthy BMI (18.5–24.9)",
            "🩺 Monitor blood glucose levels regularly; consult an endocrinologist",
        ])
    if results.get('heart', 0) >= 0.5:
        recs.extend([
            "🫀 Reduce sodium intake to < 2g/day to lower blood pressure",
            "🚭 Avoid smoking and limit alcohol consumption",
            "🐟 Include omega-3 rich foods (salmon, flaxseed, walnuts) in your diet",
            "💊 Discuss cholesterol levels with a cardiologist",
        ])
    if results.get('obesity', 0) >= 0.5:
        recs.extend([
            "🥦 Replace processed foods with whole fruits, vegetables, and grains",
            "🏃 Incorporate aerobic + strength training exercises weekly",
            "💧 Drink at least 2 litres of water per day",
            "😴 Ensure 7–8 hours of quality sleep nightly",
        ])
    if results.get('mental', 0) >= 0.5:
        recs.extend([
            "🧘 Practice mindfulness meditation or breathing exercises daily",
            "👥 Seek support from a mental health professional if stress persists",
            "📵 Limit screen time and social media use in evenings",
            "📓 Maintain a daily journal to process thoughts and emotions",
        ])
    if not recs:
        recs = [
            "✅ Great news — your current risk levels are low!",
            "🥗 Maintain a balanced, nutritious diet",
            "🏃 Stay physically active with regular exercise",
            "🩺 Schedule annual health check-ups as preventive care",
            "😴 Prioritise quality sleep and stress management",
        ]
    return recs

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <span style="font-size:3rem;">🏥</span>
        <h2 style="color:#2980b9; margin:0.5rem 0;">HealthPredict AI</h2>
        <p style="color:#666; font-size:0.85rem;">Smart Disease Risk System</p>
    </div>
    <hr style="border:1px solid #eee;">
    """, unsafe_allow_html=True)
    
    nav = st.radio(
        "Navigation",
        ["🏠 Home & Predict", "📊 Model Performance", "📖 About the Project"],
        label_visibility="collapsed"
    )
    
    st.markdown("""
    <hr style="border:1px solid #eee;">
    <div style="font-size:0.8rem; color:#888; text-align:center;">
        <p><b>Diseases Covered</b></p>
        <p>🩸 Diabetes</p>
        <p>❤️ Heart Disease</p>
        <p>⚖️ Obesity</p>
        <p>🧠 Mental Health Stress</p>
        <hr>
        <p>Built by <b>Akrit Pathania</b></p>
        <p>BTech CSE (DS) · 3rd Year</p>
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOME & PREDICT
# ═════════════════════════════════════════════════════════════════════════════
if "Home" in nav:
    # Hero
    st.markdown("""
    <div class="hero-header">
        <h1>🏥 Smart Disease Risk Prediction System</h1>
        <p>AI-powered multi-disease risk analysis using your lifestyle & health data</p>
        <p style="font-size:0.85rem; opacity:0.7;">Powered by Random Forest · Gradient Boosting · Logistic Regression</p>
    </div>
    """, unsafe_allow_html=True)

    # ── INPUT FORM ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📝 Enter Your Health Profile</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🩸 Diabetes Inputs", "❤️ Heart Disease Inputs", "⚖️ Obesity Inputs", "🧠 Mental Health Inputs"])
    
    user_data = {}

    # ── TAB 1: DIABETES ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("**Clinical & Biological Metrics**")
        col1, col2, col3 = st.columns(3)
        with col1:
            user_data['Pregnancies'] = st.number_input("Pregnancies", 0, 20, 1, help="Number of pregnancies (enter 0 if male)")
            user_data['Glucose'] = st.slider("Blood Glucose (mg/dL)", 50, 300, 110, help="Plasma glucose concentration")
            user_data['BloodPressure'] = st.slider("Blood Pressure (mm Hg)", 30, 140, 72)
        with col2:
            user_data['SkinThickness'] = st.slider("Skin Thickness (mm)", 0, 100, 25)
            user_data['Insulin'] = st.slider("Insulin (mu U/ml)", 0, 900, 80)
            user_data['BMI'] = st.slider("BMI", 10.0, 70.0, 25.0, step=0.5)
        with col3:
            user_data['DiabetesPedigreeFunction'] = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.4, step=0.01, help="Genetic likelihood of diabetes")
            user_data['Age'] = st.slider("Age", 18, 100, 30)
        
        st.info("💡 Diabetes Pedigree Function reflects hereditary risk. Higher value = stronger family history.")

    # ── TAB 2: HEART DISEASE ─────────────────────────────────────────────────
    with tab2:
        st.markdown("**Cardiovascular Indicators**")
        col1, col2, col3 = st.columns(3)
        with col1:
            heart_age = st.slider("Age", 18, 100, 50, key="heart_age")
            heart_sex = st.selectbox("Sex", ["Male", "Female"])
            heart_cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
            heart_trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, 120)
        with col2:
            heart_chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
            heart_fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [False, True])
            heart_restecg = st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t abnormality"])
            heart_thalch = st.slider("Max Heart Rate Achieved", 60, 220, 150)
        with col3:
            heart_exang = st.selectbox("Exercise-Induced Angina", [False, True])
            heart_oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 7.0, 1.0, step=0.1)
            heart_slope = st.selectbox("Slope of Peak ST", ["upsloping", "flat", "downsloping"])
            heart_ca = st.slider("Major Vessels Colored (Fluoroscopy)", 0, 4, 0)
            heart_thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

        # Encode heart features
        from sklearn.preprocessing import LabelEncoder
        heart_le_dict = joblib.load(os.path.join(MODEL_DIR, "heart_encoders.pkl")) if MODELS['heart'] else {}
        
        def encode_val(le_dict, col, val):
            if col in le_dict:
                le = le_dict[col]
                val_str = str(val)
                if val_str in le.classes_:
                    return le.transform([val_str])[0]
                else:
                    return 0
            return val
        
        user_data['age'] = heart_age
        user_data['sex'] = encode_val(heart_le_dict, 'sex', heart_sex)
        user_data['cp'] = encode_val(heart_le_dict, 'cp', heart_cp)
        user_data['trestbps'] = heart_trestbps
        user_data['chol'] = heart_chol
        user_data['fbs'] = encode_val(heart_le_dict, 'fbs', heart_fbs)
        user_data['restecg'] = encode_val(heart_le_dict, 'restecg', heart_restecg)
        user_data['thalch'] = heart_thalch
        user_data['exang'] = encode_val(heart_le_dict, 'exang', heart_exang)
        user_data['oldpeak'] = heart_oldpeak
        user_data['slope'] = encode_val(heart_le_dict, 'slope', heart_slope)
        user_data['ca'] = heart_ca
        user_data['thal'] = encode_val(heart_le_dict, 'thal', heart_thal)

        st.info("💡 ST depression (oldpeak) and chest pain type are among the strongest predictors of heart disease.")

    # ── TAB 3: OBESITY ────────────────────────────────────────────────────────
    with tab3:
        st.markdown("**Lifestyle & Physical Data**")
        col1, col2, col3 = st.columns(3)
        
        obesity_le_dict = joblib.load(os.path.join(MODEL_DIR, "obesity_encoders.pkl")) if MODELS['obesity'] else {}
        
        with col1:
            ob_gender = st.selectbox("Gender", ["Male", "Female"], key="ob_gender")
            ob_age = st.slider("Age", 14, 80, 25, key="ob_age")
            ob_height = st.slider("Height (m)", 1.40, 2.10, 1.70, step=0.01)
            ob_weight = st.slider("Weight (kg)", 30.0, 200.0, 70.0, step=0.5)
            ob_family = st.selectbox("Family History of Overweight", [1, 0], format_func=lambda x: "Yes" if x else "No")
        with col2:
            ob_favc = st.selectbox("Frequent High-Calorie Food?", [1, 0], format_func=lambda x: "Yes" if x else "No")
            ob_fcvc = st.slider("Vegetable Frequency (1–3)", 1.0, 3.0, 2.0, step=0.5)
            ob_ncp = st.slider("Main Meals per Day", 1.0, 4.0, 3.0, step=0.5)
            ob_caec = st.selectbox("Eating Between Meals", ["no", "Sometimes", "Frequently", "Always"])
            ob_smoke = st.selectbox("Smoker?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with col3:
            ob_ch2o = st.slider("Water Intake (litres/day)", 1.0, 3.0, 2.0, step=0.1)
            ob_scc = st.selectbox("Monitor Calorie Intake?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            ob_faf = st.slider("Physical Activity Frequency (0–3)", 0.0, 3.0, 1.0, step=0.5)
            ob_tue = st.slider("Tech Device Usage (hrs/day)", 0.0, 3.0, 1.0, step=0.5)
            ob_calc = st.selectbox("Alcohol Consumption", ["no", "Sometimes", "Frequently", "Always"])
            ob_mtrans = st.selectbox("Primary Transport", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
        
        user_data['Gender'] = encode_val(obesity_le_dict, 'Gender', ob_gender)
        user_data['Age'] = ob_age
        user_data['Height'] = ob_height
        user_data['Weight'] = ob_weight
        user_data['family_history_with_overweight'] = ob_family
        user_data['FAVC'] = ob_favc
        user_data['FCVC'] = ob_fcvc
        user_data['NCP'] = ob_ncp
        user_data['CAEC'] = encode_val(obesity_le_dict, 'CAEC', ob_caec)
        user_data['SMOKE'] = ob_smoke
        user_data['CH2O'] = ob_ch2o
        user_data['SCC'] = ob_scc
        user_data['FAF'] = ob_faf
        user_data['TUE'] = ob_tue
        user_data['CALC'] = encode_val(obesity_le_dict, 'CALC', ob_calc)
        user_data['MTRANS'] = encode_val(obesity_le_dict, 'MTRANS', ob_mtrans)

        st.info("💡 Physical activity frequency, vegetable consumption, and transport mode are key obesity indicators.")

    # ── TAB 4: MENTAL HEALTH ──────────────────────────────────────────────────
    with tab4:
        st.markdown("**Workplace & Mental Health Factors**")
        col1, col2 = st.columns(2)
        
        mental_le_dict = joblib.load(os.path.join(MODEL_DIR, "mental_encoders.pkl")) if MODELS['mental'] else {}
        
        def encode_mental(col, val):
            return encode_val(mental_le_dict, col, val)
        
        with col1:
            m_age = st.slider("Age", 18, 75, 30, key="m_age")
            m_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="m_gender")
            m_family = st.selectbox("Family History of Mental Illness", ["Yes", "No"], key="m_fam")
            m_wi = st.selectbox("Does Work Interfere with Mental Health?", ["Often", "Sometimes", "Rarely", "Never"])
            m_emp = st.selectbox("Company Size", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
            m_remote = st.selectbox("Remote Work?", ["Yes", "No"])
            m_tech = st.selectbox("Tech Company?", ["Yes", "No"])
        with col2:
            m_benefits = st.selectbox("Does Employer Provide Mental Health Benefits?", ["Yes", "No", "Don't know"])
            m_care = st.selectbox("Are Care Options Available?", ["Yes", "No", "Not sure"])
            m_wellness = st.selectbox("Employer has Wellness Programme?", ["Yes", "No", "Don't know"])
            m_seekhelp = st.selectbox("Employer Encourages Seeking Help?", ["Yes", "No", "Don't know"])
            m_anon = st.selectbox("Anonymity Protected if You Seek Help?", ["Yes", "No", "Don't know"])
            m_mhcons = st.selectbox("Negative Consequence from Mental Health Disclosure?", ["Yes", "No", "Maybe"])
            m_phcons = st.selectbox("Negative Consequence from Physical Health Disclosure?", ["Yes", "No", "Maybe"])
        
        user_data['Age_m'] = m_age
        user_data['Gender_m'] = encode_mental('Gender', m_gender)
        user_data['family_history'] = encode_mental('family_history', m_family)
        user_data['work_interfere'] = encode_mental('work_interfere', m_wi)
        user_data['no_employees'] = encode_mental('no_employees', m_emp)
        user_data['remote_work'] = encode_mental('remote_work', m_remote)
        user_data['tech_company'] = encode_mental('tech_company', m_tech)
        user_data['benefits'] = encode_mental('benefits', m_benefits)
        user_data['care_options'] = encode_mental('care_options', m_care)
        user_data['wellness_program'] = encode_mental('wellness_program', m_wellness)
        user_data['seek_help'] = encode_mental('seek_help', m_seekhelp)
        user_data['anonymity'] = encode_mental('anonymity', m_anon)
        user_data['mental_health_consequence'] = encode_mental('mental_health_consequence', m_mhcons)
        user_data['phys_health_consequence'] = encode_mental('phys_health_consequence', m_phcons)

        st.info("💡 Workplace support factors — benefits, anonymity, and family history — strongly predict mental health stress.")

    # ── PREDICT BUTTON ───────────────────────────────────────────────────────
    st.markdown("---")
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        predict_btn = st.button("🔍 Analyse My Health Risk", use_container_width=True, type="primary")

    if predict_btn:
        st.markdown('<div class="section-title">📋 Your Personalised Risk Assessment</div>', unsafe_allow_html=True)

        with st.spinner("🔬 Running AI models..."):
            # Build mental health input correctly
            mental_input = {
                'Age': user_data.get('Age_m', 30),
                'Gender': user_data.get('Gender_m', 0),
                'family_history': user_data.get('family_history', 0),
                'work_interfere': user_data.get('work_interfere', 0),
                'no_employees': user_data.get('no_employees', 0),
                'remote_work': user_data.get('remote_work', 0),
                'tech_company': user_data.get('tech_company', 0),
                'benefits': user_data.get('benefits', 0),
                'care_options': user_data.get('care_options', 0),
                'wellness_program': user_data.get('wellness_program', 0),
                'seek_help': user_data.get('seek_help', 0),
                'anonymity': user_data.get('anonymity', 0),
                'mental_health_consequence': user_data.get('mental_health_consequence', 0),
                'phys_health_consequence': user_data.get('phys_health_consequence', 0),
            }

            d_prob, d_pipe = predict_diabetes(user_data)
            h_prob, h_pipe = predict_heart(user_data)
            o_prob, o_pipe = predict_obesity(user_data)
            m_prob, m_pipe = predict_mental(mental_input)

        results = {
            'diabetes': d_prob or 0,
            'heart': h_prob or 0,
            'obesity': o_prob or 0,
            'mental': m_prob or 0,
        }

        # ── RISK SUMMARY METRICS ─────────────────────────────────────────────
        overall_risk = np.mean(list(results.values()))
        high_risks = sum(1 for v in results.values() if v >= 0.65)
        moderate_risks = sum(1 for v in results.values() if 0.40 <= v < 0.65)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            color = get_risk_color(overall_risk)
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{color}dd,{color}88);color:white;padding:1rem;border-radius:10px;text-align:center;">
                <div style="font-size:2rem;font-weight:800;">{int(overall_risk*100)}%</div>
                <div style="font-size:0.85rem;">Overall Risk Score</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#e74c3cdd,#e74c3c88);color:white;padding:1rem;border-radius:10px;text-align:center;">
                <div style="font-size:2rem;font-weight:800;">{high_risks}</div>
                <div style="font-size:0.85rem;">High Risk Areas</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#f39c12dd,#f39c1288);color:white;padding:1rem;border-radius:10px;text-align:center;">
                <div style="font-size:2rem;font-weight:800;">{moderate_risks}</div>
                <div style="font-size:0.85rem;">Moderate Risk Areas</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            low_risks = 4 - high_risks - moderate_risks
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#27ae60dd,#27ae6088);color:white;padding:1rem;border-radius:10px;text-align:center;">
                <div style="font-size:2rem;font-weight:800;">{low_risks}</div>
                <div style="font-size:0.85rem;">Low Risk Areas</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── INDIVIDUAL RISK CARDS ────────────────────────────────────────────
        col_left, col_right = st.columns(2)
        with col_left:
            if d_prob is not None:
                render_risk_card("Diabetes Risk", "🩸", d_prob, d_pipe, MODELS['diabetes']['features'])
            if o_prob is not None:
                render_risk_card("Obesity Risk", "⚖️", o_prob, o_pipe, MODELS['obesity']['features'])
        with col_right:
            if h_prob is not None:
                render_risk_card("Heart Disease Risk", "❤️", h_prob, h_pipe, MODELS['heart']['features'])
            if m_prob is not None:
                render_risk_card("Mental Health Stress Risk", "🧠", m_prob, m_pipe, MODELS['mental']['features'])

        # ── RECOMMENDATIONS ──────────────────────────────────────────────────
        st.markdown('<div class="section-title">💊 Personalised Health Recommendations</div>', unsafe_allow_html=True)
        recs = get_recommendations(results)
        cols = st.columns(2)
        for i, rec in enumerate(recs):
            with cols[i % 2]:
                st.markdown(f'<div class="advice-card">{rec}</div>', unsafe_allow_html=True)

        # ── DISCLAIMER ───────────────────────────────────────────────────────
        st.markdown("""
        <div class="disclaimer">
            ⚠️ <b>Medical Disclaimer:</b> This tool is for educational and research purposes only. 
            Predictions are based on statistical models and are <b>not a substitute for professional medical diagnosis</b>. 
            Please consult a qualified healthcare provider for medical advice.
        </div>
        """, unsafe_allow_html=True)

        # ── SAVE TO HISTORY ──────────────────────────────────────────────────
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        st.session_state['history'].append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'diabetes': round(d_prob or 0, 3),
            'heart': round(h_prob or 0, 3),
            'obesity': round(o_prob or 0, 3),
            'mental': round(m_prob or 0, 3),
            'overall': round(overall_risk, 3),
        })

    # ── SESSION HISTORY ───────────────────────────────────────────────────────
    if 'history' in st.session_state and st.session_state['history']:
        st.markdown('<div class="section-title">📈 Prediction History (This Session)</div>', unsafe_allow_html=True)
        hist_df = pd.DataFrame(st.session_state['history'])
        hist_df.columns = ['Time', 'Diabetes %', 'Heart %', 'Obesity %', 'Mental %', 'Overall %']
        for col in ['Diabetes %', 'Heart %', 'Obesity %', 'Mental %', 'Overall %']:
            hist_df[col] = (hist_df[col] * 100).round(1)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif "Performance" in nav:
    st.markdown("""
    <div class="hero-header">
        <h1>📊 Model Performance Dashboard</h1>
        <p>Evaluation metrics for all trained ML models</p>
    </div>
    """, unsafe_allow_html=True)

    performance_data = {
        'Model': ['🩸 Diabetes (RF)', '❤️ Heart Disease (GBM)', '⚖️ Obesity (RF)', '🧠 Mental Health (RF)'],
        'Algorithm': ['Random Forest', 'Gradient Boosting', 'Random Forest', 'Random Forest'],
        'Accuracy': [0.766, 0.842, 0.974, 0.833],
        'AUC-ROC': [0.825, 0.903, 0.994, 0.913],
        'CV AUC (5-fold)': [0.838, 0.790, 0.995, 0.888],
        'Training Samples': [614, 736, 16607, 1008],
        'Test Samples': [154, 185, 4152, 252],
    }
    perf_df = pd.DataFrame(performance_data)

    st.markdown('<div class="section-title">📋 Summary Table</div>', unsafe_allow_html=True)
    
    # Color-coded display
    def color_accuracy(val):
        color = '#27ae60' if val >= 0.85 else '#f39c12' if val >= 0.75 else '#e74c3c'
        return f'background-color: {color}22; color: {color}; font-weight: bold'
    
    styled = perf_df.style.map(color_accuracy, subset=['Accuracy'])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">📊 Performance Visualisation</div>', unsafe_allow_html=True)
    
    # Bar charts using streamlit native
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Test Accuracy by Model**")
        acc_df = pd.DataFrame({
            'Model': ['Diabetes', 'Heart', 'Obesity', 'Mental'],
            'Accuracy': [76.6, 84.2, 97.4, 83.3]
        }).set_index('Model')
        st.bar_chart(acc_df)
    with col2:
        st.markdown("**AUC-ROC Score by Model**")
        auc_df = pd.DataFrame({
            'Model': ['Diabetes', 'Heart', 'Obesity', 'Mental'],
            'AUC-ROC': [82.5, 90.3, 99.4, 91.3]
        }).set_index('Model')
        st.bar_chart(auc_df)

    st.markdown('<div class="section-title">🔬 Model Details</div>', unsafe_allow_html=True)
    
    details = [
        ("🩸 Diabetes", "Random Forest (200 trees, depth=8)", "PIMA Indians Diabetes (768 samples)", 
         "Glucose, BMI, Age, DiabetesPedigreeFunction, Insulin", "76.6%", "82.5%"),
        ("❤️ Heart Disease", "Gradient Boosting (200 est., depth=4, lr=0.05)", "UCI Heart Disease (921 samples, 4 cities)",
         "Chest pain type, Oldpeak, Max heart rate, Thalassemia, CA", "84.2%", "90.3%"),
        ("⚖️ Obesity", "Random Forest (200 trees, depth=10)", "Obesity Levels — Eating Habits (20,758 samples)",
         "Weight, FAF, CAEC, FAVC, CH2O", "97.4%", "99.4%"),
        ("🧠 Mental Health", "Random Forest (150 trees, depth=6)", "OSMI Mental Health in Tech Survey (1,259 samples)",
         "Family history, Work interference, Benefits, Anonymity", "83.3%", "91.3%"),
    ]
    
    for name, algo, data, top_feat, acc, auc in details:
        with st.expander(f"{name} Model Details"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Algorithm:** {algo}")
                st.markdown(f"**Dataset:** {data}")
                st.markdown(f"**Key Features:** {top_feat}")
            with c2:
                st.metric("Test Accuracy", acc)
                st.metric("AUC-ROC", auc)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif "About" in nav:
    st.markdown("""
    <div class="hero-header">
        <h1>📖 About This Project</h1>
        <p>Smart Disease Risk Prediction System — Major Project Documentation</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ## 🎯 Project Overview

        The **Smart Disease Risk Prediction System** is a multi-label classification project that predicts 
        an individual's risk for four major health conditions simultaneously using lifestyle, clinical, 
        and behavioural data.

        Unlike traditional single-disease tools, this system provides a **holistic health risk profile** 
        enabling early detection and personalised preventive recommendations.

        ## 🌍 Problem Statement

        Millions of people are diagnosed with lifestyle diseases only after symptoms appear — 
        often when the condition has already progressed significantly. The lack of accessible, 
        personalised risk assessment tools prevents early intervention.

        ## 🤖 Solution
        
        An AI-powered system that:
        - Takes user lifestyle and clinical inputs
        - Runs them through four specialised ML models
        - Returns probability-based risk scores with explanations
        - Provides actionable, personalised health recommendations

        ## 🏗️ System Architecture

        ```
        User Input → Data Preprocessing → Feature Engineering
             ↓
        ML Models (RF / GBM / LR)
             ↓
        Risk Scoring (Probability Calibration)
             ↓
        Explainability (Feature Importances)
             ↓
        Recommendation Engine → Dashboard Output
        ```

        ## 📊 Datasets Used

        | Dataset | Source | Samples | Target |
        |---------|--------|---------|--------|
        | PIMA Indians Diabetes | Kaggle/UCI | 768 | Diabetes |
        | UCI Heart Disease | Kaggle | 921 | Heart Disease |
        | Obesity Levels | Kaggle | 20,758 | Obesity |
        | OSMI Mental Health Survey | Kaggle | 1,259 | Mental Stress |

        ## 🧰 Technology Stack
        
        - **Frontend:** Streamlit (Python)
        - **ML Framework:** Scikit-learn
        - **Algorithms:** Random Forest, Gradient Boosting
        - **Data Processing:** Pandas, NumPy
        - **Model Serialisation:** Joblib
        - **Visualisation:** Streamlit native charts
        - **Deployment:** Streamlit Cloud / Render
        """)

    with col2:
        st.markdown("""
        ## 👨‍💻 Developer

        **Akrit Pathania**  
        BTech CSE (Data Science)  
        3rd Year Major Project
        
        ---
        
        ## 📈 Model Accuracy
        
        | Disease | Accuracy |
        |---------|----------|
        | 🩸 Diabetes | 76.6% |
        | ❤️ Heart | 84.2% |
        | ⚖️ Obesity | 97.4% |
        | 🧠 Mental | 83.3% |
        
        ---
        
        ## 🔬 Key Features
        
        ✅ Multi-disease prediction  
        ✅ Real-time risk scoring  
        ✅ Feature importance display  
        ✅ Personalised recommendations  
        ✅ Session history tracking  
        ✅ Mobile-responsive UI  
        ✅ Production-ready ML pipeline  
        
        ---
        
        ## 📁 Project Structure
        ```
        smart_disease_prediction/
        ├── app.py          # Main Streamlit app
        ├── train_models.py # ML training pipeline
        ├── models/         # Saved .pkl model files
        ├── data/           # Datasets
        └── requirements.txt
        ```
        """)

    st.markdown("""
    <div class="disclaimer" style="margin-top:2rem;">
        ⚠️ <b>Disclaimer:</b> This system is built for academic purposes as part of a BTech Major Project. 
        All predictions are probabilistic estimates from ML models trained on public datasets. 
        This is NOT a substitute for professional medical advice, diagnosis, or treatment.
    </div>
    """, unsafe_allow_html=True)
