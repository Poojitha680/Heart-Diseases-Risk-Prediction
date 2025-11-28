import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Load pre-trained model, features, and scaler

MODEL_PATH = os.path.join('..', 'models', 'best_model.pkl')
FEATURES_PATH = os.path.join('..', 'models', 'feature_columns.pkl')
SCALER_PATH = os.path.join('..', 'models', 'scaler.pkl')

# Load model and feature columns
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, 'rb') as f:
        feature_columns = pickle.load(f)
except Exception as e:
    st.error(f" Error loading model or feature columns: {e}")
    st.stop()

# Load scaler (optional)
scaler = None
if os.path.exists(SCALER_PATH):
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.warning(f" Scaler could not be loaded: {e}")
else:
    st.warning(" No scaler.pkl found. Predictions may be less accurate.")

# Page setup
st.set_page_config(page_title="Heart Disease Predictor", page_icon="💓", layout="wide")
st.title("💓 Heart Disease Risk Prediction")
st.markdown("Enter patient details to assess the risk of heart disease using the trained ML model.")

# User Input Section
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, value=50)
    sex = st.selectbox("Sex", ("Male", "Female"))
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, value=120)
    chol = st.number_input("Cholesterol Level (mg/dl)", 100, 400, value=200)
    fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl (1=True, 0=False)", [0, 1])
    restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", 70, 210, value=150)
    exang = st.selectbox("Exercise Induced Angina (1=True, 0=False)", [0, 1])
    oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, step=0.1, value=1.0)
    slope = st.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (1=Normal, 2=Fixed Defect, 3=Reversible)", [1, 2, 3])


# Prepare DataFrame

user_input = {
    'age': age,
    'sex': 1 if sex == "Male" else 0,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}

input_df = pd.DataFrame([user_input])

# Align columns
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_columns]

# Scale input
if scaler:
    input_scaled = scaler.transform(input_df)
else:
    input_scaled = input_df


# Sensitivity Control

st.sidebar.header(" Prediction Settings")
threshold = st.sidebar.slider(
    "Set Risk Sensitivity Threshold",
    0.1, 0.9, 0.35, 0.05,
    help="Lower threshold → more sensitive to high-risk cases (detects more positives)."
)

# Prediction Section

if st.button("Predict"):
    try:
        # Get probability
        prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else 0.5
        prediction = 1 if prob >= threshold else 0

        st.subheader("Prediction Result:")

        if prediction == 1:
            st.error(" **High risk of heart disease detected.** Immediate medical consultation advised.")
        else:
            st.success(" **Low risk of heart disease detected.** Keep maintaining a healthy lifestyle!")

        st.progress(float(prob))
        st.write(f"**Predicted Heart Disease Probability:** {prob * 100:.2f}%")
        st.write(f"**Applied Sensitivity Threshold:** {threshold * 100:.0f}%")

    except Exception as e:
        st.error(f" Prediction error: {e}")
