import streamlit as st
import pandas as pd
import joblib

st.title("Developer Analytics: Stress & Productivity Predictor")

LR_model = joblib.load('stress_model.pkl')
dt_model = joblib.load('dt_model.pkl')
scaler_stress = joblib.load('scaler.pkl')
scaler_prod = joblib.load('scaler_prod.pkl')
features_stress = joblib.load('features.pkl')
features_prod = joblib.load('features_prod.pkl')

st.header("Input Developer Metrics")
input_data = {}
for feature in features_stress:
    input_data[feature] = st.number_input(feature, value=0.0)

df_input = pd.DataFrame([input_data])

st.subheader("Stress Prediction")
if st.button("Predict Stress Level"):
    try:
        df_scaled = pd.DataFrame(scaler_stress.transform(df_input), columns=features_stress)
        stress_pred = LR_model.predict(df_scaled)
        st.success(f"Predicted Stress Level: {stress_pred[0]:.2f}")
    except Exception as e:
        st.error(f"Error in Stress Prediction: {e}")

st.subheader("Productivity Prediction")
if st.button("Predict Productivity Level"):
    try:
        df_scaled_prod = pd.DataFrame(scaler_prod.transform(df_input[features_prod]), columns=features_prod)
        prod_pred = dt_model.predict(df_scaled_prod)
        st.success(f"Predicted Productivity Level: {prod_pred[0]}")
    except Exception as e:
        st.error(f"Error in Productivity Prediction: {e}")
