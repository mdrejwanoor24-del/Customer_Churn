import streamlit as st
import joblib
import pandas as pd
 
# Load model (pipeline)
model = joblib.load("best_model.pkl")
 
st.set_page_config(page_title="Churn Prediction App")
 
st.title("📊 Customer Churn Prediction")
 
st.write("Enter customer details below:")
 
# ---------------- INPUT FIELDS ---------------- #
 
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72)
 
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
 
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
 
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
 
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
 
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
 
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
 
PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
 
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)
 
# ---------------- CREATE DATAFRAME ---------------- #
 
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [SeniorCitizen],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "tenure": [tenure],
    "PhoneService": [PhoneService],
    "MultipleLines": [MultipleLines],
    "InternetService": [InternetService],
    "OnlineSecurity": [OnlineSecurity],
    "OnlineBackup": [OnlineBackup],
    "DeviceProtection": [DeviceProtection],
    "TechSupport": [TechSupport],
    "StreamingTV": [StreamingTV],
    "StreamingMovies": [StreamingMovies],
    "Contract": [Contract],
    "PaperlessBilling": [PaperlessBilling],
    "PaymentMethod": [PaymentMethod],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges]
})
 
# ---------------- PREDICTION ---------------- #
 
if st.button("Predict Churn"):
 
    prediction = model.predict(input_data)[0]
 
    # Optional probability
    prob = model.predict_proba(input_data)[0][1]
 
    if prediction == 1:
        st.error(f"❌ Customer is likely to churn (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer is not likely to churn (Probability: {prob:.2f})")