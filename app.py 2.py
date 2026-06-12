import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
with open("credit_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Credit Risk Assessment System")

st.write("Enter applicant details")

loan_amnt = st.number_input("Loan Amount", value=5000.0)
int_rate = st.number_input("Interest Rate", value=10.0)
installment = st.number_input("Installment", value=200.0)
annual_inc = st.number_input("Annual Income", value=50000.0)
dti = st.number_input("Debt-To-Income Ratio", value=15.0)

if st.button("Predict Risk"):

    data = pd.DataFrame({
        'loan_amnt': [loan_amnt],
        'int_rate': [int_rate],
        'installment': [installment],
        'annual_inc': [annual_inc],
        'dti': [dti]
    })

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]

    st.subheader("Prediction Result")

    st.write(f"Risk Probability: {probability:.2%}")

    if prediction == 1:
        st.error("High Credit Risk")
    else:
        st.success("Low Credit Risk")