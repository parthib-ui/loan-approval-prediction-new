import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")
st.title("🏦 Loan Approval Prediction App")
st.write("### Fill in the applicant details to predict the loan approval status:")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encode inputs same as training
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
dependents = 3 if dependents == "3+" else int(dependents)
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# Final input array
input_data = np.array([[gender, married, dependents, education, self_employed,
                        applicant_income, coapplicant_income, loan_amount,
                        loan_amount_term, credit_history, property_area]])

# Scale inputs
scaled_input = scaler.transform(input_data)

# Prediction
if st.button("Check Loan Status"):
    prediction = model.predict(scaled_input)
    if prediction[0] == 1:
        st.success("✅ Loan is likely to be Approved!")
    else:
        st.error("❌ Loan is likely to be Rejected.")
