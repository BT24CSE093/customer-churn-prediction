
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("Customer Churn Prediction")
st.write("Enter customer details to estimate churn risk.")

# Load saved model from the same folder as app.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Customer input form
with st.form("customer_form"):
    st.subheader("Customer Information")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

    tenure = st.slider("Tenure (months)", 0, 72, 12)

    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )
    backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )
    protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )
    support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )
    tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )
    movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=840.0
    )

    submitted = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )

# Make prediction
if submitted:
    customer = pd.DataFrame([{
        "Gender": gender,
        "Senior Citizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure,
        "Phone Service": phone,
        "Multiple Lines": multiple,
        "Internet Service": internet,
        "Online Security": security,
        "Online Backup": backup,
        "Device Protection": protection,
        "Tech Support": support,
        "Streaming TV": tv,
        "Streaming Movies": movies,
        "Contract": contract,
        "Paperless Billing": paperless,
        "Payment Method": payment,
        "Monthly Charges": monthly,
        "Total Charges": total
    }])

    probability = model.predict_proba(customer)[0, 1]
    threshold = 0.40

    st.subheader("Prediction Results")
    st.metric(
        "Estimated Churn Probability",
        f"{probability:.1%}"
    )

    if probability >= threshold:
        st.warning(
            "Higher churn risk. Consider customer retention outreach."
        )
    else:
        st.success(
            "Lower churn risk based on this model."
        )

    st.caption(
        "This is a model estimate, not a guarantee. "
        "The 40% threshold prioritizes identifying more potential churners."
    )
