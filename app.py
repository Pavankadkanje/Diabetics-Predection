
import pickle
import pandas as pd
import streamlit as st

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# =========================
# LOAD MODEL & SCALER
# =========================
with open("diabetes_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# =========================
# TITLE
# =========================
st.title("🩺 Diabetes Prediction System")

st.markdown(
    "Enter the patient's clinical metrics below to predict diabetes status."
)

st.header("Patient Metrics Input")

# =========================
# INPUT SECTION
# =========================
col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "Glucose Level (mg/dL)",
        min_value=0,
        max_value=300,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure (mm Hg)",
        min_value=0,
        max_value=200,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:

    insulin = st.number_input(
        "Insulin Level (mu U/ml)",
        min_value=0,
        max_value=900,
        value=80
    )

    bmi = st.number_input(
        "BMI (Body Mass Index)",
        min_value=0.0,
        max_value=70.0,
        value=32.0,
        step=0.1
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=33,
        step=1
    )

# =========================
# PREDICTION BUTTON
# =========================
if st.button("Predict Diabetes Status", type="primary"):

    # Validation
    if glucose == 0 or bmi == 0:
        st.warning("Please enter valid medical values.")
        st.stop()

    # Feature names
    feature_names = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    # Input dataframe
    input_data = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]]

    input_df = pd.DataFrame(
        input_data,
        columns=feature_names
    )

    try:

        # Scale input
        scaled_data = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(scaled_data)

        st.markdown("---")

        # Result
        if prediction[0] == 1:
            st.error(
                "### ⚠️ Result: The person is likely diabetic."
            )
        else:
            st.success(
                "### ✅ Result: The person is not diabetic."
            )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

