import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Load exported preprocessors, dropdown options, and trained model
@st.cache_resource
def load_artifacts():
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("dropdown_options.pkl", "rb") as f:
        dropdown_options = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return encoder, scaler, dropdown_options, model

try:
    encoder, scaler, dropdown_options, model = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Missing file: {e.filename}. Make sure encoder.pkl, scaler.pkl, "
        "dropdown_options.pkl, and model.pkl are in the same folder as app.py."
    )
    st.stop()

# App Header
st.title("🚗 Used Car Price Prediction")
st.write("Nayi car ki details enter karein aur estimated resale price check karein.")
st.divider()

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        company = st.selectbox(
            "Car Company",
            options=dropdown_options["companies"]
        )
        # Filter car names matching the selected company (fallback to all if none match)
        filtered_names = [
            n for n in dropdown_options["names"]
            if n.lower().startswith(company.lower())
        ]
        name_options = filtered_names if filtered_names else dropdown_options["names"]

        name = st.selectbox(
            "Car Model (First 3 Words)",
            options=name_options
        )
        fuel_type = st.selectbox(
            "Fuel Type",
            options=dropdown_options["fuel_types"]
        )

    with col2:
        year = st.number_input(
            "Manufacturing Year",
            min_value=1995,
            max_value=2026,
            value=2016,
            step=1
        )
        kms_driven = st.number_input(
            "Kilometers Driven (min 1,000 km)",
            min_value=1000,
            max_value=500000,
            value=35000,
            step=500
        )

    submitted = st.form_submit_button("Predict Price 💰", use_container_width=True)

# Prediction Pipeline (Matches notebook Step 2 & Step 4 exactly)
if submitted:
    # 1. Clean car name to first 3 words (same as training pipeline)
    clean_name = " ".join(name.split(" ")[:3])

    # 2. Build input DataFrame with exact training column order
    input_df = pd.DataFrame([{
        "name": clean_name,
        "company": company,
        "year": int(year),
        "kms_driven": int(kms_driven),
        "fuel_type": fuel_type
    }])

    cat_cols = ["name", "company", "fuel_type"]
    num_cols = ["year", "kms_driven"]

    # 3. Transform categorical & numerical features separately
    X_cat = encoder.transform(input_df[cat_cols])
    X_num = scaler.transform(input_df[num_cols])

    # 4. Combine horizontally in exact training order: [cat_cols, num_cols]
    X_processed = np.hstack((X_cat, X_num))

    # 5. Predict in log-space and reverse using expm1
    pred_log = model.predict(X_processed)
    predicted_price = float(np.expm1(pred_log)[0])

    st.success(f"### Estimated Car Price: ₹ {predicted_price:,.0f}")

    # Summary of selected inputs
    with st.expander("Input Summary"):
        st.dataframe(input_df, use_container_width=True)