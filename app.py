import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load and Train Model (once)
# -----------------------------
@st.cache_resource
def load_model():
    # Load dataset
    data = pd.read_csv("flood_risk_dataset.csv")

    # Select only required features
    features = ["Rainfall (mm)", "Temperature (°C)", "Water Level (m)"]
    X = data[features]
    y = data["Flood Occurred"]

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

model = load_model()

# -----------------------------
# Prediction Function
# -----------------------------
def predict_flood(rainfall, temperature, water_level):
    input_data = pd.DataFrame([{
        "Rainfall (mm)": rainfall,
        "Temperature (°C)": temperature,
        "Water Level (m)": water_level
    }])
    prediction = model.predict(input_data)[0]
    return "Yes" if prediction == 1 else "No"

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Flood Risk Prediction", layout="centered")

st.title("🌊 Flood Risk Prediction Tool")

st.write("Select the values below to check if there is a flood risk.")

# Sliders for inputs
rainfall = st.slider("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0, step=1.0)
temperature = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
water_level = st.slider("Water Level (m)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)


# Prediction button
if st.button("Predict Flood Risk"):
    result = predict_flood(rainfall, temperature, water_level)
    st.success(f"🌧️ Flood Risk: **{result}**")
