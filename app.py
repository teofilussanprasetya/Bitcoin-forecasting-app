import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# ======================
# CONFIG
# ======================
st.set_page_config(
    page_title="Bitcoin Price Forecasting",
    layout="centered"
)

BASE_DIR = Path(__file__).parent

# ======================
# LOAD MODELS
# ======================
@st.cache_resource
def load_models():
    models = {}

    # -------- ARIMAX --------
    try:
        arimax_bundle = pickle.load(open(BASE_DIR / "arimax_model.pkl", "rb"))

        models["ARIMAX"] = {
            "model": arimax_bundle["model"],   # 🔧 FIX DI SINI
            "scaler_exog": pickle.load(open(BASE_DIR / "arimax_scaler_exog.pkl", "rb")),
            "scaler_y": pickle.load(open(BASE_DIR / "arimax_scaler_y.pkl", "rb")),
        }
        st.success("ARIMAX loaded")
    except Exception as e:
        st.error(f"ARIMAX error: {e}")

    # -------- SARIMAX --------
    try:
        sarimax_bundle = pickle.load(open(BASE_DIR / "sarimax_model.pkl", "rb"))

        models["SARIMAX"] = {
            "model": sarimax_bundle["model"],  # 🔧 FIX DI SINI
            "scaler_exog": pickle.load(open(BASE_DIR / "sarimax_scaler_exog.pkl", "rb")),
            "scaler_y": pickle.load(open(BASE_DIR / "sarimax_scaler_y.pkl", "rb")),
        }
        st.success("SARIMAX loaded")
    except Exception as e:
        st.error(f"SARIMAX error: {e}")

    return models


models = load_models()

# ======================
# UI
# ======================
st.title("📈 Bitcoin Price Forecasting")

model_choice = st.selectbox(
    "Select Model",
    list(models.keys())
)

days = st.number_input(
    "Forecast Horizon (Days)",
    min_value=1,
    max_value=60,
    value=14
)

st.caption(
    "Forecast menggunakan pendekatan persistence pada variabel exogenous "
    "(nilai dipertahankan dari data pelatihan terakhir)."
)

# ======================
# FORECAST
# ======================
if st.button("Run Forecast"):
    try:
        bundle = models[model_choice]

        model = bundle["model"]           # SEKARANG SUDAH BENAR
        scaler_exog = bundle["scaler_exog"]
        scaler_y = bundle["scaler_y"]

        # EXOG FUTURE
        exog_future = scaler_exog.inverse_transform(
            np.zeros((days, scaler_exog.n_features_in_))
        )

        forecast_scaled = model.forecast(
            steps=days,
            exog=exog_future
        )

        forecast = scaler_y.inverse_transform(
            np.array(forecast_scaled).reshape(-1, 1)
        ).flatten()

        df = pd.DataFrame({
            "Day": range(1, days + 1),
            "Predicted Bitcoin Price": forecast
        })

        st.subheader("Forecast Result")
        st.line_chart(df.set_index("Day"))
        st.dataframe(df)

    except Exception as e:
        st.error(f"Forecast failed: {e}")
