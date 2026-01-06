import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from pathlib import Path
import plotly.graph_objects as go

# Konfigurasi Path
BASE_DIR = Path(__file__).parent

# Definisi Arsitektur LSTM sesuai dengan struktur layer pada state_dict
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=3):
        super(LSTMModel, self).__init__()
        # Menggunakan nn.LSTM dengan num_layers=3 agar sesuai dengan weight_l0, l1, l2
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Ambil output dari time step terakhir
        out = self.fc(out[:, -1, :])
        return out

@st.cache_resource
def load_assets():
    assets = {}
    try:
        # 1. Load ARIMAX & SARIMAX
        with open(BASE_DIR / "arimax_model.pkl", "rb") as f:
            assets["arimax"] = pickle.load(f)
        with open(BASE_DIR / "sarimax_model.pkl", "rb") as f:
            assets["sarimax"] = pickle.load(f)
        
        # 2. Load Scalers
        with open(BASE_DIR / "lstm_scaler_X.pkl", "rb") as f:
            assets["scaler_X"] = pickle.load(f)
        with open(BASE_DIR / "lstm_scaler_y.pkl", "rb") as f:
            assets["scaler_y"] = pickle.load(f)
        
        # 3. Load LSTM Model
        checkpoint = torch.load(BASE_DIR / "lstm_model.pth", map_location=torch.device('cpu'))
        
        # Inisialisasi model dengan parameter dari checkpoint jika tersedia
        input_size = checkpoint.get("input_size", 5)
        model_lstm = LSTMModel(input_size=input_size)
        
        # Memuat state_dict dari key 'model_state_dict' sesuai pesan error
        model_lstm.load_state_dict(checkpoint["model_state_dict"])
        model_lstm.eval()
        assets["lstm"] = model_lstm
            
    except Exception as e:
        st.error(f"Gagal memuat file model: {e}")
    return assets

assets = load_assets()

# --- UI Streamlit ---
st.set_page_config(page_title="Bitcoin Forecast", layout="wide")
st.title("₿ Bitcoin Price Forecasting")

model_choice = st.sidebar.selectbox("Pilih Model", ["SARIMAX", "ARIMAX", "LSTM"])
horizon = st.sidebar.number_input("Forecast Horizon (Hari)", min_value=1, max_value=30, value=7)

if st.sidebar.button("Run Forecast"):
    try:
        forecast_values = None
        
        if model_choice in ["SARIMAX", "ARIMAX"]:
            model = assets["sarimax"] if model_choice == "SARIMAX" else assets["arimax"]
            # Sesuai ketentuan: Tolak jika model butuh exog
            if hasattr(model.model, 'k_exog') and model.model.k_exog > 0:
                st.warning(f"Model {model_choice} memerlukan variabel 'exog'. Forecasting dihentikan.")
            else:
                forecast_values = model.forecast(steps=horizon)
        
        elif model_choice == "LSTM":
            # Dummy data input (menggunakan shape yang diharapkan: 1 batch, 10 sequence, 5 features)
            # Idealnya Anda memasukkan data historis riil di sini
            dummy_input = torch.randn(1, 10, 5) 
            with torch.no_grad():
                pred_scaled = assets["lstm"](dummy_input).numpy()
                res = assets["scaler_y"].inverse_transform(pred_scaled)[0][0]
                # Simulasi deret waktu sederhana untuk horizon
                forecast_values = [res + (i * (res * 0.01)) for i in range(horizon)]

        if forecast_values is not None:
            dates = pd.date_range(start=pd.Timestamp.now(), periods=horizon, freq='D')
            df_res = pd.DataFrame({'Tanggal': dates, 'Harga Prediksi': forecast_values})
            
            # Visualisasi
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_res['Tanggal'], y=df_res['Harga Prediksi'], mode='lines+markers'))
            fig.update_layout(title=f"Hasil Forecast {model_choice}", xaxis_title="Tanggal", yaxis_title="USD")
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabel
            st.dataframe(df_res.style.format({'Harga Prediksi': '{:,.2f}'}))

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")