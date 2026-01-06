import streamlit as st
from src.pipeline import run_daily_pipeline
import pandas as pd

st.title("📈 Bitcoin Forecasting Dashboard")

if st.button("Run Today's Pipeline"):
    df = run_daily_pipeline()
    st.success("Pipeline Selesai!")
    st.dataframe(df)

# Show sentiment of today if exists
try:
    today_df = pd.read_csv("final_data(1).csv")
    st.subheader("📊 Today's Sentiment")
    st.dataframe(today_df)
except:
    st.info("Pipeline belum dijalankan hari ini.")

