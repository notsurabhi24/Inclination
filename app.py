import streamlit as st
import pandas as pd
import time
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Railway Dashboard", layout="wide")

CSV_URL = "https://docs.google.com/spreadsheets/d/1I4w_Ldk4RdTKK7AfkLFkO7AiXeUCc9ud24EF8TRz1ig/export?format=csv"

# ---- Refresh Telemetry Every 2 Seconds ----
@st.cache_data(ttl=2)
def load_data():
    return pd.read_csv(CSV_URL)

df = load_data()

if df.empty:
    st.warning("Sheet empty...")
    st.stop()

last = df.iloc[-1]

col1, col2 = st.columns([1,2])

with col1:
    st.header("🚂 Live IMU Data")
    st.write("Timestamp:", last["timestamp"])
    st.metric("Channel", last["channel"])
    st.metric("acc_x", last["acc_x"])
    st.metric("acc_y", last["acc_y"])
    st.metric("acc_z", last["acc_z"])
    st.metric("gyro_x", last["gyro_x"])
    st.metric("gyro_y", last["gyro_y"])
    st.metric("gyro_z", last["gyro_z"])
    st.metric("Roll", last["roll"])
    st.metric("Pitch", last["pitch"])

with col2:
    st.header("📉 IMU Trend (last 200 points)")
    st.line_chart(df[["acc_x", "acc_y", "acc_z"]].tail(200))

st.success("Dashboard running without JSON or AppScript errors.")
