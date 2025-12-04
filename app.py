# app_smooth_map.py
import streamlit as st
import requests, math, time
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

GET_URL = st.secrets.get("GET_URL", "https://script.google.com/macros/s/AKfycbz-NRGeJRi-T7Jmf7mtjYqTOWvoZdqqHmYLIfVjD6IdydPmUbXkY52vOqpbHq3IV-zTSg/exec")
POLL_INTERVAL_MS = 1000
PERMITTED_CANT_MM = 165.0
WARNING_RATIO = 0.8
GAUGE_M = 1.676

st.set_page_config(layout="wide", page_title="Rail Cant Monitor - Smooth")
st.title("Rail Cant Monitor — Smooth real-time")

st_autorefresh(interval=POLL_INTERVAL_MS, limit=0, key="autorefresh")

def haversine_km(p1,p2):
    lat1,lon1 = p1; lat2,lon2 = p2
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2-lat1); dlambda = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def fetch_rows():
    r = requests.get(GET_URL, timeout=8); r.raise_for_status(); arr = r.json(); return pd.DataFrame(arr)

try:
    df = fetch_rows()
except Exception as e:
    st.error("Could not fetch telemetry: " + str(e)); st.stop()

if df.empty or len(df) < 2:
    st.info("Waiting for telemetry (need >=2 rows)"); st.stop()

df.columns = [c.strip() for c in df.columns]
for c in ["lat","lng","cant_mm","grad_mm","cant_deg","grad_deg"]:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
df['timestamp_ist'] = pd.to_datetime(df['timestamp_ist'], errors='coerce')

start = (float(df.iloc[0].lat), float(df.iloc[0].lng))
end   = (float(df.iloc[-1].lat), float(df.iloc[-1].lng))
total_km = haversine_km(start,end)

# interpolate between last two rows for smooth position
past = df.iloc[-2]; now = df.iloc[-1]
t0 = pd.to_datetime(past['timestamp_ist']); t1 = pd.to_datetime(now['timestamp_ist'])
current_time = pd.to_datetime(datetime.now())
if t1 <= t0:
    interp_lat, interp_lng = float(now['lat']), float(now['lng'])
else:
    frac = (current_time - t0).total_seconds() / max(1.0,(t1 - t0).total_seconds())
    frac = max(0.0, min(1.0, frac))
    interp_lat = float(past['lat']) + (float(now['lat']) - float(past['lat']))*frac
    interp_lng = float(past['lng']) + (float(now['lng']) - float(past['lng']))*frac

done_km = haversine_km(start, (interp_lat, interp_lng))
pct_done = min(100.0, (done_km / total_km * 100.0) if total_km>0 else 0.0)

cant_mm = float(now.get('cant_mm') or 0.0)
grad_mm = float(now.get('grad_mm') or 0.0)
warn_thresh = WARNING_RATIO * PERMITTED_CANT_MM
status = "GREEN"
if cant_mm >= PERMITTED_CANT_MM: status = "RED"
elif cant_mm >= warn_thresh: status = "YELLOW"
color_map = {"GREEN":"#2ECC71","YELLOW":"#F1C40F","RED":"#E74C3C"}

col_map, col_metrics = st.columns([2,1])
with col_map:
    st.subheader("Map (latest path, smooth marker)")
    map_df = df[['lat','lng']].dropna().tail(1000).rename(columns={'lat':'latitude','lng':'longitude'})
    st.map(map_df)
    st.markdown(f"**Smooth marker:** {interp_lat:.6f}, {interp_lng:.6f}")
    st.write(f"Progress: **{pct_done:.2f}%** — {done_km:.3f} km of {total_km:.3f} km")

with col_metrics:
    st.subheader("Live")
    st.metric("Cant (mm)", f"{cant_mm:.2f}")
    st.metric("Cant (deg)", f"{(math.degrees(math.atan((cant_mm/1000.0)/GAUGE_M))):.3f}°")
    st.metric("Gradient (deg)", f"{(math.degrees(math.atan((grad_mm/1000.0)/GAUGE_M))):.3f}°")
    st.markdown(f"<div style='display:flex;align-items:center;gap:10px'><div style='width:24px;height:24px;background:{color_map[status]};border-radius:6px'></div><div><b>Status:</b> {status}</div></div>", unsafe_allow_html=True)
    if status == "RED": st.error("Threshold exceeded!")
    elif status == "YELLOW": st.warning("Approaching threshold.")
    else: st.success("Within allowed limits.")

st.subheader("History")
h = df.tail(1000)
st.line_chart(h[['cant_mm','grad_mm']].fillna(method='ffill'))

with st.expander("Recent rows"):
    st.dataframe(h.tail(200))
