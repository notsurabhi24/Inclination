# app.py (revamped)
import streamlit as st
import requests, math, time
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# ---------- CONFIG ----------
GET_URL = st.secrets.get("GET_URL", "https://script.google.com/u/0/home/projects/1VJzysk2zX_5PxNPQ4iHyFds3XZSeyVnTnRrep7Q6GLhEOmZ0NPbyMprD/edit")
POLL_INTERVAL_MS = 1000
PERMITTED_CANT_MM = 165.0   # RDSO/IR typical permit (mm)
WARNING_RATIO = 0.8         # yellow when > 80% of permitted
GAUGE_M = 1.676             # BG gauge in meters
st.set_page_config(layout="wide", page_title="Rail Cant Monitor Revamp")

# Auto refresh
st_autorefresh(interval=POLL_INTERVAL_MS, limit=0, key="autorefresh")

def mm_to_deg(mm, gauge_m=GAUGE_M):
    return math.degrees(math.atan((mm/1000.0)/gauge_m))

def fetch_rows():
    r = requests.get(GET_URL, timeout=8)
    r.raise_for_status()
    arr = r.json()
    return pd.DataFrame(arr)

def reverse_geocode(lat, lon):
    # use Nominatim for demo; respect rate limits in production
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=18&addressdetails=1"
        headers = {"User-Agent":"RailCantMonitor/1.0 (email@example.com)"}
        r = requests.get(url, headers=headers, timeout=6)
        if r.status_code == 200:
            js = r.json()
            display = js.get("display_name") or js.get("name")
            return display
    except Exception:
        return None
    return None

# Load data
try:
    df = fetch_rows()
except Exception as e:
    st.error(f"Could not fetch telemetry: {e}")
    st.stop()

if df.empty:
    st.info("No telemetry yet.")
    st.stop()

# Normalize & types
df.columns = [c.strip() for c in df.columns]
for c in ["lat","lng","cant_mm","grad_mm","cant_deg","grad_deg"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Latest reading
latest = df.iloc[-1]
lat = latest.get("lat")
lng = latest.get("lng")
cant_mm = float(latest.get("cant_mm") or 0.0)
grad_mm = float(latest.get("grad_mm") or 0.0)
cant_deg = float(latest.get("cant_deg") or mm_to_deg(cant_mm))
grad_deg = float(latest.get("grad_deg") or mm_to_deg(grad_mm))

# Progress: compute percent along path if start & end appear in sheet header or first/last points
# We'll assume the path is from first row to last row in sheet (simulator uses A->B).
start = (float(df.iloc[0].lat), float(df.iloc[0].lng))
end   = (float(df.iloc[-1].lat), float(df.iloc[-1].lng)) if len(df)>1 else start

def haversine_km(p1,p2):
    lat1,lon1 = p1; lat2,lon2 = p2
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2-lat1); dlambda = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

total_km = haversine_km(start, end)
done_km = haversine_km(start, (lat, lng))
pct_done = min(100.0, (done_km/total_km*100) if total_km>0 else 0.0)

# Reverse geocode current position (cache short-term in session_state)
if "last_loc" not in st.session_state or st.session_state.get("last_loc_coords") != (lat,lng):
    st.session_state["last_loc_coords"] = (lat,lng)
    st.session_state["last_loc_name"] = reverse_geocode(lat,lng)

loc_name = st.session_state.get("last_loc_name") or f"{lat:.6f}, {lng:.6f}"

# Threshold status
warn_thresh_mm = WARNING_RATIO * PERMITTED_CANT_MM
status = "GREEN"
if cant_mm >= PERMITTED_CANT_MM:
    status = "RED"
elif cant_mm >= warn_thresh_mm:
    status = "YELLOW"

def status_color(s):
    return {"GREEN":"#2ECC71","YELLOW":"#F1C40F","RED":"#E74C3C"}[s]

# UI layout
c1,c2 = st.columns([2,1])
with c1:
    st.subheader("Map & Progress")
    # map path: lat/lng history
    path = [[float(r.lng), float(r.lat)] for _,r in df[['lat','lng']].dropna().iterrows()]
    deck = pdk.Deck(
        initial_view_state = pdk.ViewState(latitude=lat, longitude=lng, zoom=14, pitch=30),
        layers = [
            pdk.Layer("ScatterplotLayer", data=[{"lat":lat,"lng":lng}], get_position='[lng, lat]', get_radius=30),
            pdk.Layer("PathLayer", data=[{"path":path}], get_path="path", width_scale=20, width_min_pixels=2)
        ],
        map_style='road'
    )
    st.pydeck_chart(deck)
    st.write(f"Progress: **{pct_done:.2f}%** — {done_km:.3f} km of {total_km:.3f} km")
    st.write(f"Current location: **{loc_name}**")

with c2:
    st.subheader("Live status")
    st.metric("Cant (mm)", f"{cant_mm:.2f} mm", delta=None)
    st.metric("Cant (deg)", f"{cant_deg:.3f}°", delta=None)
    st.metric("Gradient (deg)", f"{grad_deg:.3f}°", delta=None)

    # BIG status button
    colb = st.empty()
    color = status_color(status)
    html = f"""
    <div style='display:flex;align-items:center;gap:10px'>
      <div style='width:20px;height:20px;background:{color};border-radius:4px'></div>
      <div><b>Status:</b> {status}</div>
    </div>
    """
    colb.markdown(html, unsafe_allow_html=True)

    if status == "RED":
        st.error("Threshold exceeded! Immediate check required.")
    elif status == "YELLOW":
        st.warning("Approaching allowed cant. Monitor closely.")
    else:
        st.success("Within allowed cant limits.")

# historical charts
st.subheader("History")
df['timestamp_ist'] = pd.to_datetime(df['timestamp_ist'], errors='coerce')
h = df.tail(500)
fig = go.Figure()
if 'cant_mm' in h.columns:
    fig.add_trace(go.Scatter(x=h['timestamp_ist'], y=h['cant_mm'], name='Cant (mm)'))
if 'grad_mm' in h.columns:
    fig.add_trace(go.Scatter(x=h['timestamp_ist'], y=h['grad_mm'], name='Grad (mm)'))
fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
st.plotly_chart(fig, use_container_width=True)

# recent table
with st.expander("Recent rows"):
    st.dataframe(h.tail(200))

# admin: explain threshold values & source
st.markdown("""
**Threshold notes:**  
We use `PERMITTED_CANT_MM = 165 mm` (typical RDSO/IR guidance) and flag **YELLOW** when > 80% of that value, **RED** when above permitted. These numbers are for demonstration and must be adapted to the exact route classification and official documentation. Sources: RDSO / Indian Railways track manuals.  
""")
