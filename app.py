# app.py (Streamlit dashboard - revamped)
import streamlit as st
import requests, math, time
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime

# ---------- CONFIG ----------
# Put your GET_URL into st.secrets["GET_URL"] (recommended) or paste it here:
GET_URL = st.secrets.get("GET_URL", "PASTE_YOUR_GET_URL_HERE")
POLL_INTERVAL_MS = 1000
PERMITTED_CANT_MM = 165.0   # default - change if you have a different requirement
WARNING_RATIO = 0.8
GAUGE_M = 1.676
st.set_page_config(layout="wide", page_title="Rail Cant Monitor Revamp")

st.title("Rail Cant Monitor — Live")

st_autorefresh(interval=POLL_INTERVAL_MS, limit=0, key="autorefresh")

def mm_to_deg(mm, gauge_m=GAUGE_M):
    return math.degrees(math.atan((mm/1000.0)/gauge_m))

def fetch_rows():
    r = requests.get(GET_URL, timeout=8)
    r.raise_for_status()
    arr = r.json()
    return pd.DataFrame(arr)

def reverse_geocode(lat, lon, use_google=False, google_key=None):
    try:
        if use_google and google_key:
            url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={google_key}"
            r = requests.get(url, timeout=6)
            if r.status_code == 200:
                js = r.json()
                if js.get("results"):
                    return js["results"][0]["formatted_address"]
        # default: Nominatim (free, rate-limited)
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=18&addressdetails=1"
        headers = {"User-Agent":"RailCantMonitor/1.0 (youremail@example.com)"}
        r = requests.get(url, headers=headers, timeout=6)
        if r.status_code == 200:
            js = r.json()
            return js.get("display_name") or js.get("name")
    except Exception:
        return None
    return None

# load
try:
    df = fetch_rows()
except Exception as e:
    st.error("Failed fetching telemetry. Check GET_URL and deployment. " + str(e))
    st.stop()

if df.empty:
    st.info("No telemetry yet.")
    st.stop()

# normalize
df.columns = [c.strip() for c in df.columns]
for c in ["lat","lng","cant_mm","grad_mm","cant_deg","grad_deg"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
df['timestamp_ist'] = pd.to_datetime(df['timestamp_ist'], errors='coerce')

latest = df.iloc[-1]
lat = float(latest['lat']); lng = float(latest['lng'])
cant_mm = float(latest.get('cant_mm') or 0.0)
grad_mm = float(latest.get('grad_mm') or 0.0)
cant_deg = float(latest.get('cant_deg') or mm_to_deg(cant_mm))
grad_deg = float(latest.get('grad_deg') or mm_to_deg(grad_mm))

# route progress: first row -> last row
start = (float(df.iloc[0].lat), float(df.iloc[0].lng))
end   = (float(df.iloc[-1].lat), float(df.iloc[-1].lng))
def haversine_km(p1,p2):
    lat1,lon1 = p1; lat2,lon2 = p2
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2-lat1); dlambda = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
total_km = haversine_km(start, end)
done_km = haversine_km(start, (lat,lng))
pct_done = min(100.0, (done_km/total_km*100) if total_km>0 else 0.0)

# reverse geocode (cached)
if "last_loc_coords" not in st.session_state or st.session_state["last_loc_coords"] != (lat,lng):
    st.session_state["last_loc_coords"] = (lat,lng)
    st.session_state["last_loc_name"] = reverse_geocode(lat,lng)
loc_name = st.session_state.get("last_loc_name") or f"{lat:.6f}, {lng:.6f}"

# status color
warn_thresh = WARNING_RATIO * PERMITTED_CANT_MM
status = "GREEN"
if cant_mm >= PERMITTED_CANT_MM:
    status = "RED"
elif cant_mm >= warn_thresh:
    status = "YELLOW"
color_map = {"GREEN":"#2ECC71","YELLOW":"#F1C40F","RED":"#E74C3C"}
color = color_map[status]

# Layout
c1,c2 = st.columns([2,1])
with c1:
    st.subheader("Map — path & animated marker")
    path = [[float(r.lng), float(r.lat)] for _,r in df[['lat','lng']].dropna().iterrows()]
    # for smoother UX, animate a separate marker following an interpolated/smoothed path
    smooth_marker = {"lat": lat, "lng": lng}  # you can implement interpolation here if you want
    deck = pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lng, zoom=14, pitch=35),
        layers=[
            pdk.Layer("PathLayer", data=[{"path": path}], get_path="path", width_scale=20, width_min_pixels=2),
            pdk.Layer("ScatterplotLayer", data=[{"lat": smooth_marker["lat"], "lng": smooth_marker["lng"]}], get_position='[lng, lat]', get_radius=30)
        ],
        map_style='road'
    )
    st.pydeck_chart(deck)
    st.write(f"Progress: **{pct_done:.2f}%** — {done_km:.3f} km of {total_km:.3f} km")
    st.write(f"Current location: **{loc_name}**")

with c2:
    st.subheader("Live readings & threshold")
    st.metric("Cant (mm)", f"{cant_mm:.2f} mm")
    st.metric("Cant (deg)", f"{cant_deg:.3f}°")
    st.metric("Gradient (deg)", f"{grad_deg:.3f}°")
    # color indicator
    st.markdown(f"<div style='display:flex;align-items:center;gap:10px'><div style='width:24px;height:24px;background:{color};border-radius:4px'></div><div><b>Status:</b> {status}</div></div>", unsafe_allow_html=True)
    if status == "RED":
        st.error("Threshold exceeded — take immediate action.")
    elif status == "YELLOW":
        st.warning("Approaching threshold — monitor closely.")
    else:
        st.success("Within allowed cant limits.")

# charts
st.subheader("History")
h = df.tail(1000)
fig = go.Figure()
if 'cant_mm' in h.columns: fig.add_trace(go.Scatter(x=h['timestamp_ist'], y=h['cant_mm'], name='Cant mm'))
if 'grad_mm' in h.columns: fig.add_trace(go.Scatter(x=h['timestamp_ist'], y=h['grad_mm'], name='Grad mm'))
fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Recent raw rows"):
    st.dataframe(h.tail(200))

st.markdown("**Notes:** Threshold default is 165 mm (RDSO/IR-based demo). Adjust `PERMITTED_CANT_MM` for your route. Reverse geocoding uses Nominatim (free but rate-limited); for production use Google Geocoding (API key).")
