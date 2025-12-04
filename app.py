# app_safe_minimal.py
import streamlit as st
import requests
import math
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from typing import Tuple

# ---------- CONFIG ----------
GET_URL = st.secrets.get("GET_URL", "https://script.google.com/macros/s/AKfycbz-NRGeJRi-T7Jmf7mtjYqTOWvoZdqqHmYLIfVjD6IdydPmUbXkY52vOqpbHq3IV-zTSg/exec")  # <-- put your /exec URL here or set in .streamlit/secrets.toml
READ_REFRESH_MS = 2000       # readings refresh
MAP_REFRESH_SEC = 300        # if you keep caching functions, they can be added later
PERMITTED_CANT_MM = 165.0
WARNING_RATIO = 0.8
GAUGE_M = 1.676

st.set_page_config(layout="wide", page_title="Rail Cant Monitor — Safe Minimal")
st.title("Rail Cant Monitor — Safe Minimal")

# auto-refresh every READ_REFRESH_MS ms
st_autorefresh(interval=READ_REFRESH_MS, limit=0, key="autorefresh")

# ---------- helper functions ----------
def haversine_km(p1: Tuple[float,float], p2: Tuple[float,float]) -> float:
    lat1,lon1 = p1; lat2,lon2 = p2
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2-lat1); dlambda = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def fetch_rows_defensive(url: str):
    """Fetch GET_URL and return a pandas DataFrame.
       If the response is not JSON or cannot be parsed, show raw content in the app and stop.
    """
    try:
        r = requests.get(url, timeout=12)
    except Exception as e:
        st.error("Network error contacting GET_URL: " + str(e))
        st.stop()

    content_type = r.headers.get("content-type","")
    # If it's not JSON, show the response so the user/developer can see the problem
    if 'application/json' not in content_type.lower():
        st.error(f"GET_URL returned non-JSON (status {r.status_code}, content-type: {content_type})")
        st.markdown("**Raw server response (first 2000 chars):**")
        st.code(r.text[:2000])
        st.stop()

    # Try parse JSON
    try:
        arr = r.json()
    except Exception as e:
        st.error("Failed to parse JSON from GET_URL: " + str(e))
        st.markdown("**Raw server response (first 2000 chars):**")
        st.code(r.text[:2000])
        st.stop()

    # Ensure we have a list (array)
    if not isinstance(arr, list):
        st.error("GET_URL returned JSON but not an array. Showing raw JSON.")
        st.code(str(arr)[:2000])
        st.stop()

    # Convert to DataFrame safely
    df = pd.DataFrame(arr)
    return df

def safe_to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def safe_interp_pos(past_row, now_row, current_time):
    """Interpolate between past_row and now_row when safe, otherwise fallback to now_row coords."""
    try:
        # must have timestamps
        t0 = pd.to_datetime(past_row.get("timestamp_ist", None))
        t1 = pd.to_datetime(now_row.get("timestamp_ist", None))
        if pd.isna(t0) or pd.isna(t1) or t1 <= t0:
            return safe_to_float(now_row.get("lat")), safe_to_float(now_row.get("lng"))
        # lat/lng must exist
        lat0 = safe_to_float(past_row.get("lat"), None)
        lat1 = safe_to_float(now_row.get("lat"), None)
        lng0 = safe_to_float(past_row.get("lng"), None)
        lng1 = safe_to_float(now_row.get("lng"), None)
        if lat0 is None or lat1 is None or lng0 is None or lng1 is None:
            return safe_to_float(now_row.get("lat")), safe_to_float(now_row.get("lng"))
        frac = (current_time - t0).total_seconds() / max(1.0, (t1 - t0).total_seconds())
        frac = max(0.0, min(1.0, frac))
        lat = lat0 + (lat1 - lat0) * frac
        lng = lng0 + (lng1 - lng0) * frac
        return lat, lng
    except Exception:
        # anything goes wrong -> fallback to latest position
        return safe_to_float(now_row.get("lat")), safe_to_float(now_row.get("lng"))

# ---------- main ----------
df = fetch_rows_defensive(GET_URL)

# minimum safe checks
if df.empty:
    st.info("No telemetry rows in sheet yet.")
    st.stop()

# Normalize column names
df.columns = [c.strip() for c in df.columns]

# Coerce numeric columns safely where present
for c in ["lat","lng","cant_mm","grad_mm","cant_deg","grad_deg"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Ensure timestamp parsed if present
if "timestamp_ist" in df.columns:
    df["timestamp_ist"] = pd.to_datetime(df["timestamp_ist"], errors="coerce")

# Determine latest row (use last valid row)
now_row = df.iloc[-1].to_dict()
past_row = df.iloc[-2].to_dict() if len(df) >= 2 else None

current_time = pd.to_datetime(datetime.now())

# Attempt safe interpolation only if past_row exists
if past_row is not None:
    interp_lat, interp_lng = safe_interp_pos(past_row, now_row, current_time)
else:
    interp_lat = safe_to_float(now_row.get("lat"))
    interp_lng = safe_to_float(now_row.get("lng"))

# UI layout (simple)
col_map, col_metrics = st.columns([2,1])

with col_map:
    st.subheader("Map (latest telemetry)")
    # Use st.map with recent path for context (if lat/lng present)
    if "lat" in df.columns and "lng" in df.columns:
        map_df = df[["lat","lng"]].dropna().rename(columns={"lat":"latitude", "lng":"longitude"}).tail(1000)
        if not map_df.empty:
            st.map(map_df)
        else:
            st.info("Telemetry present but no valid lat/lng values yet.")
    else:
        st.info("No lat/lng columns present in telemetry yet.")
    st.markdown(f"**Current (smoothed) position:** {interp_lat:.6f}, {interp_lng:.6f}")

with col_metrics:
    st.subheader("Live readings")
    cant_val = safe_to_float(now_row.get("cant_mm", 0.0))
    grad_val = safe_to_float(now_row.get("grad_mm", 0.0))
    st.metric("Cant (mm)", f"{cant_val:.2f}")
    st.metric("Gradient (mm)", f"{grad_val:.2f}")
    # status
    warn_thresh = WARNING_RATIO * PERMITTED_CANT_MM
    status = "GREEN"
    if cant_val >= PERMITTED_CANT_MM:
        status = "RED"
    elif cant_val >= warn_thresh:
        status = "YELLOW"
    if status == "RED":
        st.error("Threshold exceeded — RED")
    elif status == "YELLOW":
        st.warning("Approaching threshold — YELLOW")
    else:
        st.success("Within limits — GREEN")

st.subheader("Recent rows (latest 200)")
# show a safe subset of columns
show_cols = [c for c in ["timestamp_ist","lat","lng","cant_mm","grad_mm","notes"] if c in df.columns]
st.dataframe(df[show_cols].tail(200))
