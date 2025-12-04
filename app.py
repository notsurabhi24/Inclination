# app_smooth_map_rails.py
# Streamlit app: smooth real-time marker + railway geometry fetched every 5 minutes (cached)
import streamlit as st
import requests, math, time, itertools
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from typing import List, Tuple

# ---------- CONFIG ----------
GET_URL = st.secrets.get("GET_URL", "https://script.google.com/u/0/home/projects/1RUiJczzFKyM1B_yh7hQoXxp3Jn5AHR7F4RV8m_yUH7GI5Q6fB3L2ti5f/edit")   # <-- put your /exec URL here or set in st.secrets
READ_REFRESH_MS = 2000    # refresh readings every 2 seconds
MAP_REFRESH_SEC = 300     # fetch railway geometry every 5 minutes (300s)
PERMITTED_CANT_MM = 165.0
WARNING_RATIO = 0.8
GAUGE_M = 1.676

st.set_page_config(layout="wide", page_title="Rail Cant Monitor — Rails + Smooth Marker")
st.title("Rail Cant Monitor — Smooth marker + Railway tracks (map refresh every 5 min)")

# auto refresh loop (this causes the script to re-run every READ_REFRESH_MS)
st_autorefresh(interval=READ_REFRESH_MS, limit=0, key="autorefresh")

# ---------- utility functions ----------
def haversine_km(p1: Tuple[float,float], p2: Tuple[float,float]) -> float:
    lat1,lon1 = p1; lat2,lon2 = p2
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2-lat1); dlambda = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

@st.cache_data(ttl=MAP_REFRESH_SEC)
def fetch_railway_geo(lat_min: float, lon_min: float, lat_max: float, lon_max: float):
    """
    Fetch railway ways from Overpass API inside bbox.
    Returns list of polylines (list of (lat,lng)).
    Cached for MAP_REFRESH_SEC seconds to avoid re-fetching repeatedly.
    """
    bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    # Overpass QL: fetch ways with railway tag within bbox, include node coordinates
    query = f"""
    [out:json][timeout:25];
    (
      way["railway"~"rail|light_rail|railway"]({bbox});
    );
    out body;
    >;
    out skel qt;
    """
    url = "https://overpass-api.de/api/interpreter"
    try:
        r = requests.post(url, data=query.strip(), timeout=30)
        r.raise_for_status()
        js = r.json()
        # parse nodes
        nodes = {}
        ways = []
        for el in js.get("elements", []):
            if el.get("type") == "node":
                nodes[el["id"]] = (el["lat"], el["lon"])
        # extract ways (sequence of node ids -> coordinates)
        for el in js.get("elements", []):
            if el.get("type") == "way":
                node_ids = el.get("nodes", [])
                coords = [nodes[nid] for nid in node_ids if nid in nodes]
                if len(coords) >= 2:
                    ways.append(coords)
        # return ways as list of lists of (lat,lng)
        if len(ways) == 0:
            return None
        return ways
    except Exception as e:
        # network error / rate-limited -> return None (caller will fallback)
        return None

def fetch_rows(get_url: str) -> pd.DataFrame:
    r = requests.get(get_url, timeout=10)
    r.raise_for_status()
    arr = r.json()
    return pd.DataFrame(arr)

def compute_interp_pos(past_row, now_row, current_time: datetime):
    # past_row/now_row are pandas Series with timestamp_ist, lat, lng
    try:
        t0 = pd.to_datetime(past_row["timestamp_ist"])
        t1 = pd.to_datetime(now_row["timestamp_ist"])
    except Exception:
        return float(now_row["lat"]), float(now_row["lng"])
    if t1 <= t0:
        return float(now_row["lat"]), float(now_row["lng"])
    frac = (current_time - t0).total_seconds() / max(1.0, (t1 - t0).total_seconds())
    frac = max(0.0, min(1.0, frac))
    lat = float(past_row["lat"]) + (float(now_row["lat"]) - float(past_row["lat"])) * frac
    lng = float(past_row["lng"]) + (float(now_row["lng"]) - float(past_row["lng"])) * frac
    return lat, lng

# ---------- main ----------
# 1) load telemetry rows
try:
    df = fetch_rows(GET_URL)
except Exception as e:
    st.error("Could not fetch telemetry from your Google Sheet. Check GET_URL and Apps Script deployment.")
    st.write("Error:", e)
    st.stop()

if df.empty or len(df) < 2:
    st.info("Waiting for telemetry (need at least 2 rows).")
    st.stop()

# normalize column names and types
df.columns = [c.strip() for c in df.columns]
for c in ["lat","lng","cant_mm","grad_mm","cant_deg","grad_deg"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
if "timestamp_ist" in df.columns:
    df["timestamp_ist"] = pd.to_datetime(df["timestamp_ist"], errors="coerce")

# compute route bounding box from telemetry (expand slightly)
lat_min = float(df["lat"].min()); lat_max = float(df["lat"].max())
lng_min = float(df["lng"].min()); lng_max = float(df["lng"].max())
pad_lat = max(0.01, (lat_max - lat_min) * 0.2 + 0.005)   # ensure small bbox if points are close
pad_lng = max(0.01, (lng_max - lng_min) * 0.2 + 0.005)
lat_min_q, lat_max_q = lat_min - pad_lat, lat_max + pad_lat
lng_min_q, lng_max_q = lng_min - pad_lng, lng_max + pad_lng

# 2) fetch railway geometry (cached for MAP_REFRESH_SEC)
railways = fetch_railway_geo(lat_min_q, lng_min_q, lat_max_q, lng_max_q)

# If Overpass returned nothing, fallback to drawing the logged path as the "track"
if not railways:
    # build single polyline from telemetry path
    path_coords = [(float(r["lat"]), float(r["lng"])) for _, r in df[["lat","lng"]].dropna().iterrows()]
    railways = [path_coords]

# 3) prepare interpolated marker between last two rows for smooth motion
past = df.iloc[-2]
now = df.iloc[-1]
current_time = pd.to_datetime(datetime.now())
interp_lat, interp_lng = compute_interp_pos(past, now, current_time)

# 4) progress calculation (based on logged route start->end)
start = (float(df.iloc[0]["lat"]), float(df.iloc[0]["lng"]))
end = (float(df.iloc[-1]["lat"]), float(df.iloc[-1]["lng"]))
total_km = haversine_km(start, end)
done_km = haversine_km(start, (interp_lat, interp_lng))
pct_done = min(100.0, (done_km / total_km * 100.0) if total_km > 0 else 0.0)

# 5) threshold status
cant_mm = float(now.get("cant_mm") or 0.0)
warn_thresh = WARNING_RATIO * PERMITTED_CANT_MM
status = "GREEN"
if cant_mm >= PERMITTED_CANT_MM:
    status = "RED"
elif cant_mm >= warn_thresh:
    status = "YELLOW"
color_map = {"GREEN":"#2ECC71","YELLOW":"#F1C40F","RED":"#E74C3C"}

# ---------- UI layout ----------
col_map, col_live = st.columns([2,1])

with col_map:
    st.subheader("Map — railway track (refreshed every 5 minutes) + smooth position")
    # prepare map dataframe for st.map: telemetry path + current smooth point
    map_path_df = pd.DataFrame([{"latitude": lat, "longitude": lng} for lat, lng in list(itertools.chain.from_iterable(railways))])
    # drop duplicates (st.map will plot points; path effect is visible by many points)
    map_path_df = map_path_df.drop_duplicates().tail(5000)  # limit to 5k points to keep UI responsive

    # Show track (as points) using st.map - stable and lightweight
    st.map(map_path_df)

    # show smooth marker coordinates and percent
    st.markdown(f"**Smooth marker:** {interp_lat:.6f}, {interp_lng:.6f}")
    st.write(f"Progress: **{pct_done:.2f}%** — {done_km:.3f} km of {total_km:.3f} km")

    # small legend + sample last N path points table
    with st.expander("Show sampled railway polyline preview (first 200 points)"):
        st.dataframe(map_path_df.head(200))

with col_live:
    st.subheader("Live readings & status")
    st.metric("Cant (mm)", f"{cant_mm:.2f}")
    st.metric("Cant (deg)", f"{(math.degrees(math.atan((cant_mm/1000.0)/GAUGE_M))):.3f}°")
    st.metric("Gradient (deg)", f"{(math.degrees(math.atan((float(now.get('grad_mm') or 0.0)/1000.0)/GAUGE_M))):.3f}°")
    st.markdown(f"<div style='display:flex;gap:12px;align-items:center'><div style='width:18px;height:18px;background:{color_map[status]};border-radius:6px'></div><div><b>Status:</b> {status}</div></div>", unsafe_allow_html=True)
    if status == "RED":
        st.error("Threshold exceeded — immediate attention advised.")
    elif status == "YELLOW":
        st.warning("Approaching threshold — monitor closely.")
    else:
        st.success("Within allowed cant limits.")

# History and recent rows (bottom)
st.subheader("History & recent rows")
h = df.tail(1000)
st.line_chart(h[["cant_mm","grad_mm"]].fillna(method="ffill"))
with st.expander("Recent raw rows (latest 200)"):
    st.dataframe(h.tail(200))

st.caption("Railway geometry comes from OpenStreetMap (Overpass API) and is cached for 5 minutes to prevent repeated fetching / UI flashing. If Overpass fails, the logged telemetry path is used as a fallback.")
