# app_fixed_final.py
import streamlit as st
import requests, math, itertools, time
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from typing import Tuple

# ------- CONFIG -------
GET_URL = st.secrets.get("GET_URL", "https://script.google.com/macros/s/AKfycbz-NRGeJRi-T7Jmf7mtjYqTOWvoZdqqHmYLIfVjD6IdydPmUbXkY52vOqpbHq3IV-zTSg/exec")  # <-- put your /exec URL here or set in .streamlit/secrets.toml
READ_REFRESH_MS = 2000          # refresh readings every 2 seconds
MAP_REFRESH_SEC = 300           # refresh heavy railway geometry every 5 minutes
PERMITTED_CANT_MM = 165.0
WARNING_RATIO = 0.8
GAUGE_M = 1.676

st.set_page_config(layout="wide", page_title="Rail Cant Monitor (fixed)", initial_sidebar_state="collapsed")
st.title("Rail Cant Monitor — Stable UI (map refresh cached)")

# lightweight auto refresh
st_autorefresh(interval=READ_REFRESH_MS, limit=0, key="autorefresh")

# ---------- helpers ----------
def haversine_km(p1: Tuple[float,float], p2: Tuple[float,float]) -> float:
    lat1,lon1 = p1; lat2,lon2 = p2
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2-lat1); dlambda = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

@st.cache_data(ttl=MAP_REFRESH_SEC)
def fetch_railway_geo_cached(lat_min, lon_min, lat_max, lon_max):
    """
    Query Overpass for railway ways inside bbox. Cached for MAP_REFRESH_SEC seconds.
    Returns list of polylines (list of (lat,lng)) or None on failure.
    """
    bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    q = f"""
    [out:json][timeout:25];
    (
      way["railway"~"rail|light_rail|railway"]({bbox});
    );
    out body;
    >;
    out skel qt;
    """
    try:
        r = requests.post("https://overpass-api.de/api/interpreter", data=q.strip(), timeout=25)
        r.raise_for_status()
        js = r.json()
        nodes = {}
        ways = []
        for el in js.get("elements", []):
            if el.get("type") == "node":
                nodes[el["id"]] = (el["lat"], el["lon"])
        for el in js.get("elements", []):
            if el.get("type") == "way":
                node_ids = el.get("nodes", [])
                coords = [nodes[nid] for nid in node_ids if nid in nodes]
                if len(coords) >= 2:
                    ways.append(coords)
        return ways if ways else None
    except Exception:
        return None

def fetch_rows_safe(get_url: str) -> pd.DataFrame:
    """Fetch telemetry JSON safely; on non-JSON show raw response in the app and stop."""
    try:
        r = requests.get(get_url, timeout=12)
    except Exception as e:
        st.error("Network error when contacting GET_URL: " + str(e))
        st.stop()
    ct = r.headers.get("content-type","")
    if 'application/json' not in ct.lower():
        st.error(f"GET_URL returned non-JSON (status {r.status_code}, content-type: {ct})")
        st.markdown("**Server response (first 2000 chars):**")
        st.code(r.text[:2000])
        st.stop()
    try:
        arr = r.json()
    except Exception as e:
        st.error("Failed to parse JSON from GET_URL: " + str(e))
        st.markdown("**Server raw response (first 2000 chars):**")
        st.code(r.text[:2000])
        st.stop()
    if not isinstance(arr, list):
        st.error("GET_URL returned JSON but not an array. Showing raw JSON (first 2000 chars).")
        st.code(str(arr)[:2000])
        st.stop()
    return pd.DataFrame(arr)

def interp_pos(past_row, now_row, current_time: datetime) -> Tuple[float,float]:
    try:
        t0 = pd.to_datetime(past_row["timestamp_ist"]); t1 = pd.to_datetime(now_row["timestamp_ist"])
    except Exception:
        return float(now_row["lat"]), float(now_row["lng"])
    if t1 <= t0:
        return float(now_row["lat"]), float(now_row["lng"])
    frac = (current_time - t0).total_seconds() / max(1.0, (t1 - t0).total_seconds())
    frac = max(0.0, min(1.0, frac))
    lat = float(past_row["lat"]) + (float(now_row["lat"]) - float(past_row["lat"])) * frac
    lng = float(past_row["lng"]) + (float(now_row["lng"]) - float(past_row["lng"])) * frac
    return lat, lng

# ---------- main (use small containers to reduce perceived flash) ----------
# Containers (preserve layout between reruns)
header_col, map_col = st.columns([1,3])
with header_col:
    last_update_placeholder = st.empty()
    status_placeholder = st.empty()

map_container = st.container()
left_col, right_col = st.columns([3,1])

# 1) fetch telemetry
df = fetch_rows_safe(GET_URL)
if df.empty or len(df) < 2:
    st.info("Waiting for telemetry (need >=2 rows).")
    st.stop()

# normalize
df.columns = [c.strip() for c in df.columns]
for c in ["lat","lng","cant_mm","grad_mm","cant_deg","grad_deg"]:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
if "timestamp_ist" in df.columns:
    df["timestamp_ist"] = pd.to_datetime(df["timestamp_ist"], errors="coerce")

# compute bbox and fetch cached railway geometry (every MAP_REFRESH_SEC seconds)
lat_min = float(df["lat"].min()); lat_max = float(df["lat"].max())
lng_min = float(df["lng"].min()); lng_max = float(df["lng"].max())
pad_lat = max(0.01, (lat_max - lat_min) * 0.2 + 0.005)
pad_lng = max(0.01, (lng_max - lng_min) * 0.2 + 0.005)
lat_min_q, lat_max_q = lat_min - pad_lat, lat_max + pad_lat
lng_min_q, lng_max_q = lng_min - pad_lng, lng_max + pad_lng

railways = fetch_railway_geo_cached(lat_min_q, lng_min_q, lat_max_q, lng_max_q)
if not railways:
    path_coords = [(float(r["lat"]), float(r["lng"])) for _, r in df[["lat","lng"]].dropna().iterrows()]
    railways = [path_coords]

# compute smooth marker pos
past = df.iloc[-2]; now = df.iloc[-1]
current_time = pd.to_datetime(datetime.now())
interp_lat, interp_lng = interp_pos(past, now, current_time)

# progress
start = (float(df.iloc[0]["lat"]), float(df.iloc[0]["lng"]))
end = (float(df.iloc[-1]["lat"]), float(df.iloc[-1]["lng"]))
total_km = haversine_km(start, end)
done_km = haversine_km(start, (interp_lat, interp_lng))
pct_done = min(100.0, (done_km / total_km * 100.0) if total_km > 0 else 0.0)

# status
cant_mm = float(now.get("cant_mm") or 0.0)
warn_thresh = WARNING_RATIO * PERMITTED_CANT_MM
status = "GREEN"
if cant_mm >= PERMITTED_CANT_MM: status = "RED"
elif cant_mm >= warn_thresh: status = "YELLOW"
color_map = {"GREEN":"#2ECC71","YELLOW":"#F1C40F","RED":"#E74C3C"}

# Update small header containers (fast)
last_update_placeholder.markdown(f"**Last poll:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
status_placeholder.markdown(f"<div style='display:flex;gap:8px;align-items:center'><div style='width:12px;height:12px;background:{color_map[status]};border-radius:4px'></div><div><b>Status:</b> {status}</div></div>", unsafe_allow_html=True)

# Map + path (in container to avoid full-layout flicker)
with map_container:
    st.subheader("Map (rail track is cached, re-fetched every 5 min)")
    # st.map expects DataFrame with 'latitude' & 'longitude'
    pts = list(itertools.chain.from_iterable(railways))
    map_df = pd.DataFrame([{"latitude": lat, "longitude": lng} for lat, lng in pts]).drop_duplicates().tail(5000)
    # draw path points (stable rendering)
    st.map(map_df)
    st.markdown(f"**Smooth marker:** {interp_lat:.6f}, {interp_lng:.6f}")
    st.write(f"Progress: **{pct_done:.2f}%** — {done_km:.3f} km of {total_km:.3f} km")

# Right column (metrics) updated rapidly
with right_col:
    st.subheader("Live readings")
    st.metric("Cant (mm)", f"{cant_mm:.2f}")
    st.metric("Cant (deg)", f"{(math.degrees(math.atan((cant_mm/1000.0)/GAUGE_M))):.3f}°")
    st.metric("Gradient (deg)", f"{(math.degrees(math.atan((float(now.get('grad_mm') or 0.0)/1000.0)/GAUGE_M))):.3f}°")
    if status == "RED":
        st.error("Threshold exceeded — immediate attention.")
    elif status == "YELLOW":
        st.warning("Approaching threshold.")
    else:
        st.success("Within allowed limits.")

# history (bottom)
st.subheader("History (latest)")
h = df.tail(1000)
st.line_chart(h[["cant_mm","grad_mm"]].fillna(method="ffill"))
with st.expander("Recent rows"):
    st.dataframe(h.tail(200))
