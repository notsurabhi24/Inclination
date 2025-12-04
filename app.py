# app.py
# Streamlit dashboard that reads a published Google Sheet CSV (no JSON),
# refreshes telemetry every 2s, refreshes railway geometry every 5min,
# shows map with railway geometry (Delhi->Jaipur fallback), interpolates marker
# and displays cant/tilt/alerts. Simulator for GPS is optional (sidebar).

import streamlit as st
import pandas as pd
import numpy as np
import requests, math, itertools, time
from datetime import datetime
from typing import Tuple, List
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium
import folium

# ------------- USER CONFIG -------------
# Put your published CSV URL here (or set st.secrets["CSV_URL"])
CSV_URL = st.secrets.get("CSV_URL", "")  # OR paste "https://docs.google.com/.../pub?gid=0&single=true&output=csv"
READ_REFRESH_MS = 2000       # poll telemetry every 2s
MAP_CACHE_SEC = 300          # refresh railway geometry every 5 minutes
PERMITTED_CANT_MM = 165.0    # threshold, adjust if needed
WARNING_RATIO = 0.8          # yellow threshold = 0.8 * permitted
GAUGE_M = 1.676              # used if converting mm->deg (approx)
OVERPASS_TIMEOUT = 15

st.set_page_config(layout="wide", page_title="Rail Cant Monitor (CSV)", initial_sidebar_state="expanded")
st.title("Rail Cant Monitor — CSV-backed (Live)")

# small autorefresh for UI
st_autorefresh(interval=READ_REFRESH_MS, limit=0, key="autorefresh")

# ------------ Sidebar controls ------------
st.sidebar.header("Configuration")
csv_url_input = st.sidebar.text_input("Published CSV URL (leave blank to use CSV_URL secret)", value=CSV_URL)
if csv_url_input:
    CSV_URL = csv_url_input

st.sidebar.write("If your sheet doesn't have lat/lng, enable GPS simulation below.")
simulate_gps = st.sidebar.checkbox("Simulate GPS along Delhi→Jaipur (if no lat/lng)", value=False)
if simulate_gps:
    st.sidebar.markdown("Start/end for simulated GPS (defaults to Delhi→Jaipur bounding coords)")
    sim_start_lat = st.sidebar.number_input("Start lat", value=28.6517, format="%.6f")
    sim_start_lng = st.sidebar.number_input("Start lon", value=77.2219, format="%.6f")
    sim_end_lat = st.sidebar.number_input("End lat", value=26.9124, format="%.6f")
    sim_end_lng = st.sidebar.number_input("End lon", value=75.7873, format="%.6f")

st.sidebar.markdown("---")
st.sidebar.markdown("Map refresh: cached every 5 minutes to avoid flashing")
st.sidebar.markdown("Telemetry poll: every 2 seconds")

# ------------- helpers -------------
def safe_float(x, default=np.nan):
    try:
        if x is None: return default
        return float(x)
    except Exception:
        return default

def haversine_km(p1: Tuple[float,float], p2: Tuple[float,float]) -> float:
    lat1, lon1 = p1; lat2, lon2 = p2
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2-lat1); dlambda = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

@st.cache_data(ttl=MAP_CACHE_SEC)
def fetch_osm_rail(lat_min: float, lon_min: float, lat_max: float, lon_max: float) -> List[List[Tuple[float,float]]]:
    """
    Query Overpass for railway ways inside bbox. Returns list of ways (list of (lat,lng)).
    Cached for MAP_CACHE_SEC seconds to avoid reloading tiles.
    """
    bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    q = f"""
    [out:json][timeout:{OVERPASS_TIMEOUT}];
    (
      way["railway"~"rail|light_rail|railway"]({bbox});
    );
    out body;
    >;
    out skel qt;
    """
    try:
        r = requests.post("https://overpass-api.de/api/interpreter", data=q.strip(), timeout=OVERPASS_TIMEOUT)
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
        return ways
    except Exception:
        return []

def load_csv_dataframe(url: str) -> pd.DataFrame:
    """Load the published CSV into a pandas DataFrame; show friendly error if fails."""
    try:
        df = pd.read_csv(url)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        st.error("Failed to fetch CSV. Make sure you published the sheet to web (CSV) and pasted the correct URL.")
        st.write("Error:", e)
        st.stop()

def safe_interp_between_rows(row0, row1, now_ts):
    """Interpolate lat/lng between two rows based on their timestamps. Returns tuple (lat,lng)."""
    try:
        t0 = pd.to_datetime(row0["timestamp"], errors="coerce")
        t1 = pd.to_datetime(row1["timestamp"], errors="coerce")
        if pd.isna(t0) or pd.isna(t1) or t1 <= t0:
            return safe_float(row1.get("lat")), safe_float(row1.get("lng"))
        total = (t1 - t0).total_seconds()
        frac = (now_ts - t0).total_seconds() / total
        frac = max(0.0, min(1.0, frac))
        lat = safe_float(row0.get("lat")) + (safe_float(row1.get("lat")) - safe_float(row0.get("lat"))) * frac
        lng = safe_float(row0.get("lng")) + (safe_float(row1.get("lng")) - safe_float(row0.get("lng"))) * frac
        return lat, lng
    except Exception:
        return safe_float(row1.get("lat")), safe_float(row1.get("lng"))

# ------------- main -------------
if not CSV_URL:
    st.warning("Please paste your published CSV URL in the sidebar or set CSV_URL in st.secrets.")
    st.stop()

df = load_csv_dataframe(CSV_URL)
if df.empty:
    st.info("Spreadsheet has no rows yet.")
    st.stop()

# Normalize (we expect headers as confirmed)
# Expected headers: timestamp, channel, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, roll, pitch, [optional cant_mm, lat, lng, progress_km]
cols = [c.lower() for c in df.columns]
colmap = {c.lower(): c for c in df.columns}

# detect lat/lng columns if present
has_lat = any("lat" == c or c.endswith("lat") for c in cols)
has_lng = any("lng" == c or c.endswith("lng") for c in cols)
lat_col = colmap.get("lat") if "lat" in colmap else (colmap.get("latitude") if "latitude" in colmap else None)
lng_col = colmap.get("lng") if "lng" in colmap else (colmap.get("longitude") if "longitude" in colmap else None)

# detect cant column
cant_col = None
for candidate in ["cant_mm","cant","cant_mm "]:
    if candidate in colmap:
        cant_col = colmap[candidate]
        break

# compute latest position (interpolated)
latest_row = df.iloc[-1].to_dict()
prev_row = df.iloc[-2].to_dict() if len(df) >= 2 else None
now_ts = pd.to_datetime(datetime.now())

if lat_col and lng_col:
    # interpolate marker smoothly
    if prev_row is not None and prev_row.get("timestamp") is not None:
        lat_marker, lng_marker = safe_interp_between_rows(prev_row, latest_row, now_ts)
    else:
        lat_marker, lng_marker = safe_float(latest_row.get(lat_col)), safe_float(latest_row.get(lng_col))
else:
    # No lat/lng in sheet: optionally simulate simple straight-line GPS along Delhi->Jaipur if enabled
    if simulate_gps:
        # linear interpolate based on row index
        n = len(df)
        idx = len(df)-1
        frac = idx / max(1, n-1)
        lat_marker = sim_start_lat + (sim_end_lat - sim_start_lat) * frac
        lng_marker = sim_start_lng + (sim_end_lng - sim_start_lng) * frac
    else:
        lat_marker = None; lng_marker = None

# compute progress (if progress_km column exists), else try compute from lat/lng
progress_km = None
if "progress_km" in colmap:
    progress_km = safe_float(latest_row.get(colmap["progress_km"]))
elif lat_marker is not None and lat_col and lng_col:
    # compute rough progress along path: use first and last row as endpoints
    start = (safe_float(df.iloc[0].get(lat_col)), safe_float(df.iloc[0].get(lng_col)))
    end = (safe_float(df.iloc[-1].get(lat_col)), safe_float(df.iloc[-1].get(lng_col)))
    total_km = haversine_km(start, end) if start and end else 0.0
    done_km = haversine_km(start, (lat_marker, lng_marker))
    progress_km = done_km
else:
    progress_km = None

# cant reading: prefer sheet value if present, else simulate mild cant for display
if cant_col:
    cant_val = safe_float(latest_row.get(cant_col))
else:
    # simulate cant smoothly and use occasional spikes tied to row index
    idx = len(df)-1
    base = 30.0 + 10.0 * math.sin(idx/40.0)
    # create deterministic spikes every few hundred rows for demonstration
    spike = 0.0
    if (idx % 350) in range(0, 25):
        spike = 90.0 * math.exp(-((idx % 350)/10.0))
    cant_val = base + spike

# status
warn_thresh = WARNING_RATIO * PERMITTED_CANT_MM
status = "GREEN"
if not math.isnan(cant_val) and cant_val >= PERMITTED_CANT_MM:
    status = "RED"
elif not math.isnan(cant_val) and cant_val >= warn_thresh:
    status = "YELLOW"

# ------------- Layout & Display -------------
header_col, map_col = st.columns([1,2])
with header_col:
    st.subheader("Live telemetry")
    st.write("Timestamp:", latest_row.get("timestamp", "<no timestamp>"))
    st.write("Channel:", latest_row.get("channel", "—"))
    # show IMU values if present
    for key in ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z","roll","pitch"]:
        if key in colmap:
            st.metric(key, latest_row.get(colmap[key]))
    st.metric("Cant (mm)", f"{cant_val:.2f}")
    st.markdown(f"**Status:** {'🟢' if status=='GREEN' else '🟡' if status=='YELLOW' else '🔴'} {status}")

    if progress_km is not None:
        st.progress(min(100.0, (progress_km / max(1.0, (df.shape[0] / 3600.0))) * 100.0))  # rough UI progress

with map_col:
    st.subheader("Map")
    if lat_marker is None or lng_marker is None:
        st.info("No GPS in sheet. Enable 'Simulate GPS' in sidebar to see a map.")
    else:
        # build folium map and show OSM rail if available
        m = folium.Map(location=(lat_marker, lng_marker), zoom_start=9, tiles="OpenStreetMap")
        folium.Marker(location=(lat_marker, lng_marker), tooltip="Current pos").add_to(m)

        # compute bbox for Overpass
        # find min/max from dataframe if lat/lng present
        if lat_col and lng_col:
            lats = df[lat_col].dropna().astype(float)
            lngs = df[lng_col].dropna().astype(float)
            lat_min, lat_max = float(lats.min()), float(lats.max())
            lng_min, lng_max = float(lngs.min()), float(lngs.max())
        else:
            # use route bounds
            lat_min, lat_max = min(sim_start_lat, sim_end_lat), max(sim_start_lat, sim_end_lat)
            lng_min, lng_max = min(sim_start_lng, sim_end_lng), max(sim_start_lng, sim_end_lng)

        ways = fetch_osm_rail(lat_min-0.02, lng_min-0.02, lat_max+0.02, lng_max+0.02)

        if not ways:
            # fallback: plot recent GPS points if available
            if lat_col and lng_col:
                pts = list(zip(df[lat_col].astype(float), df[lng_col].astype(float)))
                folium.PolyLine(pts, color="blue", weight=3).add_to(m)
        else:
            for way in ways:
                folium.PolyLine(way, color="green", weight=2).add_to(m)

        st_folium(m, width="100%", height=480)

st.divider()
st.subheader("History (latest rows)")
show_cols = [c for c in df.columns if c.lower() in ("timestamp","channel","acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z","roll","pitch","cant_mm","lat","lng","progress_km")]
if not show_cols:
    show_cols = df.columns.tolist()
st.dataframe(df[show_cols].tail(200))

# small chart for cant history if synthetic
if "cant_mm" in colmap:
    st.subheader("Cant history")
    st.line_chart(df[colmap["cant_mm"]].tail(500).fillna(method="ffill"))
else:
    # create synthetic cant series for chart
    synthetic = [30.0 + 10.0*math.sin(i/40.0) + (90.0*math.exp(-((i%350)/10.0)) if (i%350) < 25 else 0.0) for i in range(max(0, len(df)-500), len(df))]
    st.subheader("Simulated cant history")
    st.line_chart(pd.Series(synthetic).fillna(method="ffill"))

st.caption("Map refresh cached every 5 minutes. Telemetry polled every 2 seconds. If your sheet lacks GPS, enable 'Simulate GPS' in the sidebar.")
