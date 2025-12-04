# app.py — Tilt (degrees) version: uses TiltX=roll(deg), TiltY=pitch(deg)
import streamlit as st
import pandas as pd
import numpy as np
import math, requests, itertools, io
from datetime import datetime, timedelta
from typing import List, Tuple
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium
import folium

# ---------- CONFIG ----------
READ_REFRESH_MS = 2000
MAP_CACHE_SEC = 300
PERMITTED_CANT_MM = 165.0
WARNING_RATIO = 0.8
GAUGE_M = 1.676  # meters (Indian broad gauge)
DELHI_LAT, DELHI_LON = 28.6517, 77.2219
JAIPUR_LAT, JAIPUR_LON = 26.9124, 75.7873

st.set_page_config(layout="wide", page_title="Railboard Tilt Monitor (degrees)")
st.title("Railboard Tilt Monitor — Tilt values in degrees")

st_autorefresh(interval=READ_REFRESH_MS, limit=0, key="autorefresh")

# ---------- helpers ----------
def safe_float(x, default=np.nan):
    try:
        if x is None: return default
        return float(x)
    except Exception:
        return default

def haversine_km(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    lat1, lon1 = a; lat2, lon2 = b
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2-lat1); dlambda = math.radians(lon2-lon1)
    a_ = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a_), math.sqrt(1-a_))

@st.cache_data(ttl=MAP_CACHE_SEC)
def fetch_osm_railways(lat_min, lon_min, lat_max, lon_max) -> List[List[Tuple[float,float]]]:
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
                coords = [nodes[nid] for nid in el.get("nodes", []) if nid in nodes]
                if len(coords) >= 2:
                    ways.append(coords)
        return ways
    except Exception:
        return []

def flatten_way_points(ways: List[List[Tuple[float,float]]]) -> List[Tuple[float,float]]:
    pts = list(itertools.chain.from_iterable(ways))
    clean = []
    for p in pts:
        if not clean or p != clean[-1]:
            clean.append(p)
    return clean

def cumulative_distances(pts: List[Tuple[float,float]]) -> np.ndarray:
    ds = [0.0]
    for i in range(1, len(pts)):
        ds.append(ds[-1] + haversine_km(pts[i-1], pts[i]))
    return np.array(ds)

def interp_point_along(pts: List[Tuple[float,float]], dists_km: np.ndarray, frac: float) -> Tuple[float,float]:
    if frac <= 0: return pts[0]
    if frac >= 1: return pts[-1]
    total = dists_km[-1]
    target = frac * total
    idx = np.searchsorted(dists_km, target)
    if idx == 0: return pts[0]
    if idx >= len(pts): return pts[-1]
    t0 = dists_km[idx-1]; t1 = dists_km[idx]
    if t1 == t0: return pts[idx]
    alpha = (target - t0) / (t1 - t0)
    lat = pts[idx-1][0] + alpha * (pts[idx][0] - pts[idx-1][0])
    lon = pts[idx-1][1] + alpha * (pts[idx][1] - pts[idx-1][1])
    return (lat, lon)

# ---------- UI: get CSV URL ----------
st.sidebar.header("Spreadsheet CSV")
csv_url = st.sidebar.text_input("Paste published CSV URL (docs.google.com/.../pub?output=csv)", value="")
if not csv_url:
    st.sidebar.info("Paste the published CSV URL from your Google Sheet (File → Share → Publish to web → CSV).")
    st.stop()

# ---------- Load CSV ----------
@st.cache_data(ttl=1)
def load_csv(url):
    df = pd.read_csv(url)
    df.columns = [c.strip() for c in df.columns]
    return df

try:
    df = load_csv(csv_url)
except Exception as e:
    st.error("Failed to load CSV. Check URL and that the sheet is published to web as CSV.")
    st.write(e)
    st.stop()

if df.empty:
    st.info("No rows yet.")
    st.stop()

# ---------- Expected columns ----------
cols_lower = {c.lower(): c for c in df.columns}
required = ["timestamp","tiltx_0","tiltx_1","tiltx_2","tiltx_3","tilty_0","tilty_1","tilty_2","tilty_3","tiltz_0","tiltz_1","tiltz_2","tiltz_3"]
missing = [r for r in required if r not in cols_lower]
if missing:
    st.error(f"Missing columns (case-insensitive): {missing}. Make sure your sheet has these columns.")
    st.stop()

# map names
def col(name): return cols_lower.get(name.lower())

# parse timestamps
ts_col = None
for c in df.columns:
    if "timestamp" in c.lower():
        ts_col = c; break
if ts_col:
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
else:
    st.error("No timestamp column found.")
    st.stop()

# lat/lon detection and OSM fetch
lat_col = cols_lower.get("latitude")
lon_col = cols_lower.get("longitude")
have_latlng = lat_col is not None and lon_col is not None and df[lat_col].notna().sum() > 0

bbox_pad = 0.2
lat_min = min(DELHI_LAT, JAIPUR_LAT) - bbox_pad
lat_max = max(DELHI_LAT, JAIPUR_LAT) + bbox_pad
lon_min = min(DELHI_LON, JAIPUR_LON) - bbox_pad
lon_max = max(DELHI_LON, JAIPUR_LON) + bbox_pad

ways = fetch_osm_railways(lat_min, lon_min, lat_max, lon_max)
flat_pts = flatten_way_points(ways) if ways else [(DELHI_LAT,DELHI_LON),(JAIPUR_LAT,JAIPUR_LON)]
dists = cumulative_distances(flat_pts)
total_km = float(dists[-1]) if len(dists)>0 else haversine_km((DELHI_LAT,DELHI_LON),(JAIPUR_LAT,JAIPUR_LON))

# ---------- Build approximate GPS if missing ----------
if not have_latlng:
    st.info("Latitude/Longitude missing — approximating along Delhi→Jaipur using timestamps.")
    first_ts = df[ts_col].dropna().iloc[0] if df[ts_col].dropna().shape[0]>0 else None
    last_ts = df[ts_col].dropna().iloc[-1] if df[ts_col].dropna().shape[0]>1 else None
    if first_ts is None or last_ts is None or first_ts==last_ts:
        # fallback linear by index
        fracs = [i/(len(df)-1) if len(df)>1 else 0 for i in range(len(df))]
    else:
        total_sec = (last_ts - first_ts).total_seconds()
        fracs = []
        for t in df[ts_col]:
            if pd.isna(t):
                fracs.append(0.0)
            else:
                fracs.append(max(0.0, min(1.0, (t - first_ts).total_seconds() / max(1.0, total_sec))))
    approx_lats=[]; approx_lons=[]
    for frac in fracs:
        latx,lngx = interp_point_along(flat_pts, dists, frac)
        approx_lats.append(latx); approx_lons.append(lngx)
    df["Latitude_approx"] = approx_lats
    df["Longitude_approx"] = approx_lons
else:
    df["Latitude_approx"] = df[lat_col].astype(float)
    df["Longitude_approx"] = df[lon_col].astype(float)

# ---------- Compute cant & pitch from TiltX/TiltY (degrees) ----------
# per-row: mean_roll_deg, left_mean_roll_deg, right_mean_roll_deg, cant_mm, left_right_diff_mm
mean_rolls=[]; left_mean=[]; right_mean=[]; cant_mm_list=[]; left_right_diff_mm_list=[]; mean_pitch_list=[]
for _, row in df.iterrows():
    tx0 = safe_float(row[col("tiltx_0")]); tx1 = safe_float(row[col("tiltx_1")])
    tx2 = safe_float(row[col("tiltx_2")]); tx3 = safe_float(row[col("tiltx_3")])
    ty0 = safe_float(row[col("tilty_0")]); ty1 = safe_float(row[col("tilty_1")])
    ty2 = safe_float(row[col("tilty_2")]); ty3 = safe_float(row[col("tilty_3")])
    # mean roll/pitch across 4 sensors
    rolls = np.array([tx0, tx1, tx2, tx3], dtype=float)
    pitches = np.array([ty0, ty1, ty2, ty3], dtype=float)
    mean_roll_deg = np.nanmean(rolls)
    mean_pitch_deg = np.nanmean(pitches)
    left_mean_roll_deg = np.nanmean([tx0, tx3])  # FL, BL
    right_mean_roll_deg = np.nanmean([tx1, tx2]) # FR, BR
    # convert deg->rad
    mean_roll_rad = math.radians(mean_roll_deg)
    left_right_diff_rad = math.radians(left_mean_roll_deg - right_mean_roll_deg)
    # cant across gauge using mean roll
    cant_mm = math.tan(mean_roll_rad) * GAUGE_M * 1000.0
    left_right_diff_mm = math.tan(left_right_diff_rad) * GAUGE_M * 1000.0
    mean_rolls.append(mean_roll_deg)
    mean_pitch_list.append(mean_pitch_deg)
    left_mean.append(left_mean_roll_deg)
    right_mean.append(right_mean_roll_deg)
    cant_mm_list.append(cant_mm)
    left_right_diff_mm_list.append(left_right_diff_mm)

df["mean_roll_deg"] = mean_rolls
df["mean_pitch_deg"] = mean_pitch_list
df["left_mean_roll_deg"] = left_mean
df["right_mean_roll_deg"] = right_mean
df["cant_mm"] = cant_mm_list
df["left_right_diff_mm"] = left_right_diff_mm_list

# ---------- compute progress and ETA ----------
# fraction mapping from timestamps
first_ts = df[ts_col].dropna().iloc[0] if df[ts_col].dropna().shape[0]>0 else None
last_ts = df[ts_col].dropna().iloc[-1] if df[ts_col].dropna().shape[0]>1 else None
if first_ts is not None and last_ts is not None and first_ts != last_ts:
    now_row = df.iloc[-1]
    if pd.notna(now_row[ts_col]):
        frac_now = max(0.0, min(1.0, (now_row[ts_col] - first_ts).total_seconds() / max(1.0, (last_ts - first_ts).total_seconds())))
    else:
        frac_now = (len(df)-1)/max(1,len(df)-1)
else:
    frac_now = (len(df)-1)/max(1,len(df)-1)
progress_km = frac_now * total_km
eta = None
try:
    if first_ts is not None and last_ts is not None and last_ts != first_ts:
        total_time_sec = max(1.0,(last_ts - first_ts).total_seconds())
        avg_speed_kps = total_km / total_time_sec
        remaining_km = total_km * (1.0 - frac_now)
        eta_seconds = remaining_km / max(1e-9, avg_speed_kps)
        eta = datetime.now() + timedelta(seconds=eta_seconds)
except Exception:
    eta = None

# ---------- UI layout ----------
col_map, col_metrics = st.columns([2,1])

with col_map:
    st.subheader("Map: railway + telemetry path (approx if no real GPS)")
    center = (df["Latitude_approx"].iloc[-1], df["Longitude_approx"].iloc[-1])
    m = folium.Map(location=center, zoom_start=8)
    if flat_pts:
        folium.PolyLine(flat_pts, color="darkblue", weight=3, opacity=0.6, tooltip="OSM railway").add_to(m)
    pts = list(zip(df["Latitude_approx"].astype(float), df["Longitude_approx"].astype(float)))
    folium.PolyLine(pts, color="red", weight=2.5, opacity=0.9, tooltip="Telemetry path (approx)").add_to(m)
    folium.CircleMarker(location=pts[-1], radius=6, color="black", fill=True, fill_color="red", tooltip="Current pos").add_to(m)
    st_folium(m, width=900, height=520)

with col_metrics:
    st.subheader("Latest metrics")
    latest = df.iloc[-1]
    st.write("Timestamp:", latest[ts_col])
    st.metric("Latitude (approx)", f"{latest['Latitude_approx']:.6f}")
    st.metric("Longitude (approx)", f"{latest['Longitude_approx']:.6f}")
    st.metric("Progress (km)", f"{progress_km:.2f} / {total_km:.1f}")
    st.metric("Mean roll (deg)", f"{latest['mean_roll_deg']:.3f}")
    st.metric("Mean pitch (deg)", f"{latest['mean_pitch_deg']:.3f}")
    st.metric("Cant (mm)", f"{latest['cant_mm']:.2f}")
    st.metric("Left-right diff (mm)", f"{latest['left_right_diff_mm']:.2f}")
    st.metric("ETA (approx)", eta.strftime('%Y-%m-%d %H:%M:%S') if eta else "—")
    # status
    color = "🟢"
    if latest["cant_mm"] >= PERMITTED_CANT_MM:
        color = "🔴"
    elif latest["cant_mm"] >= WARNING_RATIO * PERMITTED_CANT_MM:
        color = "🟡"
    st.markdown(f"**Status:** {color}")

# charts
st.subheader("Time series — cant (mm) and left-right diff (mm)")
st.line_chart(df[["cant_mm","left_right_diff_mm"]].fillna(method="ffill").tail(1000))

st.subheader("TiltX (roll) sensors — last 200 rows")
st.line_chart(df[[col("tiltx_0"), col("tiltx_1"), col("tiltx_2"), col("tiltx_3")]].tail(200).astype(float))

st.subheader("TiltY (pitch) sensors — last 200 rows")
st.line_chart(df[[col("tilty_0"), col("tilty_1"), col("tilty_2"), col("tilty_3")]].tail(200).astype(float))

st.subheader("Latest telemetry rows (tail)")
st.dataframe(df.tail(200))

# augmented CSV download
st.subheader("Download augmented CSV")
buf = io.StringIO()
df.to_csv(buf, index=False)
st.download_button("Download augmented CSV (incl. Latitude_approx, cant_mm, etc.)", data=buf.getvalue(), file_name="augmented_telemetry.csv", mime="text/csv")

st.caption("Notes: TiltX_* are interpreted as roll in degrees; TiltY_* as pitch in degrees. Cant (mm) is computed as tan(mean_roll) * gauge * 1000. Left-right diff is difference between left and right mean rolls converted similarly. If your Tilt values are in a different convention, tell me and I'll adjust the signs/axes.")
