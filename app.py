# app.py
# Streamlit dashboard for your sheet with headers:
# Timestamp,TiltX_0,TiltY_0,TiltZ_0,TiltX_1,TiltY_1,TiltZ_1,TiltX_2,TiltY_2,TiltZ_2,TiltX_3,TiltY_3,TiltZ_3,Latitude,Longitude
# - polls data every 2s
# - approximates GPS along Delhi->Jaipur at 40 km/h if Latitude/Longitude missing
# - computes cant_mm and left/right diffs from TiltX (degrees)

import streamlit as st
import pandas as pd
import numpy as np
import math, requests, io, itertools
from datetime import datetime, timedelta
from typing import List, Tuple
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium
import folium
from urllib.parse import urlparse, parse_qs

# ---------- CONFIG ----------
READ_REFRESH_MS = 2000       # UI poll interval (2s)
MAP_CACHE_SEC = 300          # cache OSM fetch 5 min
GAUGE_M = 1.676              # Indian broad gauge in meters
TRAIN_SPEED_KMPH = 40.0      # speed for GPS approx
TRAIN_SPEED_MPS = TRAIN_SPEED_KMPH * 1000.0 / 3600.0
DELHI_LAT, DELHI_LON = 28.6517, 77.2219
JAIPUR_LAT, JAIPUR_LON = 26.9124, 75.7873
PERMITTED_CANT_MM = 165.0
WARNING_RATIO = 0.8

st.set_page_config(layout="wide", page_title="Rail Tilt Monitor (Degrees)")
st.title("Rail Tilt Monitor — live visualization (Tilt degrees, 4 sensors)")

# auto-refresh for UI
st_autorefresh(interval=READ_REFRESH_MS, limit=0, key="autorefresh")

# ---------- helpers ----------
def safe_float(x, default=np.nan):
    try:
        if x is None: return default
        return float(x)
    except Exception:
        return default

def to_export_csv_url(maybe_url: str) -> str:
    """Convert Google edit link or GitHub blob link to raw CSV URL if possible."""
    if not maybe_url:
        return maybe_url
    u = maybe_url.strip()
    # GitHub blob -> raw
    if "github.com" in u and "/blob/" in u:
        return u.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    # already raw / export
    if "raw.githubusercontent.com" in u or "output=csv" in u or "/export?format=csv" in u or "/pub?gid=" in u:
        return u
    # google sheet edit -> export
    try:
        p = urlparse(u)
        if "/spreadsheets/d/" in p.path:
            parts = p.path.split('/')
            if "d" in parts:
                sid = parts[parts.index("d")+1]
                q = parse_qs(p.query)
                gid = q.get("gid", [None])[0]
                if not gid and p.fragment:
                    frag_q = parse_qs(p.fragment)
                    gid = frag_q.get("gid", [None])[0]
                if not gid:
                    gid = "0"
                return f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid}"
    except Exception:
        pass
    return u

def load_csv_robust(url: str) -> pd.DataFrame:
    """Try to fetch & parse CSV from a variety of input URLs (Google edit links, GitHub blob, raw CSV)."""
    if not url:
        raise RuntimeError("No URL provided")
    csv_url = to_export_csv_url(url)
    try:
        r = requests.get(csv_url, timeout=20)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Could not GET URL: {csv_url}\nError: {e}")
    text = r.text
    head = text[:1000].lower()
    if "<!doctype html" in head or "<html" in head or "login" in head:
        raise RuntimeError("URL returned HTML (not CSV). If using Google Sheets, publish to web or make sharing 'anyone with link' and use the export URL or let the app convert your edit link.")
    # try normal pandas first
    try:
        df = pd.read_csv(io.StringIO(text))
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        # fallback tolerant parse
        try:
            df = pd.read_csv(io.StringIO(text), engine="python", on_bad_lines="skip")
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to parse CSV. Pandas error: {e}\nFirst 400 chars:\n{head}")

def haversine_km(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    lat1, lon1 = a; lat2, lon2 = b
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2-lat1); dlambda = math.radians(lon2-lon1)
    a_ = math.sin(dphi/2)*2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)*2
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

def interp_point_along(pts: List[Tuple[float,float]], dists_km: np.ndarray, dist_along_km: float) -> Tuple[float,float]:
    if dist_along_km <= 0: return pts[0]
    if dist_along_km >= dists_km[-1]: return pts[-1]
    idx = np.searchsorted(dists_km, dist_along_km)
    if idx == 0: return pts[0]
    t0 = dists_km[idx-1]; t1 = dists_km[idx]
    if t1 == t0: return pts[idx]
    alpha = (dist_along_km - t0) / (t1 - t0)
    lat = pts[idx-1][0] + alpha * (pts[idx][0] - pts[idx-1][0])
    lon = pts[idx-1][1] + alpha * (pts[idx][1] - pts[idx-1][1])
    return (lat, lon)

# ---------- UI input ----------
st.sidebar.header("Data source (paste URL)")
src_url = st.sidebar.text_input("Paste Google Sheets edit/share URL or GitHub file URL (the app auto-converts).", value="")
if not src_url:
    st.sidebar.info("Paste the sheet or GitHub file URL here; no file upload necessary.")
    st.stop()

# ---------- load sheet ----------
try:
    df = load_csv_robust(src_url)
except Exception as e:
    st.error("Failed to load CSV — check link/share permissions or publish-to-web. Error:")
    st.write(e)
    st.stop()

if df.empty:
    st.info("No rows found in the sheet.")
    st.stop()

# ---------- expected headers (case-insensitive) ----------
expected_headers = ["timestamp","tiltx_0","tilty_0","tiltz_0","tiltx_1","tilty_1","tiltz_1","tiltx_2","tilty_2","tiltz_2","tiltx_3","tilty_3","tiltz_3","latitude","longitude"]
cols_lower = {c.lower(): c for c in df.columns}

# Check for necessary TiltX and TiltY columns
required = ["tiltx_0","tiltx_1","tiltx_2","tiltx_3","tilty_0","tilty_1","tilty_2","tilty_3"]
missing = [r for r in required if r not in cols_lower]
if missing:
    st.error(f"Missing required tilt columns: {missing}. Please ensure your sheet has TiltX_0..TiltX_3 and TiltY_0..TiltY_3 (case-insensitive). Found columns: {list(df.columns)}")
    st.stop()

def col(name): return cols_lower.get(name.lower())

# parse timestamp column if present
ts_col = None
for c in df.columns:
    if "time" in c.lower() or "timestamp" in c.lower():
        ts_col = c; break
if ts_col:
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
else:
    st.warning("No timestamp column detected; the app will fallback to row-index timing for GPS approximation.")

# ---------- fetch railway geometry ----------
bbox_pad = 0.2
lat_min = min(DELHI_LAT, JAIPUR_LAT) - bbox_pad
lat_max = max(DELHI_LAT, JAIPUR_LAT) + bbox_pad
lon_min = min(DELHI_LON, JAIPUR_LON) - bbox_pad
lon_max = max(DELHI_LON, JAIPUR_LON) + bbox_pad

ways = fetch_osm_railways(lat_min, lon_min, lat_max, lon_max)
flat_pts = flatten_way_points(ways) if ways else [(DELHI_LAT,DELHI_LON),(JAIPUR_LAT,JAIPUR_LON)]
dists = cumulative_distances(flat_pts)
total_km = float(dists[-1]) if len(dists)>0 else haversine_km((DELHI_LAT,DELHI_LON),(JAIPUR_LAT,JAIPUR_LON))

# ---------- compute elapsed seconds and distance along route (40 km/h) ----------
if ts_col and df[ts_col].notna().any():
    first_ts = df[ts_col].dropna().iloc[0]
    elapsed_s = []
    for i, row in df.iterrows():
        if pd.notna(row.get(ts_col)):
            elapsed = (row[ts_col] - first_ts).total_seconds()
            if elapsed < 0:
                elapsed = max(0, i)
        else:
            elapsed = i
        elapsed_s.append(elapsed)
else:
    elapsed_s = [i for i in range(len(df))]

# distance in km along route using fixed speed
dist_along_km = [(s * TRAIN_SPEED_MPS) / 1000.0 for s in elapsed_s]
dist_along_km = [min(d, total_km) for d in dist_along_km]

# interpolate lat/lon
lats = []; lons = []
for dkm in dist_along_km:
    latx, longx = interp_point_along(flat_pts, dists, dkm)
    lats.append(latx); lons.append(longx)
df["Latitude_approx"] = lats
df["Longitude_approx"] = lons
df["elapsed_s"] = elapsed_s
df["dist_along_km"] = dist_along_km

# ---------- compute roll/pitch/cant from TiltX/TiltY (degrees) ----------
mean_rolls=[]; mean_pitch=[]
left_mean=[]; right_mean=[]; cant_mm_list=[]; left_right_diff_mm_list=[]
for _, row in df.iterrows():
    tx0 = safe_float(row[col("tiltx_0")]); tx1 = safe_float(row[col("tiltx_1")])
    tx2 = safe_float(row[col("tiltx_2")]); tx3 = safe_float(row[col("tiltx_3")])
    ty0 = safe_float(row[col("tilty_0")]); ty1 = safe_float(row[col("tilty_1")])
    ty2 = safe_float(row[col("tilty_2")]); ty3 = safe_float(row[col("tilty_3")])
    rolls = np.array([tx0, tx1, tx2, tx3], dtype=float)
    pitches = np.array([ty0, ty1, ty2, ty3], dtype=float)
    mean_roll_deg = float(np.nanmean(rolls))
    mean_pitch_deg = float(np.nanmean(pitches))
    left_mean_roll_deg = float(np.nanmean([tx0, tx3]))
    right_mean_roll_deg = float(np.nanmean([tx1, tx2]))
    mean_rolls.append(mean_roll_deg); mean_pitch.append(mean_pitch_deg)
    left_mean.append(left_mean_roll_deg); right_mean.append(right_mean_roll_deg)
    mean_roll_rad = math.radians(mean_roll_deg)
    left_right_diff_rad = math.radians(left_mean_roll_deg - right_mean_roll_deg)
    cant_mm = math.tan(mean_roll_rad) * GAUGE_M * 1000.0
    left_right_diff_mm = math.tan(left_right_diff_rad) * GAUGE_M * 1000.0
    cant_mm_list.append(cant_mm); left_right_diff_mm_list.append(left_right_diff_mm)

df["mean_roll_deg"] = mean_rolls
df["mean_pitch_deg"] = mean_pitch
df["left_mean_roll_deg"] = left_mean
df["right_mean_roll_deg"] = right_mean
df["cant_mm"] = cant_mm_list
df["left_right_diff_mm"] = left_right_diff_mm_list

# ---------- ETA & progress ----------
last_dist = df["dist_along_km"].iloc[-1]
progress_km = last_dist
eta = None
remaining_km = max(0.0, total_km - last_dist)
if TRAIN_SPEED_MPS > 0:
    avg_kps = TRAIN_SPEED_MPS / 1000.0
    if avg_kps > 0:
        eta_seconds = remaining_km / avg_kps
        eta = datetime.now() + timedelta(seconds=eta_seconds)

# ---------- UI layout ----------
col_map, col_metrics = st.columns([2,1])

with col_map:
    st.subheader("Map — OSM railway + telemetry path (approximated if needed)")
    center = (df["Latitude_approx"].iloc[-1], df["Longitude_approx"].iloc[-1])
    m = folium.Map(location=center, zoom_start=8)
    if flat_pts:
        folium.PolyLine(flat_pts, color="darkblue", weight=3, opacity=0.6, tooltip="OSM railway").add_to(m)
    pts = list(zip(df["Latitude_approx"].astype(float), df["Longitude_approx"].astype(float)))
    folium.PolyLine(pts, color="red", weight=2.5, opacity=0.9, tooltip="Telemetry path (approx)").add_to(m)
    folium.CircleMarker(location=pts[-1], radius=6, color="black", fill=True, fill_color="red", tooltip="Current pos").add_to(m)
    st_folium(m, width=900, height=520)

with col_metrics:
    st.subheader("Latest telemetry")
    latest = df.iloc[-1]
    st.write("Timestamp:", latest.get(col("timestamp")) if col("timestamp") in cols_lower else latest.get(df.columns[0], "—"))
    st.metric("Latitude (approx)", f"{latest['Latitude_approx']:.6f}")
    st.metric("Longitude (approx)", f"{latest['Longitude_approx']:.6f}")
    st.metric("Elapsed (s)", int(latest["elapsed_s"]))
    st.metric("Dist along (km)", f"{latest['dist_along_km']:.3f}")
    st.metric("Progress (km)", f"{progress_km:.2f} / {total_km:.1f}")
    st.metric("Mean roll (deg)", f"{latest['mean_roll_deg']:.3f}")
    st.metric("Mean pitch (deg)", f"{latest['mean_pitch_deg']:.3f}")
    st.metric("Cant (mm)", f"{latest['cant_mm']:.2f}")
    st.metric("Left-right diff (mm)", f"{latest['left_right_diff_mm']:.2f}")
    st.metric("ETA (approx)", eta.strftime("%Y-%m-%d %H:%M:%S") if eta else "—")
    color = "🟢"
    if latest["cant_mm"] >= PERMITTED_CANT_MM:
        color = "🔴"
    elif latest["cant_mm"] >= WARNING_RATIO * PERMITTED_CANT_MM:
        color = "🟡"
    st.markdown(f"*Status:* {color}")

st.divider()
st.subheader("Charts — cant & left/right diff")
st.line_chart(df[["cant_mm","left_right_diff_mm"]].fillna(method="ffill").tail(400))

st.subheader("Tilt sensors (TiltX = roll, TiltY = pitch)")
st.line_chart(df[[col("tiltx_0"), col("tiltx_1"), col("tiltx_2"), col("tiltx_3")]].tail(200).astype(float))
st.line_chart(df[[col("tilty_0"), col("tilty_1"), col("tilty_2"), col("tilty_3")]].tail(200).astype(float))

st.subheader("Latest telemetry rows (tail)")
st.dataframe(df.tail(200))

st.subheader("Download augmented CSV")
buf = io.StringIO()
df.to_csv(buf, index=False)
st.download_button("Download augmented CSV (Latitude_approx,cant_mm,...)", data=buf.getvalue(), file_name="augmented_telemetry.csv", mime="text/csv")

st.caption("Notes: Paste the sheet or GitHub file link in the sidebar. If the sheet is private, publish or share so the app can fetch it. GPS is approximated using 40 km/h along the OSM railway geometry; nothing writes back to your sheet.")
