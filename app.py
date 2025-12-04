# app.py
# Streamlit dashboard: reads published Google Sheet CSV, computes/approximates GPS along Delhi->Jaipur if missing,
# computes cant from 4 corner TiltZ values, shows map with real railway overlay (Overpass), progress, ETA,
# and allows download of augmented CSV (no automatic writes to your sheet).

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
READ_REFRESH_MS = 2000   # poll CSV every 2s for UI
MAP_CACHE_SEC = 300      # cache Overpass results 5 min
PERMITTED_CANT_MM = 165.0
WARNING_RATIO = 0.8
GAUGE_M = 1.676

# Delhi -> Jaipur bounding coordinates used to get OSM railway
DELHI_LAT, DELHI_LON = 28.6517, 77.2219
JAIPUR_LAT, JAIPUR_LON = 26.9124, 75.7873

st.set_page_config(layout="wide", page_title="Rail Tilt (Delhi→Jaipur) Monitor")
st.title("Rail Tilt Monitor — show live sheet + approximate GPS along Delhi→Jaipur")

st.markdown("App reads a published Google Sheets CSV. If Latitude/Longitude missing, it will approximate GPS along Delhi→Jaipur track using Overpass OSM geometry and the timestamps in the sheet.")

# small auto-refresh
st_autorefresh(interval=READ_REFRESH_MS, limit=0, key="autorefresh")

# ---------- helper functions ----------
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
    # deduplicate consecutive duplicates
    clean = []
    for p in pts:
        if not clean or p != clean[-1]:
            clean.append(p)
    return clean

def cumulative_distances(pts: List[Tuple[float,float]]) -> np.ndarray:
    ds = [0.0]
    for i in range(1, len(pts)):
        ds.append(ds[-1] + haversine_km(pts[i-1], pts[i]))
    return np.array(ds)  # in km

def interp_point_along(pts: List[Tuple[float,float]], dists_km: np.ndarray, frac: float) -> Tuple[float,float]:
    if frac <= 0: return pts[0]
    if frac >= 1: return pts[-1]
    total = dists_km[-1]
    target = frac * total
    idx = np.searchsorted(dists_km, target)
    if idx == 0: return pts[0]
    if idx >= len(pts): return pts[-1]
    # linear interpolate between idx-1 and idx
    t0 = dists_km[idx-1]; t1 = dists_km[idx]
    if t1 == t0:
        return pts[idx]
    alpha = (target - t0) / (t1 - t0)
    lat = pts[idx-1][0] + alpha * (pts[idx][0] - pts[idx-1][0])
    lon = pts[idx-1][1] + alpha * (pts[idx][1] - pts[idx-1][1])
    return (lat, lon)

def plane_fit_from_zs(z_FL, z_FR, z_BR, z_BL, board_w=0.30, board_h=0.20):
    # Fit z = a*x + b*y + c where x,y are positions of corners (meters)
    X=[]; Y=[]; Z=[]
    positions = {"FL":(-board_w/2, board_h/2), "FR":(board_w/2, board_h/2), "BR":(board_w/2, -board_h/2), "BL":(-board_w/2, -board_h/2)}
    vals = {"FL": z_FL, "FR": z_FR, "BR": z_BR, "BL": z_BL}
    try:
        for k,(x,y) in positions.items():
            X.append(x); Y.append(y); Z.append(vals[k])
        A = np.vstack([X, Y, np.ones(len(X))]).T
        sol,_,_,_ = np.linalg.lstsq(A, Z, rcond=None)
        a,b,c = sol
        grad_deg = math.degrees(math.atan(a))
        cant_deg = math.degrees(math.atan(b))
        grad_mm = math.tan(math.radians(grad_deg)) * GAUGE_M * 1000.0
        cant_mm = math.tan(math.radians(cant_deg)) * GAUGE_M * 1000.0
        return round(cant_mm,2), round(grad_mm,2), round(cant_deg,3), round(grad_deg,3)
    except Exception:
        return (np.nan,)*4

# ---------- UI: CSV input ----------
st.sidebar.header("Sheet CSV")
csv_url = st.sidebar.text_input("Published CSV URL (docs.google.com/.../pub?output=csv)", value="")
if not csv_url:
    st.sidebar.info("Paste the published CSV link from your Google Sheet here.")
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
    st.error("Failed to load CSV — check the URL and that the sheet is published to web (CSV).")
    st.write(e)
    st.stop()

if df.empty:
    st.info("No rows yet.")
    st.stop()

# Normalize expected header names (case-insensitive)
mapping_expected = {
    "timestamp":"timestamp",
    "tiltx_0":"TiltX_0","tilty_0":"TiltY_0","tiltz_0":"TiltZ_0",
    "tiltx_1":"TiltX_1","tilty_1":"TiltY_1","tiltz_1":"TiltZ_1",
    "tiltx_2":"TiltX_2","tilty_2":"TiltY_2","tiltz_2":"TiltZ_2",
    "tiltx_3":"TiltX_3","tilty_3":"TiltY_3","tiltz_3":"TiltZ_3",
    "latitude":"Latitude","longitude":"Longitude"
}
cols_lower = {c.lower(): c for c in df.columns}
# check presence
for k,v in mapping_expected.items():
    if k in cols_lower:
        mapping_expected[k] = cols_lower[k]
    else:
        # keep original expected if exists exactly
        if v in df.columns:
            mapping_expected[k] = v
        else:
            mapping_expected[k] = None

# Convenience accessor
def get_col(df, key):
    col = mapping_expected.get(key.lower())
    return col if col in df.columns else None

# detect lat/lng availability
lat_col = get_col(df, "latitude")
lon_col = get_col(df, "longitude")
have_latlng = lat_col is not None and lon_col is not None and df[lat_col].notna().sum() > 0

# Parse timestamps
ts_col = None
for c in df.columns:
    if "timestamp" in c.lower():
        ts_col = c; break
if ts_col:
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

# ---------- Fetch railway geometry for Delhi->Jaipur bbox ----------
st.sidebar.header("Railway geometry (Overpass)")
if st.sidebar.button("Fetch Delhi→Jaipur railway (cached 5min)"):
    st.sidebar.write("Fetching...")

# Use bbox that contains typical Delhi-Jaipur corridor
bbox_pad = 0.2
lat_min = min(DELHI_LAT, JAIPUR_LAT) - bbox_pad
lat_max = max(DELHI_LAT, JAIPUR_LAT) + bbox_pad
lon_min = min(DELHI_LON, JAIPUR_LON) - bbox_pad
lon_max = max(DELHI_LON, JAIPUR_LON) + bbox_pad
ways = fetch_osm_railways(lat_min, lon_min, lat_max, lon_max)
if not ways:
    st.sidebar.warning("Could not fetch railway geometry from Overpass. The app will fallback to linear simulated route.")
flat_pts = flatten_way_points(ways) if ways else []
if flat_pts:
    dists = cumulative_distances(flat_pts)
    total_km = dists[-1]
else:
    # fallback to straight-line segment between Delhi and Jaipur (approx)
    flat_pts = [(DELHI_LAT, DELHI_LON), (JAIPUR_LAT, JAIPUR_LON)]
    dists = cumulative_distances(flat_pts)
    total_km = dists[-1]

# ---------- Compute approximated GPS if needed ----------
if not have_latlng:
    st.info("Latitude/Longitude missing or empty in sheet — approximating positions along Delhi→Jaipur using timestamps.")
    # We'll map each row's timestamp between first_ts and last_ts to a fraction along the route.
    if ts_col is None:
        st.error("No timestamp column found; cannot approximate GPS without timestamps.")
        st.stop()
    first_ts = df[ts_col].dropna().iloc[0]
    last_ts = df[ts_col].dropna().iloc[-1]
    if pd.isna(first_ts) or pd.isna(last_ts) or last_ts == first_ts:
        st.error("Timestamps invalid or identical; cannot approximate GPS.")
        st.stop()

    # build new columns for approximated coords
    approx_lats = []
    approx_lons = []
    eta_list = []
    for idx, row in df.iterrows():
        t = row[ts_col]
        if pd.isna(t):
            frac = 0.0
        else:
            frac = (t - first_ts).total_seconds() / (last_ts - first_ts).total_seconds()
            frac = max(0.0, min(1.0, frac))
        lat_i, lon_i = interp_point_along(flat_pts, dists, frac)
        approx_lats.append(lat_i)
        approx_lons.append(lon_i)
        # ETA: estimate remaining distance and ETA assuming constant speed = total_km / total_time
        total_time_sec = (last_ts - first_ts).total_seconds()
        if total_time_sec > 0:
            avg_speed_kps = total_km / total_time_sec  # km per second
            remaining_km = total_km * (1.0 - frac)
            eta_seconds = remaining_km / avg_speed_kps if avg_speed_kps > 0 else None
            if eta_seconds is not None:
                eta = (last_ts + timedelta(seconds=0)) if False else (datetime.now() + timedelta(seconds=eta_seconds))
            else:
                eta = None
        else:
            eta = None
        eta_list.append(eta)
    # attach to dataframe copy
    df_est = df.copy()
    df_est["Latitude_approx"] = approx_lats
    df_est["Longitude_approx"] = approx_lons
    df_est["ETA_approx"] = eta_list
else:
    df_est = df.copy()
    # ensure numeric lat/lng columns exist
    df_est["Latitude_approx"] = df_est[lat_col].astype(float)
    df_est["Longitude_approx"] = df_est[lon_col].astype(float)
    df_est["ETA_approx"] = pd.NaT

# ---------- Compute cant from TiltZ columns (if available) ----------
tz0 = get_col(df, "tiltz_0")
tz1 = get_col(df, "tiltz_1")
tz2 = get_col(df, "tiltz_2")
tz3 = get_col(df, "tiltz_3")

cant_series = []
grad_series = []
cant_deg_series = []
grad_deg_series = []

if tz0 and tz1 and tz2 and tz3:
    # compute for each row
    for _, r in df_est.iterrows():
        z0 = safe_float(r.get(tz0), np.nan)
        z1 = safe_float(r.get(tz1), np.nan)
        z2 = safe_float(r.get(tz2), np.nan)
        z3 = safe_float(r.get(tz3), np.nan)
        if np.isnan(z0) or np.isnan(z1) or np.isnan(z2) or np.isnan(z3):
            cant_series.append(np.nan); grad_series.append(np.nan); cant_deg_series.append(np.nan); grad_deg_series.append(np.nan)
        else:
            # plane fit
            try:
                X = np.array([[-0.15,0.10,1],[0.15,0.10,1],[0.15,-0.10,1],[-0.15,-0.10,1]])
                Z = np.array([z0, z1, z2, z3])
                sol,_,_,_ = np.linalg.lstsq(X, Z, rcond=None)
                a, b, c = sol
                grad_deg = math.degrees(math.atan(a))
                cant_deg = math.degrees(math.atan(b))
                grad_mm = math.tan(math.radians(grad_deg)) * GAUGE_M * 1000.0
                cant_mm = math.tan(math.radians(cant_deg)) * GAUGE_M * 1000.0
                cant_series.append(cant_mm)
                grad_series.append(grad_mm)
                cant_deg_series.append(cant_deg)
                grad_deg_series.append(grad_deg)
            except Exception:
                cant_series.append(np.nan); grad_series.append(np.nan); cant_deg_series.append(np.nan); grad_deg_series.append(np.nan)
else:
    # no TiltZ columns
    cant_series = [np.nan] * len(df_est)
    grad_series = [np.nan] * len(df_est)
    cant_deg_series = [np.nan] * len(df_est)
    grad_deg_series = [np.nan] * len(df_est)

df_est["cant_mm"] = cant_series
df_est["grad_mm"] = grad_series
df_est["cant_deg"] = cant_deg_series
df_est["grad_deg"] = grad_deg_series

# ---------- UI: top metrics ----------
latest = df_est.iloc[-1]
st.sidebar.header("Latest metrics")
st.sidebar.write("Timestamp:", latest[ts_col] if ts_col else "—")
st.sidebar.write("Latitude:", latest["Latitude_approx"])
st.sidebar.write("Longitude:", latest["Longitude_approx"])
st.sidebar.write("Cant (mm):", round(float(latest["cant_mm"]) if not np.isnan(latest["cant_mm"]) else 0.0,2))

# status
cant_val = safe_float(latest["cant_mm"], np.nan)
status = "GREEN"
if not np.isnan(cant_val) and cant_val >= PERMITTED_CANT_MM:
    status = "RED"
elif not np.isnan(cant_val) and cant_val >= WARNING_RATIO * PERMITTED_CANT_MM:
    status = "YELLOW"

st.sidebar.markdown(f"**Status:** {'🟢' if status=='GREEN' else '🟡' if status=='YELLOW' else '🔴'} {status}")

# ---------- Map display ----------
st.subheader("Map: railway + telemetry points (approximated if needed)")
m = folium.Map(location=(latest["Latitude_approx"], latest["Longitude_approx"]) if not np.isnan(latest["Latitude_approx"]) else (DELHI_LAT, DELHI_LON), zoom_start=8)

# plot railway geometry if available
if flat_pts:
    folium.PolyLine(flat_pts, color="darkblue", weight=3, opacity=0.6, tooltip="OSM railway").add_to(m)

# plot telemetry path using approximated coords
pts = list(zip(df_est["Latitude_approx"].astype(float), df_est["Longitude_approx"].astype(float)))
folium.PolyLine(pts, color="red", weight=2, opacity=0.9, tooltip="Telemetry path (approx)").add_to(m)
folium.CircleMarker(location=pts[-1], radius=6, color="black", fill=True, fill_color="red", tooltip="Current position").add_to(m)

# show ETA for destination (approx)
if df_est["ETA_approx"].notna().any():
    eta = df_est["ETA_approx"].iloc[-1]
    if eta:
        st.write("Approx ETA to destination:", eta.strftime("%Y-%m-%d %H:%M:%S"))

st_folium(m, width=900, height=500)

# ---------- Charts & tables ----------
st.subheader("Cant / Gradient history")
if not df_est["cant_mm"].isna().all():
    st.line_chart(df_est[["cant_mm","grad_mm"]].fillna(method="ffill").tail(1000))
else:
    st.info("No TiltZ columns found — cant not computed.")

st.subheader("Latest telemetry rows (tail)")
display_cols = [c for c in df_est.columns if c in (df_est.columns.tolist())]
st.dataframe(df_est[display_cols].tail(200))

# ---------- Download augmented CSV ----------
st.subheader("Download augmented CSV (contains Latitude_approx/Longitude_approx and cant/grad columns)")
buf = io.StringIO()
df_est.to_csv(buf, index=False)
st.download_button("Download augmented CSV", data=buf.getvalue(), file_name="augmented_telemetry.csv", mime="text/csv")

st.caption("Notes: This app *approximates* GPS by mapping row timestamps to fractions along the Delhi→Jaipur railway geometry. It does NOT write back to your Google Sheet; use the downloaded CSV to paste updated values into your sheet if you want them stored. If you want the app to write approximated GPS back into the sheet directly, I can add a controlled write option using Apps Script or Google Sheets API — say the word and I'll add it.")
