# app_use_exec_streamlit.py
# Streamlit dashboard that reads telemetry from your Apps Script /exec URL (primary),
# refreshes readings every 2s, fetches railway geometry every 5min (cached),
# computes cant from 4 corner MPUs if present, shows map + tilt visualization,
# and importantly: fails gracefully and shows the raw server response if /exec returns non-JSON.

import streamlit as st
import requests, math, itertools
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List
from streamlit_autorefresh import st_autorefresh

# ---------------- CONFIG ----------------
GET_URL_DEFAULT = "https://script.google.com/macros/s/AKfycbyJqNgJLhliYxvv8jY_sB3RRuZQ7yuLuOuYsTyKDVoE2jbGW0id94aiespkOg4aO2eB/exec"
READ_REFRESH_MS = 2000      # readings refresh (2s)
MAP_REFRESH_SEC = 300       # rail geometry refresh (5 min)
PERMITTED_CANT_MM = 165.0
WARNING_RATIO = 0.8
GAUGE_M = 1.676
OVERPASS_TIMEOUT = 15

st.set_page_config(layout="wide", page_title="Rail Cant Monitor (AppsScript /exec)", initial_sidebar_state="collapsed")
st.title("Rail Cant Monitor")

# small auto refresh
st_autorefresh(interval=READ_REFRESH_MS, limit=0, key="autorefresh")

# ---------------- Sidebar: let user set /exec (or keep default) ----------------
st.sidebar.header("Data source")
get_url = st.sidebar.text_input("Apps Script /exec URL (will use this to read the sheet)", value=GET_URL_DEFAULT)
st.sidebar.caption("Make sure the Web app is deployed: Execute as = Me, Who has access = Anyone with link")

# optional: let user paste spreadsheet CSV published url as fallback (rarely needed)
csv_fallback = st.sidebar.text_input("Published CSV URL (optional, fallback)", value="")

# ----------------- utilities -----------------
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
    a1 = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a1), math.sqrt(1-a1))

@st.cache_data(ttl=MAP_REFRESH_SEC)
def fetch_railways_overpass(lat_min: float, lon_min: float, lat_max: float, lon_max: float) -> List[List[Tuple[float,float]]]:
    bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    query = f"""
    [out:json][timeout:{OVERPASS_TIMEOUT}];
    ( way["railway"~"rail|light_rail|railway"]({bbox}); );
    out body;
    >;
    out skel qt;
    """
    try:
        r = requests.post("https://overpass-api.de/api/interpreter", data=query.strip(), timeout=OVERPASS_TIMEOUT)
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

def fetch_from_exec(url: str):
    """Call the Apps Script /exec URL and return a pandas DataFrame.
       If response is non-JSON or parsing fails, show raw response and stop the app (safe behavior)."""
    try:
        r = requests.get(url, timeout=12)
    except Exception as e:
        st.error(f"Network error contacting /exec: {e}")
        st.stop()
    ct = r.headers.get("content-type", "")
    # If Apps Script returned HTML (login / drive / page not found), show it
    if "application/json" not in ct.lower():
        st.error(f"/exec returned non-JSON (status {r.status_code}, content-type: {ct}). See raw response below (first 2000 chars):")
        st.code(r.text[:2000])
        # offer fallback to published CSV if provided
        if csv_fallback:
            st.info("Attempting fallback to published CSV (you supplied a CSV fallback in sidebar).")
            try:
                df = pd.read_csv(csv_fallback)
                return df
            except Exception as e:
                st.error("CSV fallback failed: " + str(e))
        st.stop()
    # try parse JSON
    try:
        arr = r.json()
    except Exception as e:
        st.error("Failed to parse JSON from /exec. Raw response (first 2000 chars):")
        st.code(r.text[:2000])
        st.stop()
    if not isinstance(arr, list):
        st.error("/exec returned JSON but not an array. Raw JSON snippet:")
        st.code(str(arr)[:2000])
        st.stop()
    df = pd.DataFrame(arr)
    return df

# ----------------- Main: fetch telemetry from /exec -----------------
df = fetch_from_exec(get_url)

if df.empty:
    st.info("Sheet empty (no rows yet).")
    st.stop()

# Normalize column names
df.columns = [c.strip() for c in df.columns]

# Coerce numeric columns if present
for c in ["lat","lng","cant_mm","grad_mm","cant_deg","grad_deg","FL_z","FR_z","BR_z","BL_z"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
if "timestamp_ist" in df.columns:
    df["timestamp_ist"] = pd.to_datetime(df["timestamp_ist"], errors="coerce")

# Latest row and possible past
latest = df.iloc[-1].to_dict()
past = df.iloc[-2].to_dict() if len(df) >= 2 else None

# Safe interpolation (only if timestamps and lat/lng exist)
def interp_pos_safe(past_row, now_row, current_time):
    try:
        if past_row is None: return safe_float(now_row.get("lat")), safe_float(now_row.get("lng"))
        t0 = pd.to_datetime(past_row.get("timestamp_ist", None)); t1 = pd.to_datetime(now_row.get("timestamp_ist", None))
        if pd.isna(t0) or pd.isna(t1) or t1 <= t0: return safe_float(now_row.get("lat")), safe_float(now_row.get("lng"))
        lat0 = safe_float(past_row.get("lat", None)); lat1 = safe_float(now_row.get("lat", None))
        lng0 = safe_float(past_row.get("lng", None)); lng1 = safe_float(now_row.get("lng", None))
        if any(v is None for v in [lat0,lat1,lng0,lng1]): return safe_float(now_row.get("lat")), safe_float(now_row.get("lng"))
        frac = (current_time - t0).total_seconds() / max(1.0, (t1 - t0).total_seconds())
        frac = max(0.0, min(1.0, frac))
        return lat0 + (lat1 - lat0)*frac, lng0 + (lng1 - lng0)*frac
    except Exception:
        return safe_float(now_row.get("lat")), safe_float(now_row.get("lng"))

cur_time = pd.to_datetime(datetime.now())
interp_lat, interp_lng = interp_pos_safe(past, latest, cur_time)

# Compute cant from corners if available
have_corners = all(k in df.columns for k in ["FL_z","FR_z","BR_z","BL_z"])
if have_corners:
    # plane fit: z = a*x + b*y + c -> convert to cant/gradient
    X=[]; Y=[]; Z=[]
    BOARD_W = 0.30; BOARD_H = 0.20
    POS = {"FL":(-BOARD_W/2, BOARD_H/2), "FR":(BOARD_W/2, BOARD_H/2), "BR":(BOARD_W/2, -BOARD_H/2), "BL":(-BOARD_W/2, -BOARD_H/2)}
    for k,(x,y) in POS.items():
        X.append(x); Y.append(y); Z.append(safe_float(latest.get(k+"_z"), np.nan) if latest.get(k+"_z") is not None else safe_float(latest.get(k), np.nan))
    try:
        A = np.vstack([X,Y,np.ones(len(X))]).T
        sol,_,_,_ = np.linalg.lstsq(A, Z, rcond=None)
        a,b,_ = sol
        grad_deg = math.degrees(math.atan(a)); cant_deg = math.degrees(math.atan(b))
        grad_mm = math.tan(math.radians(grad_deg)) * GAUGE_M * 1000.0
        cant_mm = math.tan(math.radians(cant_deg)) * GAUGE_M * 1000.0
    except Exception:
        cant_mm = np.nan; grad_mm = np.nan; cant_deg = np.nan; grad_deg = np.nan
else:
    # fallback if cant_mm present as column
    if "cant_mm" in df.columns:
        cant_mm = safe_float(latest.get("cant_mm"), np.nan)
        cant_deg = math.degrees(math.atan((cant_mm/1000.0)/GAUGE_M)) if not np.isnan(cant_mm) else np.nan
    else:
        cant_mm = np.nan; cant_deg = np.nan
    grad_mm = safe_float(latest.get("grad_mm"), np.nan)
    grad_deg = math.degrees(math.atan((grad_mm/1000.0)/GAUGE_M)) if not np.isnan(grad_mm) else np.nan

# Status color
status = "GREEN"
warn_thresh = WARNING_RATIO * PERMITTED_CANT_MM
if not np.isnan(cant_mm) and cant_mm >= PERMITTED_CANT_MM:
    status = "RED"
elif not np.isnan(cant_mm) and cant_mm >= warn_thresh:
    status = "YELLOW"

# Build map path: use lat/lng columns if present
map_pts = []
if "lat" in df.columns and "lng" in df.columns:
    pts_df = df[["lat","lng"]].dropna().tail(5000)
    map_pts = [{"latitude": float(r["lat"]), "longitude": float(r["lng"])} for _,r in pts_df.iterrows()]

# Try to fetch railway ways (cached)
if map_pts:
    lats = [p["latitude"] for p in map_pts]; lngs = [p["longitude"] for p in map_pts]
    lat_min, lat_max = min(lats), max(lats)
    lng_min, lng_max = min(lngs), max(lngs)
    pad_lat = max(0.01, (lat_max-lat_min)*0.2 + 0.005)
    pad_lng = max(0.01, (lng_max-lng_min)*0.2 + 0.005)
    ways = fetch_railways_overpass(lat_min-pad_lat, lng_min-pad_lng, lat_max+pad_lat, lng_max+pad_lng)
else:
    ways = []

if not ways and map_pts:
    # fallback: create ways from map_pts
    ways = [[(p["latitude"], p["longitude"]) for p in map_pts]]

# ---------------- Present UI ----------------
col_map, col_metrics = st.columns([3,1])

with col_map:
    st.subheader("Map (real track if found; fallback to telemetry path)")
    if ways:
        flat = list(itertools.chain.from_iterable(ways))
        map_df = pd.DataFrame([{"latitude":lat,"longitude":lon} for lat,lon in flat]).drop_duplicates().tail(5000)
        st.map(map_df)
    elif map_pts:
        st.map(pd.DataFrame(map_pts))
    else:
        st.info("No GPS in sheet (or no valid lat/lng) — map unavailable.")
    st.markdown(f"**Smooth position:** {interp_lat:.6f}, {interp_lng:.6f}")
    st.write(f"Latest timestamp: {latest.get('timestamp_ist', '<no timestamp>')}")

with col_metrics:
    st.subheader("Readings & status")
    st.metric("Cant (mm)", f"{cant_mm:.2f}" if not np.isnan(cant_mm) else "—")
    st.metric("Cant (deg)", f"{cant_deg:.3f}°" if not np.isnan(cant_deg) else "—")
    st.metric("Gradient (deg)", f"{grad_deg:.3f}°" if not np.isnan(grad_deg) else "—")
    color = "#2ECC71" if status=="GREEN" else "#F1C40F" if status=="YELLOW" else "#E74C3C"
    st.markdown(f"<div style='display:flex;gap:10px;align-items:center'><div style='width:18px;height:18px;background:{color};border-radius:6px'></div><div><b>Status:</b> {status}</div></div>", unsafe_allow_html=True)
    if status=="RED": st.error("Threshold exceeded!")
    elif status=="YELLOW": st.warning("Approaching threshold.")

st.subheader("Recent telemetry (latest 200 rows)")
st.dataframe(df.tail(200))

st.subheader("Tilt visualization")
def render_svg(c_deg, p_deg):
    if math.isnan(c_deg): c_deg = 0.0
    if math.isnan(p_deg): p_deg = 0.0
    c = max(-20, min(20, c_deg)); p = max(-20, min(20, p_deg))
    svg = f'''
    <div style="display:flex;justify-content:center">
      <svg width="260" height="180" viewBox="0 0 260 180">
        <rect x="40" y="40" width="180" height="100" rx="12" ry="12" style="fill:#1f77b4;fill-opacity:0.12;stroke:#1f77b4;stroke-width:3;transform-origin:130px 90px;transform: rotate({-c}deg) skewX({p/2}deg);"/>
      </svg>
    </div>
    <div style="text-align:center">Roll (cant): <b>{c}°</b> | Pitch (grad): <b>{p}°</b></div>
    '''
    return svg
st.markdown(render_svg(cant_deg if not np.isnan(cant_deg) else 0.0, grad_deg if not np.isnan(grad_deg) else 0.0), unsafe_allow_html=True)

st.caption("If /exec returns a non-JSON page, the app displays the raw response so you can fix Apps Script deployment (Execute as: Me / Anyone with link). If you want, paste your Apps Script code in chat and I'll inspect/fix it so the /exec returns JSON reliably.")
