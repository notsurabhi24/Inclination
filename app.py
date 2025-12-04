# app.py
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium

st.set_page_config(layout="wide", page_title="Rail Cant Monitor — CSV-backed (Live)")

# ---------------------
# CONFIG
# ---------------------
CSV_URL_DEFAULT = ""
MAP_CACHE_TTL = 300       # seconds to cache the folium map
CSV_CACHE_TTL = 600       # cache CSV loads
TELEMETRY_POLL_SEC = 2    # used for wording only (autorefresh commented)
SIMULATE_GPS_DEFAULT = True
USE_NOW_AS_LATEST_DEFAULT = True

# Approximate railway route station coords (Delhi -> Jaipur). These are station-like anchors,
# used to create a railway-like interpolated path if CSV lacks real GPS.
RAILWAY_STATIONS = [
    (28.6406, 77.2195),  # New Delhi area
    (28.1970, 76.6166),  # Rewari-ish
    (27.8866, 76.6061),  # Alwar-ish outskirts
    (27.0238, 76.3932),  # near Bandikui / Dausa-ish
    (26.9124, 75.7873),  # Jaipur
]

# ---------------------
# HELPERS
# ---------------------
@st.cache_data(ttl=CSV_CACHE_TTL)
def load_csv(csv_url: str):
    """Load CSV from URL (cached). Returns dataframe or raises."""
    if not csv_url:
        return None
    df = pd.read_csv(csv_url)
    df.columns = [c.strip() for c in df.columns]
    return df

def detect_timestamp_column(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    candidates = [c for c in df.columns if c.lower() in ("timestamp", "time", "ts", "datetime", "created", "date")]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower():
            return c
    return None

def detect_cant_column(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    for name in df.columns:
        if name.lower() in ("cant", "cant_mm", "cant (mm)"):
            return name
    for name in df.columns:
        if "cant" in name.lower() or "inclination" in name.lower() or "value" in name.lower():
            return name
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return numeric_cols[-1]
    return None

def interpolate_along_stations(n_points, stations):
    """Interpolate n_points along the polyline formed by stations (keeps order)."""
    if n_points <= 0:
        return []
    # compute segment lengths and distribute points proportionally across segments
    segs = list(zip(stations[:-1], stations[1:]))
    seg_lengths = []
    for (a, b) in segs:
        seg_lengths.append(((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5)
    total = sum(seg_lengths)
    coords = []
    if total == 0:
        # degenerate -> repeat single point
        coords = [stations[0]] * n_points
        return coords
    # allocate integer counts per segment
    allocated = []
    for L in seg_lengths:
        allocated.append(max(1, int(round(n_points * (L / total)))))
    # adjust allocated to sum exactly n_points
    diff = sum(allocated) - n_points
    i = 0
    while diff != 0:
        if diff > 0:
            # remove one from segments with allocated>1
            if allocated[i] > 1:
                allocated[i] -= 1
                diff -= 1
        else:
            allocated[i] += 1
            diff += 1
        i = (i + 1) % len(allocated)
    # now interpolate per-segment
    for idx, ((lat1, lon1), (lat2, lon2)) in enumerate(segs):
        count = allocated[idx]
        for k in range(count):
            t = k / max(1, count - 1) if count > 1 else 0
            lat = lat1 + (lat2 - lat1) * t
            lon = lon1 + (lon2 - lon1) * t
            coords.append((lat, lon))
    # if we overshot or undershot, trim or pad
    if len(coords) > n_points:
        coords = coords[:n_points]
    while len(coords) < n_points:
        coords.append(stations[-1])
    return coords

@st.cache_data(ttl=MAP_CACHE_TTL)
def build_map(track_coords, center=None, zoom_start=7):
    if center is None:
        center = track_coords[len(track_coords)//2] if track_coords else RAILWAY_STATIONS[0]
    m = folium.Map(location=center, zoom_start=zoom_start)
    if track_coords:
        folium.PolyLine(track_coords, weight=4, opacity=0.85).add_to(m)
        folium.CircleMarker(track_coords[0], radius=6, popup="start", tooltip="start").add_to(m)
        folium.CircleMarker(track_coords[-1], radius=6, popup="end", tooltip="end").add_to(m)
    return m

# ---------------------
# UI: Sidebar
# ---------------------
st.sidebar.header("Configuration")
csv_url = st.sidebar.text_input("Published CSV URL (leave blank to use secret)", CSV_URL_DEFAULT)
simulate_gps_checkbox = st.sidebar.checkbox("Simulate GPS along railway (if no lat/lng)", value=SIMULATE_GPS_DEFAULT)
use_now_as_latest = st.sidebar.checkbox("Treat latest row timestamp as RIGHT NOW (override)", value=USE_NOW_AS_LATEST_DEFAULT)
st.sidebar.markdown(f"Map cached for {MAP_CACHE_TTL//60} minutes to avoid flashing.")
st.sidebar.markdown("Raw preview is hidden by default — expand 'Raw CSV / Debug' to view it.")

# ---------------------
# Load CSV (cached)
# ---------------------
load_error = None
df = None
try:
    df = load_csv(csv_url) if csv_url else None
except Exception as e:
    load_error = str(e)

if load_error:
    st.sidebar.error(load_error)

if df is None:
    df = pd.DataFrame()

# Normalize column names whitespace
df.columns = [c.strip() for c in df.columns]

# ---------------------
# TIMESTAMP handling: force latest = now if requested, otherwise detect or synthesize
# ---------------------
ts_col = detect_timestamp_column(df)
if ts_col:
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    # if all invalid, treat as missing
    if df[ts_col].isna().all():
        ts_col = None
    else:
        df = df.sort_values(ts_col).reset_index(drop=True)

# If no timestamp column or user wants now-as-latest override
if ts_col is None or use_now_as_latest:
    n = len(df)
    if n > 0:
        # create synthetic timestamps such that the last row = now (current time)
        now = pd.Timestamp.now()
        # assign timestamps spaced 1 second apart ending at now
        synthetic = [now - pd.Timedelta(seconds=(n - 1 - i)) for i in range(n)]
        df = df.copy()
        df["timestamp"] = synthetic
        ts_col = "timestamp"
    else:
        ts_col = None

# ---------------------
# CANT detection and formatting
# ---------------------
cant_col = detect_cant_column(df)
if cant_col:
    df[cant_col] = pd.to_numeric(df[cant_col], errors="coerce")

# ---------------------
# GPS / TRACK handling
# ---------------------
# normalize lat/lng names
if "latitude" in df.columns and "longitude" in df.columns and not ("lat" in df.columns and "lng" in df.columns):
    df = df.rename(columns={"latitude": "lat", "longitude": "lng"})

has_latlng = ("lat" in df.columns and "lng" in df.columns)

if not has_latlng and simulate_gps_checkbox and len(df) > 0:
    # Make a railway-like track by interpolating between RAILWAY_STATIONS
    coords = interpolate_along_stations(len(df), RAILWAY_STATIONS)
    df = df.copy()
    df["lat"] = [c[0] for c in coords]
    df["lng"] = [c[1] for c in coords]
else:
    if "lat" in df.columns and "lng" in df.columns and len(df) > 0:
        coords = list(zip(df["lat"].astype(float).tolist(), df["lng"].astype(float).tolist()))
    else:
        coords = []

# ---------------------
# LAYOUT
# ---------------------
left_col, mid_col, right_col = st.columns([1.0, 2.0, 1.2])

with left_col:
    st.title("Live telemetry")
    ts_placeholder = st.empty()
    chan_placeholder = st.empty()
    val_placeholder = st.empty()
    status_placeholder = st.empty()
    st.markdown("---")
    st.write(f"Polling interval: {TELEMETRY_POLL_SEC}s (autorefresh disabled by default)")

with mid_col:
    st.title("Map")
    # render map inside the middle column — calling st_folium here will render map (we don't display returned dict)
    folium_map = build_map(coords, center=coords[len(coords)//2] if coords else RAILWAY_STATIONS[0], zoom_start=7)
    st_folium(folium_map, width="100%", height=520)

with right_col:
    st.title("Controls")
    st.markdown("Use the sidebar to change CSV URL, enable GPS simulation, or toggle timestamp override.")
    with st.expander("Raw CSV / Debug", expanded=False):
        st.dataframe(df.head(200), use_container_width=True)

# ---------------------
# TELEMETRY DISPLAY (friendly)
# ---------------------
if len(df) > 0:
    latest_row = df.iloc[-1]
    # timestamp formatting
    if ts_col and ts_col in latest_row and pd.notna(latest_row[ts_col]):
        ts_val = pd.to_datetime(latest_row[ts_col])
        ts_str = ts_val.strftime("%Y-%m-%d %H:%M:%S")
    else:
        ts_str = "—"

    # channel
    channel_val = latest_row.get("channel", latest_row.get("Channel", "—"))

    # cant value
    if cant_col:
        latest_val_raw = latest_row.get(cant_col, None)
        try:
            latest_val = float(latest_val_raw)
            latest_val_str = f"{latest_val:.2f}"
        except Exception:
            latest_val = None
            latest_val_str = "—"
    else:
        latest_val = None
        latest_val_str = "—"

    ts_placeholder.markdown(f"**Timestamp:** {ts_str}")
    chan_placeholder.markdown(f"**Channel:** {channel_val}")
    val_placeholder.markdown(f"**Cant (mm)**\n\n### {latest_val_str}")

    # status logic with clear labels
    if latest_val is None or np.isnan(latest_val):
        status_placeholder.markdown("**Status:** ⚪ NO DATA")
    else:
        if latest_val < 50:
            status_placeholder.markdown("**Status:** 🟢 GREEN")
        elif latest_val < 100:
            status_placeholder.markdown("**Status:** 🟠 AMBER")
        else:
            status_placeholder.markdown("**Status:** 🔴 RED")
else:
    ts_placeholder.markdown("**Timestamp:** —")
    chan_placeholder.markdown("**Channel:** —")
    val_placeholder.markdown("**Cant (mm)**\n\n### —")
    status_placeholder.markdown("**Status:** ⚪ NO DATA")

st.markdown("---")
st.caption("Map is cached for a short time to prevent flashing. To force a fresh map, restart the app or change the CSV URL.")
