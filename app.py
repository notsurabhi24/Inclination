# app.py
import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium

st.set_page_config(layout="wide", page_title="Rail Cant Monitor — CSV-backed (Live)")

# ---------------------
# CONFIG (edit if you want)
# ---------------------
CSV_URL_DEFAULT = ""  # default published CSV URL (leave blank and paste in sidebar)
MAP_CACHE_TTL = 300  # seconds to cache the folium map (reduce flashing)
CSV_CACHE_TTL = 600  # seconds to cache CSV loads
TELEMETRY_POLL_SEC = 2  # seconds between telemetry updates (if you enable autorefresh)
SIMULATE_GPS_DEFAULT = True  # simulate GPS if lat/lng are missing

# Delhi & Jaipur coords (approx)
DELHI = (28.6139, 77.2090)
JAIPUR = (26.9124, 75.7873)

# ---------------------
# Helper functions
# ---------------------
@st.cache_data(ttl=CSV_CACHE_TTL)
def load_csv(csv_url: str):
    """Load CSV from URL (cached). Returns dataframe or raises."""
    if not csv_url:
        return None
    try:
        df = pd.read_csv(csv_url)
        # strip whitespace from columns
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        # bubble up error to UI
        raise RuntimeError(f"Error loading CSV: {e}")

def detect_timestamp_column(df: pd.DataFrame):
    """Return the name of a timestamp-like column if found, else None."""
    if df is None or df.empty:
        return None
    candidates = [c for c in df.columns if c.lower() in ("timestamp", "time", "ts", "datetime", "created", "date")]
    if candidates:
        return candidates[0]
    # fallback: look for columns containing 'time' or 'date'
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower():
            return c
    return None

def ensure_timestamp(df: pd.DataFrame):
    """
    Ensure df has a usable timestamp column.
    Returns (df, ts_col_name).
    If no timestamp exists, creates a synthetic 'timestamp' column using monotonic recent times.
    """
    if df is None:
        return df, None

    df = df.copy()
    ts_col = detect_timestamp_column(df)
    if ts_col:
        # try to parse, coerce invalids to NaT
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        # if all values are NaT, treat as no timestamp
        if df[ts_col].isna().all():
            ts_col = None
        else:
            # sort by timestamp to make sure latest is last
            df = df.sort_values(ts_col).reset_index(drop=True)
            return df, ts_col

    # No usable timestamp found -> generate synthetic timestamps
    n = len(df)
    if n == 0:
        return df, None
    start = pd.Timestamp.now()
    # create timestamps separated by 1 second (deterministic for a run)
    synthetic_ts = [start + pd.Timedelta(seconds=i) for i in range(n)]
    df["timestamp"] = synthetic_ts
    df = df.reset_index(drop=True)
    return df, "timestamp"

def detect_cant_column(df: pd.DataFrame):
    """Detect a column that likely holds the 'cant' telemetry value."""
    if df is None or df.empty:
        return None
    # common names
    for name in df.columns:
        if name.lower() in ("cant", "cant_mm", "cant (mm)", "cant_mm "):
            return name
    # fuzzy: column name contains 'cant' or 'inclination' or 'value'
    for name in df.columns:
        if "cant" in name.lower() or "inclination" in name.lower() or "value" in name.lower():
            return name
    # fallback: return the last numeric column
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return numeric_cols[-1]
    return None

def simulate_gps_along(n_points, start=DELHI, end=JAIPUR):
    """Return lat/lon arrays linearly interpolated between two points (inclusive)."""
    if n_points <= 0:
        return [], []
    lats = np.linspace(start[0], end[0], n_points)
    lons = np.linspace(start[1], end[1], n_points)
    return lats, lons

@st.cache_data(ttl=MAP_CACHE_TTL)
def build_map(track_coords, center=None, zoom_start=7):
    """
    Build a folium.Map containing the track polyline and start/end markers.
    This function is cached to reduce rebuilds and flashing.
    """
    if center is None:
        center = track_coords[len(track_coords)//2] if track_coords else DELHI
    m = folium.Map(location=center, zoom_start=zoom_start)
    if track_coords and len(track_coords) > 0:
        folium.PolyLine(track_coords, weight=4, opacity=0.8).add_to(m)
        folium.CircleMarker(track_coords[0], radius=6, popup="start", tooltip="start").add_to(m)
        folium.CircleMarker(track_coords[-1], radius=6, popup="end", tooltip="end").add_to(m)
    return m

# ---------------------
# Sidebar: user inputs
# ---------------------
st.sidebar.header("Configuration")
csv_url = st.sidebar.text_input("Published CSV URL (leave blank to use CSV_URL secret)", CSV_URL_DEFAULT)
simulate_gps_checkbox = st.sidebar.checkbox("Simulate GPS along Delhi→Jaipur (if no lat/lng)", value=SIMULATE_GPS_DEFAULT)
st.sidebar.markdown(f"Map refresh: cached every {MAP_CACHE_TTL//60} minutes to avoid flashing")
st.sidebar.markdown(f"Telemetry poll: every {TELEMETRY_POLL_SEC} seconds")

# Optional: autorefresh. Uncomment to enable automatic polls (the app will rerun periodically).
# from streamlit_autorefresh import st_autorefresh
# st_autorefresh(interval=TELEMETRY_POLL_SEC * 1000, key="auto_refresh")

# ---------------------
# Load CSV (cached)
# ---------------------
df = None
load_error = None
try:
    df = load_csv(csv_url) if csv_url else None
except Exception as e:
    load_error = str(e)

if load_error:
    st.sidebar.error(load_error)

# If no CSV, create empty df to avoid many checks later
if df is None:
    df = pd.DataFrame()

# ---------------------
# Normalize columns and ensure timestamp
# ---------------------
# Trim column names and remove pesky newlines
df.columns = [c.strip() for c in df.columns]

df, ts_col = ensure_timestamp(df)

# ---------------------
# Ensure cant column detection
# ---------------------
cant_col = detect_cant_column(df)
if cant_col:
    # coerce to numeric if possible
    df[cant_col] = pd.to_numeric(df[cant_col], errors="coerce")

# ---------------------
# Ensure lat/lng exist (simulate if missing)
# ---------------------
has_latlng = ("lat" in df.columns and "lng" in df.columns) or ("latitude" in df.columns and "longitude" in df.columns)
# normalize names
if "latitude" in df.columns and "longitude" in df.columns and not ("lat" in df.columns and "lng" in df.columns):
    df = df.rename(columns={"latitude": "lat", "longitude": "lng"})

if (not has_latlng) and simulate_gps_checkbox and len(df) > 0:
    lats, lons = simulate_gps_along(len(df), start=DELHI, end=JAIPUR)
    df = df.copy()
    df["lat"] = lats
    df["lng"] = lons

# Build track coords if available
if "lat" in df.columns and "lng" in df.columns and len(df) > 0:
    coords = list(zip(df["lat"].astype(float).tolist(), df["lng"].astype(float).tolist()))
else:
    coords = []

# ---------------------
# Layout: left = telemetry, middle = map, right = debug/data
# ---------------------
left_col, mid_col, right_col = st.columns([1.0, 2.0, 1.2])

with left_col:
    st.title("Live telemetry")
    # placeholders so we only update these small widgets on rerun
    ts_placeholder = st.empty()
    channel_placeholder = st.empty()
    val_placeholder = st.empty()
    status_placeholder = st.empty()
    st.markdown("---")
    st.write(f"Telemetry poll: every {TELEMETRY_POLL_SEC} s (autorefresh disabled by default)")

with mid_col:
    st.title("Map")
    map_placeholder = st.empty()

with right_col:
    st.title("Raw / preview")
    st.dataframe(df.tail(10), use_container_width=True)

# ---------------------
# Build & display cached map once per cache TTL
# ---------------------
folium_map = build_map(coords, center=coords[len(coords)//2] if coords else DELHI, zoom_start=7)
# st_folium returns an object but writing it via placeholder avoids rebuilding map code block in layout
map_placeholder.write(st_folium(folium_map, width="100%", height=480))

# ---------------------
# Telemetry display logic (lightweight)
# ---------------------
if len(df) > 0:
    latest_row = df.iloc[-1]
    # timestamp
    if ts_col and ts_col in latest_row and pd.notna(latest_row[ts_col]):
        ts_value = latest_row[ts_col]
        # display as nice formatted string
        if isinstance(ts_value, (pd.Timestamp, datetime)):
            ts_str = pd.to_datetime(ts_value).strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = str(ts_value)
    else:
        ts_str = "—"

    # channel (if available)
    channel_val = latest_row.get("channel", latest_row.get("Channel", "—"))

    # telemetry value (cant)
    latest_val = latest_row.get(cant_col, "—") if cant_col else "—"

    ts_placeholder.markdown(f"**Timestamp:** {ts_str}")
    channel_placeholder.markdown(f"**Channel:** {channel_val}")
    val_placeholder.markdown(f"**Cant (mm)**\n\n### {latest_val}")
    # status logic: example thresholds (customize)
    try:
        numeric_val = float(latest_val)
        if np.isnan(numeric_val):
            status = "NO DATA"
            status_emoji = "⚪"
        elif numeric_val < 50:
            status = "GREEN"
            status_emoji = "🟢"
        elif numeric_val < 100:
            status = "AMBER"
            status_emoji = "🟠"
        else:
            status = "RED"
            status_emoji = "🔴"
    except Exception:
        status = "UNKNOWN"
        status_emoji = "⚪"

    status_placeholder.markdown(f"**Status:** {status_emoji} {status}")
else:
    ts_placeholder.markdown("**Timestamp:** —")
    channel_placeholder.markdown("**Channel:** —")
    val_placeholder.markdown("**Cant (mm)**\n\n### —")
    status_placeholder.markdown("**Status:** ⚪ NO DATA")

# ---------------------
# Footer / tips
# ---------------------
st.markdown("---")
st.caption("Map is cached for MAP_CACHE_TTL seconds to reduce flashing. To force a fresh map, restart the app or change the URL.")
