# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt
import time
import math
import os
from datetime import datetime
from math import radians, asin, sqrt, sin, cos

st.set_page_config(layout="wide", page_title="MPU + Train Simulator — Tilt Visuals")

# -------------------------
# Config & route (edit here)
# -------------------------
CSV_FILENAME = "https://docs.google.com/spreadsheets/d/1roswHtajRP9vBGBkswVZwQ8L6yQI72zWXRTLMgnFue8/edit?gid=0#gid=0"  # local CSV the simulator writes (or upload your own)
ROUTE_STATIONS = [
    ("Mumbai CSMT", 18.939821, 72.835468),
    ("Dadar",       19.0180,    72.8481),
    ("Thane",       19.18611,   72.97583),
    ("Kalyan",      19.2437,    73.13554),
    ("Lonavala",    18.74806,   73.40722),
    ("Pune Junction",18.5289,   73.8743)
]
MAX_TILT_DISPLAY = st.sidebar.slider("Max tilt shown (°)", 20, 90, 45)

# -------------------------
# Utilities
# -------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def segment_lengths(coords):
    return [haversine(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1]) for i in range(len(coords)-1)]

def interpolate_point_along_polyline(poly_coords, frac):
    if frac <= 0:
        return poly_coords[0]
    if frac >= 1:
        return poly_coords[-1]
    seg_dists = segment_lengths(poly_coords)
    total = sum(seg_dists)
    if total == 0:
        return poly_coords[0]
    target = frac * total
    cum = 0.0
    for i, sd in enumerate(seg_dists):
        if cum + sd >= target:
            rem = target - cum
            ratio = rem / sd if sd != 0 else 0
            lat1, lon1 = poly_coords[i]
            lat2, lon2 = poly_coords[i+1]
            lat = lat1 + (lat2 - lat1) * ratio
            lon = lon1 + (lon2 - lon1) * ratio
            return (lat, lon)
        cum += sd
    return poly_coords[-1]

def compute_roll_pitch_from_acc(ax, ay, az):
    # ax,ay,az are in g units
    roll = math.degrees(math.atan2(ay, az))
    denom = math.sqrt(ay*ay + az*az)
    if denom == 0:
        pitch = 0.0
    else:
        pitch = math.degrees(math.atan2(-ax, denom))
    return round(roll, 2), round(pitch, 2)

# -------------------------
# Data loading
# -------------------------
st.title("MPU tilt visualizer + train GPS simulator")
st.markdown("Upload your MPU spreadsheet (CSV/XLSX) or let the app read a local CSV named `mpu_simulated.csv`.")

uploaded = st.file_uploader("Upload CSV / Excel (optional)", type=["csv","xlsx","xls"])
use_local_if_exists = st.checkbox("Read local CSV if available", value=True)

df = None
if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.success(f"Loaded upload: {uploaded.name}")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()
else:
    if use_local_if_exists and os.path.isfile(CSV_FILENAME):
        try:
            df = pd.read_csv(CSV_FILENAME)
            st.success(f"Loaded local CSV: {CSV_FILENAME}")
        except Exception as e:
            st.error(f"Failed to read {CSV_FILENAME}: {e}")
            st.stop()
    else:
        st.info("No file provided. Use 'Use example data' to try a demo.")
        if st.button("Use example synthetic data"):
            # build small example df
            n = 120
            t0 = datetime.now()
            timestamps = [t0 + pd.Timedelta(seconds=10*i) for i in range(n)]
            df = pd.DataFrame({
                "timestamp": timestamps,
                "tilt_x": np.sin(np.linspace(0, 6*np.pi, n)) * 0.4,
                "tilt_y": np.cos(np.linspace(0, 4*np.pi, n)) * 0.3,
                "tilt_z": 1.0 + np.random.normal(0,0.02,n),
                "tilt_x1": np.random.normal(0,0.05,n),
                "tilt_y1": np.random.normal(0,0.05,n),
                "tilt_z1": 1.0 + np.random.normal(0,0.02,n),
                "tilt_x2": np.random.normal(0,0.05,n),
                "tilt_y2": np.random.normal(0,0.05,n),
                "tilt_z2": 1.0 + np.random.normal(0,0.02,n),
                "tilt_x3": np.random.normal(0,0.05,n),
                "tilt_y3": np.random.normal(0,0.05,n),
                "tilt_z3": 1.0 + np.random.normal(0,0.02,n),
            })
        else:
            st.stop()

# ensure timestamp column exists & parsed
if df is None:
    st.error("No data frame loaded.")
    st.stop()

if "timestamp" not in df.columns:
    st.error("Data must have a 'timestamp' column.")
    st.stop()

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
if df['timestamp'].isna().any():
    st.warning("Some timestamps failed to parse; dropping those rows.")
    df = df.dropna(subset=['timestamp']).reset_index(drop=True)

df = df.sort_values('timestamp').reset_index(drop=True)

# make sure tilt columns exist (create zeros if missing)
for ch in range(4):
    base = "" if ch == 0 else str(ch)
    for axis in ['x','y','z']:
        col = f"tilt_{axis}{base}"
        if col not in df.columns:
            df[col] = 0.0

# fill lat/lon by simulation if missing or all null
poly_coords = [(lat, lon) for (_, lat, lon) in ROUTE_STATIONS]
needs_fill = ('latitude' not in df.columns) or (df['latitude'].isna().all()) or ('longitude' not in df.columns) or (df['longitude'].isna().all())
if needs_fill:
    st.info("Filling missing latitude/longitude by linearly mapping timestamps across route polyline.")
    t0 = df['timestamp'].iloc[0]
    t_end = df['timestamp'].iloc[-1]
    total_seconds = (t_end - t0).total_seconds() if (t_end != t0) else 1.0
    lats, lons = [], []
    for t in df['timestamp']:
        frac = (t - t0).total_seconds() / total_seconds
        lat, lon = interpolate_point_along_polyline(poly_coords, frac)
        lats.append(lat)
        lons.append(lon)
    df['latitude'] = lats
    df['longitude'] = lons
else:
    st.success("Using latitude/longitude from file.")

# progress percent
df['progress_pct'] = ((df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / ((df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds())) * 100
df['progress_pct'] = df['progress_pct'].clip(0,100)

# -------------------------
# Playback controls
# -------------------------
st.sidebar.header("Playback")
idx = st.sidebar.slider("Select row (time index)", 0, len(df)-1, len(df)-1, step=1)
play = st.sidebar.button("Play")
stop = st.sidebar.button("Stop")
autoplay_interval = st.sidebar.slider("Animation interval (s)", 0.2, 2.0, 0.6, step=0.1)

if 'playing' not in st.session_state:
    st.session_state.playing = False

if play:
    st.session_state.playing = True
if stop:
    st.session_state.playing = False

# If playing, run a simple loop updating idx (this blocks but is fine for small demos)
if st.session_state.playing:
    placeholder = st.empty()
    start_idx = idx
    try:
        for i in range(start_idx, len(df)):
            st.session_state.current_idx = i
            # show UI once per frame
            with placeholder.container():
                # call the same UI refresh below by setting idx
                st.experimental_rerun()
            time.sleep(autoplay_interval)
            if not st.session_state.playing:
                break
    except Exception:
        st.session_state.playing = False

# current index to show
if 'current_idx' in st.session_state and st.session_state.playing:
    idx = st.session_state.current_idx
else:
    # keep sidebar chosen idx unless playing set it
    if 'current_idx' in st.session_state and not st.session_state.playing:
        # keep last played index if user stopped
        idx = st.session_state.current_idx

row = df.loc[idx]

# -------------------------
# Layout: Map + Tilt visuals
# -------------------------
left, right = st.columns((2,1))

with left:
    st.subheader("Train map & timeline")
    st.markdown(f"**Timestamp:** {row['timestamp']} — **Progress:** {row['progress_pct']:.2f}%")
    # Map: route line + points + current marker
    route_layer = pdk.Layer(
        "PathLayer",
        data=[{"path":[[lon,lat] for lat,lon in poly_coords], "name":"route"}],
        get_path="path",
        get_width=6,
        width_min_pixels=2
    )
    points_pd = df[['latitude','longitude','timestamp','progress_pct']].copy()
    points_pd['lon'] = points_pd['longitude']
    points_pd['lat'] = points_pd['latitude']
    points_layer = pdk.Layer(
        "ScatterplotLayer",
        data=points_pd.to_dict(orient='records'),
        get_position='[lon, lat]',
        get_radius=30,
        radius_min_pixels=2,
        pickable=False,
    )
    curr = {"lon": float(row['longitude']), "lat": float(row['latitude']), "name": "current"}
    current_layer = pdk.Layer(
        "ScatterplotLayer",
        data=[curr],
        get_position='[lon, lat]',
        get_radius=80,
        radius_min_pixels=6
    )
    midpoint = [float(df['longitude'].mean()), float(df['latitude'].mean())]
    deck = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(latitude=midpoint[1], longitude=midpoint[0], zoom=8, pitch=20),
        layers=[route_layer, points_layer, current_layer],
    )
    st.pydeck_chart(deck)

    st.markdown("### Tilt over time (selected channels)")
    # line charts for roll/pitch derived from tilt_x/y/z for each MPU
    # compute roll/pitch for full df for plotting
    for ch in range(4):
        base = "" if ch == 0 else str(ch)
        ax_col = f"tilt_x{base}"
        ay_col = f"tilt_y{base}"
        az_col = f"tilt_z{base}"
        roll_list = []
        pitch_list = []
        for _, r in df[[ax_col, ay_col, az_col]].iterrows():
            roll_val, pitch_val = compute_roll_pitch_from_acc(r[ax_col], r[ay_col], r[az_col])
            roll_list.append(roll_val)
            pitch_list.append(pitch_val)
        tmp = pd.DataFrame({
            "timestamp": df['timestamp'],
            f"roll_ch{ch}": roll_list,
            f"pitch_ch{ch}": pitch_list
        }).set_index('timestamp')
        st.markdown(f"**MPU {ch}**")
        st.line_chart(tmp)

with right:
    st.subheader("Tilt (degrees) — per MPU")
    st.caption("Roll = left↔right (°). Pitch = top↔bottom (°). Dot shows (roll, pitch). Circle = ±max tilt.")
    mpu_groups = [
        ("MPU0", ("tilt_x","tilt_y","tilt_z")),
        ("MPU1", ("tilt_x1","tilt_y1","tilt_z1")),
        ("MPU2", ("tilt_x2","tilt_y2","tilt_z2")),
        ("MPU3", ("tilt_x3","tilt_y3","tilt_z3")),
    ]
    # metrics
    metrics_cols = st.columns(2)
    plots = []
    for i, (title, (cx, cy, cz)) in enumerate(mpu_groups):
        ax = float(row.get(cx, 0.0))
        ay = float(row.get(cy, 0.0))
        az = float(row.get(cz, 0.0))
        roll_deg, pitch_deg = compute_roll_pitch_from_acc(ax, ay, az)
        st.markdown(f"**{title}**")
        st.metric("Roll (°) — left/right", f"{roll_deg}°")
        st.metric("Pitch (°) — top/bottom", f"{pitch_deg}°")
        st.caption(f"Raw accel (g): ax={ax:.3f}, ay={ay:.3f}, az={az:.3f}")
        plots.append((title, roll_deg, pitch_deg))

    st.markdown("#### Tilt indicators (visual)")
    # 2x2 grid of tilt indicators
    grid_cols = st.columns(2)
    for i, (title, roll_deg, pitch_deg) in enumerate(plots):
        col = grid_cols[i % 2]
        with col:
            fig, axp = plt.subplots(figsize=(3,3))
            circle = plt.Circle((0,0), MAX_TILT_DISPLAY, fill=False, linewidth=1.2, alpha=0.7)
            axp.add_patch(circle)
            axp.axhline(0, linewidth=0.7, linestyle='--')
            axp.axvline(0, linewidth=0.7, linestyle='--')
            axp.scatter([roll_deg], [pitch_deg], s=120, zorder=5)
            axp.set_xlim(-MAX_TILT_DISPLAY - 5, MAX_TILT_DISPLAY + 5)
            axp.set_ylim(-MAX_TILT_DISPLAY - 5, MAX_TILT_DISPLAY + 5)
            axp.set_xlabel("Roll (°)")
            axp.set_ylabel("Pitch (°)")
            axp.set_title(title)
            axp.set_aspect('equal', 'box')
            txt = f"({roll_deg}°, {pitch_deg}°)"
            axp.text(0.02, 0.95, txt, transform=axp.transAxes, fontsize=9, verticalalignment='top')
            # color dot red if magnitude > threshold
            mag = math.sqrt(roll_deg**2 + pitch_deg**2)
            if mag > MAX_TILT_DISPLAY:
                # make red outline
                axp.scatter([roll_deg], [pitch_deg], s=220, facecolors='none', edgecolors='red', linewidths=2, zorder=6)
            st.pyplot(fig)
            plt.close(fig)

    # summary
    avg_roll = round(sum(r for (_, r, _) in plots)/len(plots),2)
    avg_pitch = round(sum(p for (_, _, p) in plots)/len(plots),2)
    st.info(f"Average tilt across MPUs — Roll: {avg_roll}°, Pitch: {avg_pitch}°")
    st.markdown("**Absolute tilt magnitude** (sqrt(roll² + pitch²)) per last sample:")
    mags = [round(math.sqrt(r**2 + p**2), 2) for (_, r, p) in plots]
    for i, m in enumerate(mags):
        st.write(f"MPU{i}: {m}°")

# -------------------------
# Footer: data preview and download
# -------------------------
st.markdown("---")
st.subheader("Data preview & export")
st.dataframe(df.head(50))
if st.button("Download filled CSV"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Click to download CSV", data=csv, file_name="mpu_filled_for_streamlit.csv", mime="text/csv")

st.caption("Tip: If you run the simulator script (writes to 'mpu_simulated.csv'), enable 'Read local CSV' and click Play to animate the simulated trip.")
