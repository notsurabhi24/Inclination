# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import math
import pydeck as pdk
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ---------------- CONFIG ----------------
# If you want to keep the URL secret on Streamlit Cloud, set st.secrets["GET_URL"] instead.
GET_URL = st.secrets.get("GET_URL", "https://script.google.com/macros/s/AKfycbxJPe3pekL1vPY8achwH39WbUWfldCKXsTRPmz-n30GdTzo7Dg1yj53DgrtzUchsEJ5/exec")

POLL_INTERVAL_MS = 1000   # poll every 1s
SMOOTH_WINDOW = 3         # smoothing window for animation
MAX_HISTORY = 800         # keep this many rows for history charts
# ----------------------------------------

st.set_page_config(layout="wide", page_title="Rail Cant Monitor")
st.title("Live Rail Cant & Gradient Monitor")

# auto refresh (server-side)
st_autorefresh(interval=POLL_INTERVAL_MS, limit=0, key="autorefresh")

@st.cache_data(ttl=2)
def fetch_data():
    r = requests.get(GET_URL, timeout=8)
    r.raise_for_status()
    arr = r.json()
    if not arr:
        return pd.DataFrame()
    df = pd.DataFrame(arr)
    # normalize column names: strip whitespace
    df.columns = [c.strip() for c in df.columns]
    # coerce numeric fields we expect
    for col in ["lat","lng","FL_z","FR_z","BR_z","BL_z","cant_deg","gradient_deg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp_ist" in df.columns:
        df["timestamp_ist"] = pd.to_datetime(df["timestamp_ist"], errors="coerce")
    return df

df = fetch_data()

if df.empty:
    st.warning("No telemetry yet — the sheet might be empty or GET_URL is incorrect.")
    st.stop()

# trim history
df = df.tail(MAX_HISTORY).reset_index(drop=True)
latest = df.iloc[-1]

# smoothing helpers stored in session state
if "sm_roll" not in st.session_state:
    st.session_state.sm_roll = []
if "sm_pitch" not in st.session_state:
    st.session_state.sm_pitch = []

def push_smooth(value, key, window=SMOOTH_WINDOW):
    arr = getattr(st.session_state, key)
    arr.append(float(value))
    if len(arr) > window:
        arr.pop(0)
    setattr(st.session_state, key, arr)
    return sum(arr) / max(1, len(arr))

# pick columns (allow flexible names)
lat_col = "lat" if "lat" in df.columns else ( "latitude" if "latitude" in df.columns else None )
lng_col = "lng" if "lng" in df.columns else ( "longitude" if "longitude" in df.columns else None )
cant_col = "cant_deg" if "cant_deg" in df.columns else ("cant" if "cant" in df.columns else None)
grad_col = "gradient_deg" if "gradient_deg" in df.columns else ("gradient" if "gradient" in df.columns else None)

cur_cant = latest.get(cant_col, 0.0) if cant_col else 0.0
cur_grad = latest.get(grad_col, 0.0) if grad_col else 0.0

smooth_roll = push_smooth(cur_cant, "sm_roll")
smooth_pitch = push_smooth(cur_grad, "sm_pitch")

# Layout: Map left, visual right
col_map, col_vis = st.columns([2,1])

with col_map:
    st.subheader("Map — latest position & path")
    if lat_col and lng_col and not df[[lat_col,lng_col]].isnull().all().any():
        latest_lat = float(latest[lat_col])
        latest_lng = float(latest[lng_col])
        # build path (lng, lat) pairs
        path = [[float(r[lng_col]), float(r[lat_col])] for _, r in df[[lat_col,lng_col]].dropna().iterrows()]
        deck = pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=latest_lat, longitude=latest_lng, zoom=17, pitch=35),
            layers=[
                pdk.Layer("ScatterplotLayer", data=[{"lat": latest_lat, "lng": latest_lng}], get_position='[lng, lat]', get_radius=10),
                pdk.Layer("PathLayer", data=[{"path": path}], get_path="path", width_scale=20, width_min_pixels=2)
            ],
            map_style='light'
        )
        st.pydeck_chart(deck)
    else:
        st.info("No valid GPS columns found (expected 'lat'/'lng').")

with col_vis:
    st.subheader("Live metrics")
    st.metric("Cant (deg, left-right)", f"{smooth_roll:.2f}")
    st.metric("Gradient (deg, front-back)", f"{smooth_pitch:.2f}")
    st.write("Latest corner Z-values (g):")
    corner_keys = [k for k in ["FL_z","FR_z","BR_z","BL_z"] if k in df.columns]
    if corner_keys:
        st.table(pd.DataFrame([latest[k] for k in corner_keys], index=corner_keys, columns=["g"]).T)
    else:
        st.write("Corner Z columns (FL_z,FR_z,BR_z,BL_z) not found in sheet.")

# Animated 3D board
st.subheader("3D board animation (cant = roll, gradient = pitch)")

def make_board_mesh(pitch_deg, roll_deg, yaw_deg=0):
    verts = [
        (-1.5,-1,-0.05),(1.5,-1,-0.05),(1.5,1,-0.05),(-1.5,1,-0.05),
        (-1.5,-1, 0.05),(1.5,-1,0.05),(1.5,1,0.05),(-1.5,1,0.05)
    ]
    faces = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(1,2,6,5),(0,3,7,4)]
    rp = math.radians(pitch_deg); rr = math.radians(roll_deg); ry = math.radians(yaw_deg)
    Rx = [[1,0,0],[0,math.cos(rp),-math.sin(rp)],[0,math.sin(rp),math.cos(rp)]]
    Ry = [[math.cos(rr),0,math.sin(rr)],[0,1,0],[-math.sin(rr),0,math.cos(rr)]]
    Rz = [[math.cos(ry),-math.sin(ry),0],[math.sin(ry),math.cos(ry),0],[0,0,1]]
    def matmul(A,B):
        return [[sum(A[i][m]*B[m][j] for m in range(3)) for j in range(3)] for i in range(3)]
    R = matmul(Rz, matmul(Ry, Rx))
    xr,yr,zr = [],[],[]
    for x,y,z in verts:
        xr.append(R[0][0]*x + R[0][1]*y + R[0][2]*z)
        yr.append(R[1][0]*x + R[1][1]*y + R[1][2]*z)
        zr.append(R[2][0]*x + R[2][1]*y + R[2][2]*z)
    i,j,k = [],[],[]
    for a,b,c,d in faces:
        i += [a,a]; j += [b,c]; k += [c,d]
    return xr,yr,zr,i,j,k

xr,yr,zr,i,j,k = make_board_mesh(smooth_pitch, smooth_roll)
fig = go.Figure(data=[go.Mesh3d(x=xr,y=yr,z=zr,i=i,j=j,k=k,opacity=0.6)])
fig.update_layout(scene=dict(aspectmode='data', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                  margin=dict(l=0,r=0,t=0,b=0), height=420)
st.plotly_chart(fig, use_container_width=True)

# History charts
st.subheader("Recent history")
time_col = "timestamp_ist" if "timestamp_ist" in df.columns else df.columns[0]
hist = df[[time_col] + [c for c in ["cant_deg","gradient_deg"] if c in df.columns]].copy()
hist[time_col] = pd.to_datetime(hist[time_col], errors="coerce")
hist = hist.dropna(subset=[time_col]).tail(300)

hist_fig = go.Figure()
if "cant_deg" in hist.columns:
    hist_fig.add_trace(go.Scatter(x=hist[time_col], y=hist["cant_deg"], name="Cant (deg)"))
if "gradient_deg" in hist.columns:
    hist_fig.add_trace(go.Scatter(x=hist[time_col], y=hist["gradient_deg"], name="Gradient (deg)"))
hist_fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=300)
st.plotly_chart(hist_fig, use_container_width=True)

with st.expander("Recent raw rows"):
    st.dataframe(df.tail(200))

st.caption("Data source: your Google Sheet (Apps Script doGet). Poll interval: 1s.")
