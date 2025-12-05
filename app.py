import streamlit as st
import pandas as pd
import numpy as np
import math, requests, io
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from urllib.parse import urlparse, parse_qs

# ---------- CONFIG ----------
READ_REFRESH_MS = 2000       # Refresh every 2 seconds
GAUGE_M = 1.676              # Indian broad gauge
PERMITTED_CANT_MM = 165.0
WARNING_RATIO = 0.8

st.set_page_config(layout="wide", page_title="Rail Tilt Monitor")
st.title("Rail Tilt Monitor — Live Tilt Analysis (4 MPU Sensors)")

# auto-refresh
st_autorefresh(interval=READ_REFRESH_MS, limit=0, key="autorefresh")

# ---------- HELPERS ----------
def safe_float(x, default=np.nan):
    try:
        if x is None: return default
        return float(x)
    except:
        return default

def to_export_csv_url(u: str) -> str:
    """Convert Google Sheets share URL → CSV export URL."""
    if not u:
        return u
    u = u.strip()
    if "export?format=csv" in u:
        return u

    if "spreadsheets/d/" in u:
        try:
            path = u.split("/spreadsheets/d/")[1]
            sid = path.split("/")[0]
            gid = "0"
            if "gid=" in u:
                gid = u.split("gid=")[1]
            return f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid}"
        except:
            return u
    return u

def load_csv_robust(url: str) -> pd.DataFrame:
    csv_url = to_export_csv_url(url)
    r = requests.get(csv_url, timeout=20)
    text = r.text
    if "<html" in text.lower():
        raise RuntimeError("URL returned HTML, not CSV. Make sheet public or use export=csv link.")
    return pd.read_csv(io.StringIO(text))

# ---------- SIDEBAR INPUT ----------
st.sidebar.header("Data Source")
src_url = st.sidebar.text_input("Paste Google Sheets URL:", value="")
if not src_url:
    st.sidebar.info("Paste the sheet link here.")
    st.stop()

# ---------- LOAD SHEET ----------
try:
    df = load_csv_robust(src_url)
    df.columns = [c.strip() for c in df.columns]
except Exception as e:
    st.error("Failed to load CSV.")
    st.write(e)
    st.stop()

if df.empty:
    st.warning("Sheet is empty.")
    st.stop()

cols_lower = {c.lower(): c for c in df.columns}
def col(name): return cols_lower.get(name.lower())

# Find timestamp column
ts_col = None
for c in df.columns:
    if "time" in c.lower():
        ts_col = c
        break

# ---------- COMPUTE METRICS ----------
mean_rolls=[]; mean_pitch=[]; left_mean=[]; right_mean=[]
cant_mm_list=[]; diff_mm_list=[]

for _, row in df.iterrows():
    tx0,tx1,tx2,tx3 = [safe_float(row[col(f"tiltx_{i}")]) for i in range(4)]
    ty0,ty1,ty2,ty3 = [safe_float(row[col(f"tilty_{i}")]) for i in range(4)]

    rolls = np.array([tx0, tx1, tx2, tx3], dtype=float)
    pitches = np.array([ty0, ty1, ty2, ty3], dtype=float)

    mean_roll_deg = float(np.nanmean(rolls))
    mean_pitch_deg = float(np.nanmean(pitches))

    L = float(np.nanmean([tx0, tx3]))
    R = float(np.nanmean([tx1, tx2]))

    mean_roll_rad = math.radians(mean_roll_deg)
    diff_rad = math.radians(L - R)

    cant_mm = math.tan(mean_roll_rad) * GAUGE_M * 1000
    diff_mm = math.tan(diff_rad) * GAUGE_M * 1000

    mean_rolls.append(mean_roll_deg)
    mean_pitch.append(mean_pitch_deg)
    left_mean.append(L)
    right_mean.append(R)
    cant_mm_list.append(cant_mm)
    diff_mm_list.append(diff_mm)

df["mean_roll_deg"] = mean_rolls
df["mean_pitch_deg"] = mean_pitch
df["left_mean_roll_deg"] = left_mean
df["right_mean_roll_deg"] = right_mean
df["cant_mm"] = cant_mm_list
df["left_right_diff_mm"] = diff_mm_list

latest = df.iloc[-1]

# Timestamp
ts_display = str(latest[ts_col]) if ts_col else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------- UI ----------
left, right = st.columns([2,1])

with left:
    st.subheader("Live Telemetry")
    st.write("Timestamp:", ts_display)

    st.table({
        "Sensor": [
            "TiltX_0","TiltY_0","TiltZ_0",
            "TiltX_1","TiltY_1","TiltZ_1",
            "TiltX_2","TiltY_2","TiltZ_2",
            "TiltX_3","TiltY_3","TiltZ_3"
        ],
        "Value": [
            latest.get(col("tiltx_0")), latest.get(col("tilty_0")), latest.get(col("tiltz_0")),
            latest.get(col("tiltx_1")), latest.get(col("tilty_1")), latest.get(col("tiltz_1")),
            latest.get(col("tiltx_2")), latest.get(col("tilty_2")), latest.get(col("tiltz_2")),
            latest.get(col("tiltx_3")), latest.get(col("tilty_3")), latest.get(col("tiltz_3")),
        ]
    })

with right:
    st.subheader("Computed Metrics")

    def show_metric(name, val, unit=""):
        if val is None or np.isnan(val):
            st.metric(name, "—")
        else:
            st.metric(name, f"{val:,.3f} {unit}")

    show_metric("Mean roll", latest["mean_roll_deg"], "deg")
    show_metric("Mean pitch", latest["mean_pitch_deg"], "deg")
    show_metric("Cant", latest["cant_mm"], "mm")
    show_metric("Left roll", latest["left_mean_roll_deg"], "deg")
    show_metric("Right roll", latest["right_mean_roll_deg"], "deg")
    show_metric("Left–Right diff", latest["left_right_diff_mm"], "mm")

    # -------- Threshold Indicator --------
    cant = latest["cant_mm"]
    limit = PERMITTED_CANT_MM
    warn = WARNING_RATIO * limit

    st.markdown(f"**Limit:** {limit} mm  |  **Warning:** {warn:.1f} mm")

    if np.isnan(cant):
        st.markdown("**Status:** ⚪ No cant value")
    else:
        if cant >= limit:
            color = "#D9534F"; text = "DANGER — Cant exceeds limit"
        elif cant >= warn:
            color = "#F0AD4E"; text = "WARNING — Approaching limit"
        else:
            color = "#5CB85C"; text = "SAFE — Within acceptable limits"

        st.markdown(
            f"""
            <div style="background:{color}; padding:12px; border-radius:6px; color:white;">
            <b>{text}</b><br>
            Cant = {cant:.2f} mm
            </div>
            """,
            unsafe_allow_html=True
        )

# -------- Recent Data Table --------
st.subheader("Recent Telemetry")
display_cols = [ts_col] if ts_col else []
display_cols += [
    col("tiltx_0"), col("tilty_0"), col("tiltz_0"),
    col("tiltx_1"), col("tilty_1"), col("tiltz_1"),
    col("tiltx_2"), col("tilty_2"), col("tiltz_2"),
    col("tiltx_3"), col("tilty_3"), col("tiltz_3"),
    "mean_roll_deg","mean_pitch_deg","cant_mm","left_right_diff_mm"
]
display_cols = [c for c in display_cols if c in df.columns]

st.dataframe(df[display_cols].tail(200).reset_index(drop=True))

st.caption("Charts removed as per requirement. Dashboard shows only essential values and threshold alerts.")
