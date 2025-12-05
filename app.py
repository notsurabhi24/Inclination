# app.py
# Streamlit dashboard (GPS removed).
# Reads sheet/GitHub CSV link, expects headers:
# Timestamp,TiltX_0,TiltY_0,TiltZ_0,TiltX_1,TiltY_1,TiltZ_1,TiltX_2,TiltY_2,TiltZ_2,TiltX_3,TiltY_3,TiltZ_3,Latitude,Longitude
# -- but Latitude/Longitude are ignored. Focus: show mean roll/pitch, cant_mm, left-right diff_mm, timestamp from sheet.

import streamlit as st
import pandas as pd
import numpy as np
import math, requests, io
from datetime import datetime
from typing import Tuple
from streamlit_autorefresh import st_autorefresh
from urllib.parse import urlparse, parse_qs

# ---------- CONFIG ----------
READ_REFRESH_MS = 2000       # UI poll interval (2s)
GAUGE_M = 1.676              # meters (Indian broad gauge)
PERMITTED_CANT_MM = 165.0
WARNING_RATIO = 0.8

st.set_page_config(layout="wide", page_title="Rail Tilt Monitor ")
st.title("Rail Tilt Monitor — Live (Tilt degrees, 4 sensors)")

# auto-refresh
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
    if "github.com" in u and "/blob/" in u:
        return u.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    if "raw.githubusercontent.com" in u or "output=csv" in u or "/export?format=csv" in u or "/pub?gid=" in u:
        return u
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
        raise RuntimeError("URL returned HTML (not CSV). If using Google Sheets, publish to web or make sharing 'anyone with link' and use export URL.")
    try:
        df = pd.read_csv(io.StringIO(text))
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(text), engine="python", on_bad_lines="skip")
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to parse CSV. Pandas error: {e}\nFirst 400 chars:\n{head}")

# ---------- UI: get data source ----------
st.sidebar.header("Data source")
src_url = st.sidebar.text_input("Paste Google Sheets edit/share URL or GitHub file URL (no file upload)", value="")
if not src_url:
    st.sidebar.info("Paste the sheet or GitHub file URL here. The app will attempt to convert it to a CSV export URL.")
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

# ---------- normalize columns ----------
cols_lower = {c.lower(): c for c in df.columns}

# expected tilt columns (case-insensitive)
required = ["tiltx_0","tiltx_1","tiltx_2","tiltx_3","tilty_0","tilty_1","tilty_2","tilty_3"]
missing = [r for r in required if r not in cols_lower]
if missing:
    st.error(f"Missing tilt columns (case-insensitive): {missing}. Found: {list(df.columns)}")
    st.stop()

def col(name): return cols_lower.get(name.lower())

# preserve and show timestamp from sheet exactly
ts_col = None
for c in df.columns:
    if "time" in c.lower() or "timestamp" in c.lower():
        ts_col = c
        break

# If timestamp column exists, do not change it — show it as provided (string)
if ts_col:
    # keep original string for display, but also parse for sorting if needed
    try:
        df["_ts_parsed"] = pd.to_datetime(df[ts_col], errors="coerce")
    except Exception:
        df["_ts_parsed"] = pd.NaT
else:
    df["_ts_parsed"] = pd.NaT

# ---------- compute metrics ----------
mean_rolls=[]; mean_pitch=[]; left_mean=[]; right_mean=[]
cant_mm_list=[]; left_right_diff_mm_list=[]

for _, row in df.iterrows():
    tx0 = safe_float(row[col("tiltx_0")])
    tx1 = safe_float(row[col("tiltx_1")])
    tx2 = safe_float(row[col("tiltx_2")])
    tx3 = safe_float(row[col("tiltx_3")])
    ty0 = safe_float(row[col("tilty_0")])
    ty1 = safe_float(row[col("tilty_1")])
    ty2 = safe_float(row[col("tilty_2")])
    ty3 = safe_float(row[col("tilty_3")])
    # mean roll/pitch (deg)
    rolls = np.array([tx0, tx1, tx2, tx3], dtype=float)
    pitches = np.array([ty0, ty1, ty2, ty3], dtype=float)
    mean_roll_deg = float(np.nanmean(rolls))
    mean_pitch_deg = float(np.nanmean(pitches))
    # left (0,3) vs right (1,2)
    left_mean_roll_deg = float(np.nanmean([tx0, tx3]))
    right_mean_roll_deg = float(np.nanmean([tx1, tx2]))
    # convert to mm across gauge
    mean_roll_rad = math.radians(mean_roll_deg)
    left_right_diff_rad = math.radians(left_mean_roll_deg - right_mean_roll_deg)
    cant_mm = math.tan(mean_roll_rad) * GAUGE_M * 1000.0 if not np.isnan(mean_roll_deg) else np.nan
    left_right_diff_mm = math.tan(left_right_diff_rad) * GAUGE_M * 1000.0 if not np.isnan(left_mean_roll_deg - right_mean_roll_deg) else np.nan

    mean_rolls.append(mean_roll_deg)
    mean_pitch.append(mean_pitch_deg)
    left_mean.append(left_mean_roll_deg)
    right_mean.append(right_mean_roll_deg)
    cant_mm_list.append(cant_mm)
    left_right_diff_mm_list.append(left_right_diff_mm)

# attach to df
df["mean_roll_deg"] = mean_rolls
df["mean_pitch_deg"] = mean_pitch
df["left_mean_roll_deg"] = left_mean
df["right_mean_roll_deg"] = right_mean
df["cant_mm"] = cant_mm_list
df["left_right_diff_mm"] = left_right_diff_mm_list

# ---------- latest row metrics ----------
latest = df.iloc[-1]

# Build timestamp display exactly from sheet if available, else fallback to index/time now
if ts_col:
    ts_display = str(latest[ts_col])
else:
    ts_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------- UI layout ----------
left_col, right_col = st.columns([2,1])

with left_col:
    st.subheader("Live telemetry (no GPS)")
    st.write("Timestamp (from sheet):", ts_display)
    st.markdown("**Sensor tilts (degrees)**")
    st.table({
        "Sensor": ["TiltX_0","TiltY_0","TiltZ_0","TiltX_1","TiltY_1","TiltZ_1","TiltX_2","TiltY_2","TiltZ_2","TiltX_3","TiltY_3","TiltZ_3"],
        "Value (latest)": [
            latest.get(col("tiltx_0"), ""),
            latest.get(col("tilty_0"), ""),
            latest.get(cols_lower.get("tiltz_0","tiltz_0"), ""),
            latest.get(col("tiltx_1"), ""),
            latest.get(col("tilty_1"), ""),
            latest.get(cols_lower.get("tiltz_1","tiltz_1"), ""),
            latest.get(col("tiltx_2"), ""),
            latest.get(col("tilty_2"), ""),
            latest.get(cols_lower.get("tiltz_2","tiltz_2"), ""),
            latest.get(col("tiltx_3"), ""),
            latest.get(col("tilty_3"), ""),
            latest.get(cols_lower.get("tiltz_3","tiltz_3"), ""),
        ]
    })

    st.subheader("Charts (recent)")
    # show last N rows
    N = 200
    # Removed TiltX and TiltY charts per request — only show cant/left-right chart
    try:
        chart_df3 = df[["cant_mm","left_right_diff_mm"]].tail(N).astype(float)
        st.line_chart(chart_df3)
    except Exception:
        st.info("Unable to plot cant/left-right chart (check computed values).")

with right_col:
    st.subheader("Computed metrics (latest)")
    # Always show numeric values (or '—' if NaN)
    def show_metric(name, val, unit=""):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            st.metric(label=name, value="—")
        else:
            if isinstance(val, float):
                st.metric(label=name, value=f"{val:,.3f} {unit}".strip())
            else:
                st.metric(label=name, value=f"{val} {unit}".strip())

    show_metric("Mean roll (deg)", latest.get("mean_roll_deg"), "deg")
    show_metric("Mean pitch (deg)", latest.get("mean_pitch_deg"), "deg")
    show_metric("Cant (mm)", latest.get("cant_mm"), "mm")
    show_metric("Left mean roll (deg)", latest.get("left_mean_roll_deg"), "deg")
    show_metric("Right mean roll (deg)", latest.get("right_mean_roll_deg"), "deg")
    show_metric("Left-right diff (mm)", latest.get("left_right_diff_mm"), "mm")

    # status color & threshold display (project-specific)
    cant_val = safe_float(latest.get("cant_mm"), np.nan)
    limit = PERMITTED_CANT_MM
    warn = WARNING_RATIO * PERMITTED_CANT_MM

    st.markdown("**Thresholds:**")
    st.markdown(f"- Permitted cant (limit): **{limit:.1f} mm**")
    st.markdown(f"- Warning threshold (0.8 × limit): **{warn:.1f} mm**")

    # Visual colored status box using safe HTML
    if np.isnan(cant_val):
        st.markdown("**Status:** ⚪ No cant value")
    else:
        if cant_val >= limit:
            color = "#D9534F"  # red
            status_text = "OVER LIMIT"
        elif cant_val >= warn:
            color = "#F0AD4E"  # yellow/orange
            status_text = "Approaching limit"
        else:
            color = "#5CB85C"  # green
            status_text = "OK"

        box_html = f"""
        <div style="background-color:{color};padding:12px;border-radius:6px">
            <strong style="color:white">Status: {status_text}</strong><br/>
            <span style="color:white">Cant = {cant_val:.2f} mm</span>
        </div>
        """
        st.markdown(box_html, unsafe_allow_html=True)

# ---------- table of recent rows ----------
st.subheader("Recent telemetry (tail)")
# show original Timestamp column exactly as present plus computed columns
display_cols = []
if ts_col:
    display_cols.append(ts_col)
display_cols += [col("tiltx_0"), col("tilty_0"), col("tiltz_0"),
                 col("tiltx_1"), col("tilty_1"), col("tiltz_1"),
                 col("tiltx_2"), col("tilty_2"), col("tiltz_2"),
                 col("tiltx_3"), col("tilty_3"), col("tiltz_3"),
                 "mean_roll_deg","mean_pitch_deg","cant_mm","left_right_diff_mm"]
# filter only those actually in df (computed columns exist)
display_cols = [c for c in display_cols if c in df.columns or c in ["mean_roll_deg","mean_pitch_deg","cant_mm","left_right_diff_mm"]]
st.dataframe(df[display_cols].tail(200).reset_index(drop=True))

# ---------- download augmented CSV ----------
st.subheader("Download augmented CSV (includes computed columns)")
buf = io.StringIO()
df.to_csv(buf, index=False)
st.download_button("Download augmented CSV", data=buf.getvalue(), file_name="augmented_telemetry_no_gps.csv", mime="text/csv")

st.caption("Notes: GPS/map removed. Timestamp displayed exactly from your sheet. If the metrics show — then values are missing or non-numeric in the sheet; check that TiltX/TiltY columns contain numeric degree values.")
