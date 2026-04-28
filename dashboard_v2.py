# =============================================================================
# NEUROWATCH — Multi-Patient EEG Seizure Monitor
# Apple HIG colour system — Dark Mode
# Seizure state: blood-red (#C0392B / #FF0000) with CSS blink animation
# =============================================================================

import os
import json
import time
import math
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
_JSON_DIR        = os.path.join(BASE_DIR, "json")
PATIENTS_JSON    = os.path.join(_JSON_DIR, "neurowatch_patients.json")
METRICS_JSON     = os.path.join(_JSON_DIR, "neurowatch_metrics.json")
STATUS_JSON      = os.path.join(_JSON_DIR, "neurowatch_status.json")
REFRESH_INTERVAL = 5

CLASSES = ["Normal", "Pre-Seizure", "Seizure"]

PATIENT_REGISTRY = [
    {"id": "P001", "live": True},
    {"id": "P002", "live": False},
    {"id": "P003", "live": False},
]

# ── Apple HIG Dark Mode palette ──────────────────────────────────────────────
APPLE_BLUE   = "#0A84FF"   # System Blue   — primary actions
APPLE_GREEN  = "#32D74B"   # System Green  — Normal / success
APPLE_AMBER  = "#FF9F0A"   # System Orange — Pre-Seizure / warning
APPLE_INDIGO = "#5E5CE6"   # System Indigo — secondary accent
APPLE_RED    = "#FF453A"   # System Red    — standard error

# Seizure-specific: blood red
BLOOD_RED    = "#C0392B"   # deep blood red (resting state of blink)
BLOOD_HOT    = "#FF0000"   # pure red (highlight state of blink)

# Background layers — Apple Dark Mode gray scale
APPLE_BG     = "#000000"   # True black  (OLED)
APPLE_BG2    = "#1C1C1E"   # System Gray 6 Dark — card bg
APPLE_BG3    = "#2C2C2E"   # System Gray 5 Dark — elevated surface
APPLE_BORDER = "#3A3A3C"   # System Gray 4 Dark — separator
APPLE_GRAY   = "#8E8E93"   # System Gray — disabled / muted

# Label semantic colours
APPLE_LABEL  = "#FFFFFF"
APPLE_LABEL2 = "rgba(235,235,245,0.6)"
APPLE_LABEL3 = "rgba(235,235,245,0.3)"

# Glass
GLASS_BG     = "rgba(44,44,46,0.72)"
GLASS_BORDER = "rgba(255,255,255,0.10)"
GLASS_HI     = "rgba(255,255,255,0.06)"

CLASS_ICONS  = {"Normal": "🟢", "Pre-Seizure": "🟡", "Seizure": "🔴"}
PRIORITY     = {"Seizure": 2, "Pre-Seizure": 1, "Normal": 0}

# ── Botswana geography ────────────────────────────────────────────────────────
USER_LAT, USER_LNG = -22.5606, 27.1325   # user is in Palapye

PATIENT_GPS_FALLBACK = {
    "P001": {"gps_lat": -22.5606, "gps_lng": 27.1325, "city": "Palapye"},
    "P002": {"gps_lat": -24.6282, "gps_lng": 25.9231, "city": "Gaborone"},
    "P003": {"gps_lat": -22.3908, "gps_lng": 26.7106, "city": "Serowe"},
    "P004": {"gps_lat": -19.9833, "gps_lng": 23.4167, "city": "Maun"},
    "P005": {"gps_lat": -24.9833, "gps_lng": 25.3500, "city": "Kanye"},
    "P006": {"gps_lat": -24.4114, "gps_lng": 26.1500, "city": "Mochudi"},
    "P007": {"gps_lat": -25.0333, "gps_lng": 25.5500, "city": "Goodhope"},
}

BOTSWANA_HOSPITALS = [
    {"name": "Princess Marina Hospital",      "lat": -24.6481, "lng": 25.9117, "type": "General",     "city": "Gaborone"},
    {"name": "Bokamoso Private Hospital",     "lat": -24.6181, "lng": 25.9711, "type": "Private",     "city": "Gaborone"},
    {"name": "Athlone Hospital",              "lat": -24.7500, "lng": 25.9100, "type": "Psychiatric", "city": "Gaborone"},
    {"name": "Gaborone Private Hospital",     "lat": -24.6550, "lng": 25.9100, "type": "Private",     "city": "Gaborone"},
    {"name": "Palapye Primary Hospital",      "lat": -22.5650, "lng": 27.1380, "type": "Primary",     "city": "Palapye"},
    {"name": "Palapye Clinic",                "lat": -22.5750, "lng": 27.1420, "type": "Clinic",      "city": "Palapye"},
    {"name": "Serowe Primary Hospital",       "lat": -22.3940, "lng": 26.7180, "type": "Primary",     "city": "Serowe"},
    {"name": "Maun General Hospital",         "lat": -19.9893, "lng": 23.4268, "type": "General",     "city": "Maun"},
    {"name": "Scottish Livingstone Hospital", "lat": -24.9833, "lng": 25.3500, "type": "Mission",     "city": "Kanye"},
    {"name": "Kanye Primary Hospital",        "lat": -24.9900, "lng": 25.3620, "type": "Primary",     "city": "Kanye"},
    {"name": "Mochudi Primary Hospital",      "lat": -24.4114, "lng": 26.1500, "type": "Primary",     "city": "Mochudi"},
    {"name": "Deborah Retief Memorial",       "lat": -24.3989, "lng": 26.1317, "type": "Mission",     "city": "Mochudi"},
    {"name": "Goodhope Primary Hospital",     "lat": -25.0333, "lng": 25.5500, "type": "Primary",     "city": "Goodhope"},
    {"name": "Nyangabgwe Referral Hospital",  "lat": -21.1550, "lng": 27.5100, "type": "Referral",    "city": "Francistown"},
    {"name": "Lobatse Mental Hospital",       "lat": -25.2167, "lng": 25.6833, "type": "Psychiatric", "city": "Lobatse"},
]


def state_color(state):
    return {
        "Normal":      APPLE_GREEN,
        "Pre-Seizure": APPLE_AMBER,
        "Seizure":     BLOOD_RED,
    }.get(state, APPLE_GRAY)


# =============================================================================
# PAGE CONFIG  ← absolute first Streamlit call
# =============================================================================
st.set_page_config(
    page_title="NeuroWatch — Multi-Patient Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS — Apple HIG Dark + blood-red blink
# =============================================================================
st.markdown(f"""
<style>
:root {{
    --blue:      {APPLE_BLUE};
    --green:     {APPLE_GREEN};
    --amber:     {APPLE_AMBER};
    --indigo:    {APPLE_INDIGO};
    --red:       {APPLE_RED};
    --blood:     {BLOOD_RED};
    --blood-hot: {BLOOD_HOT};
    --bg:        {APPLE_BG};
    --bg2:       {APPLE_BG2};
    --bg3:       {APPLE_BG3};
    --border:    {APPLE_BORDER};
    --gray:      {APPLE_GRAY};
    --label:     {APPLE_LABEL};
    --label2:    {APPLE_LABEL2};
    --label3:    {APPLE_LABEL3};
    --glass:     {GLASS_BG};
    --gb:        {GLASS_BORDER};
    --ghi:       {GLASS_HI};
}}

html, body, [data-testid="stAppViewContainer"],
.stApp, .block-container, .main {{
    font-family: -apple-system, "SF Pro Display", "SF Pro Text",
                 "Helvetica Neue", sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--label) !important;
    min-height: 100vh;
}}

section[data-testid="stSidebar"] > div:first-child {{
    background: var(--bg2) !important;
    border-right: 0.5px solid var(--border) !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: var(--bg2) !important;
    border-radius: 10px !important;
    padding: 3px !important;
    border: 0.5px solid var(--border) !important;
    gap: 2px !important;
}}
.stTabs [data-baseweb="tab"] {{
    color: var(--gray) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border-radius: 8px !important;
    padding: 5px 14px !important;
    transition: all .2s ease !important;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: var(--bg3) !important;
    color: var(--label) !important;
    font-weight: 600 !important;
}}

/* Metric card — glass */
.metric-card {{
    background: var(--glass);
    border-radius: 16px;
    padding: 18px 16px;
    border: 0.5px solid var(--gb);
    text-align: center;
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    box-shadow: 0 2px 12px rgba(0,0,0,.4),
                inset 0 1px 0 var(--ghi);
    transition: transform .2s ease, box-shadow .2s ease;
}}
.metric-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,.5), inset 0 1px 0 var(--ghi);
}}
.metric-label {{
    color: var(--label2);
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 4px;
}}
.metric-value {{
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -.5px;
    line-height: 1;
    margin: 6px 0 4px;
}}
.metric-sub {{
    color: var(--label3);
    font-size: 11px;
}}

/* Glass panel */
.glass-panel {{
    background: var(--glass);
    border: 0.5px solid var(--gb);
    border-radius: 14px;
    padding: 16px 18px;
    backdrop-filter: blur(20px) saturate(160%);
    -webkit-backdrop-filter: blur(20px) saturate(160%);
    box-shadow: 0 2px 8px rgba(0,0,0,.35), inset 0 1px 0 var(--ghi);
    margin-bottom: 12px;
}}

/* Eyebrow label */
.section-label {{
    font-size: 11px;
    font-weight: 600;
    color: var(--gray);
    letter-spacing: .06em;
    text-transform: uppercase;
    margin: 16px 0 8px;
}}

/* Sidebar */
.sidebar-title {{
    font-size: 17px;
    font-weight: 700;
    color: var(--blue);
    letter-spacing: -.3px;
    padding-bottom: 10px;
    border-bottom: 0.5px solid var(--border);
    margin-bottom: 12px;
}}
.uptime-box {{
    background: var(--bg3);
    border: 0.5px solid var(--border);
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 12px;
    color: var(--label2);
    margin-bottom: 10px;
}}

/* Live badge */
.live-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(10,132,255,.15);
    border: 0.5px solid rgba(10,132,255,.35);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 11px;
    font-weight: 500;
    color: var(--blue);
    margin-bottom: 8px;
}}
.live-dot {{
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--red);
    animation: pulse-dot 1.4s ease-in-out infinite;
    display: inline-block;
    flex-shrink: 0;
}}
@keyframes pulse-dot {{
    0%,100%{{ opacity:1; transform:scale(1);   }}
    50%     {{ opacity:.3; transform:scale(.7); }}
}}

/* Patient badges */
.badge-live {{
    background: rgba(50,215,75,.15);
    border: 0.5px solid rgba(50,215,75,.4);
    color: var(--green);
    font-size: 9px; font-weight: 700;
    padding: 1px 7px;
    border-radius: 20px;
    letter-spacing: .06em;
    margin-left: 5px;
    text-transform: uppercase;
}}
.badge-demo {{
    background: rgba(142,142,147,.12);
    border: 0.5px solid rgba(142,142,147,.25);
    color: var(--gray);
    font-size: 9px; font-weight: 600;
    padding: 1px 7px;
    border-radius: 20px;
    letter-spacing: .06em;
    margin-left: 5px;
    text-transform: uppercase;
}}
.state-timer {{
    font-size: 10px;
    color: var(--label3);
    margin-top: 3px;
    font-variant-numeric: tabular-nums;
}}

/* ══════════════════════════════════════════════════════════
   BLOOD-RED BLINK — seizure-only animation
   Pulses between deep blood-red and hot red with outer glow
   ══════════════════════════════════════════════════════════ */
@keyframes seizure-blink {{
    0%,100% {{
        background:   rgba(139,0,0,.25);
        border-color: rgba(192,57,43,.7);
        box-shadow:   0 0 0 0 rgba(255,0,0,0);
    }}
    50% {{
        background:   rgba(255,0,0,.18);
        border-color: rgba(255,0,0,.9);
        box-shadow:   0 0 20px 6px rgba(255,0,0,.30);
    }}
}}

/* Full-width seizure alert banner */
.alert-seizure {{
    background: rgba(139,0,0,.25);
    border: 1px solid rgba(192,57,43,.7);
    border-left: 4px solid var(--blood);
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 10px;
    color: #fff;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: -.1px;
    animation: seizure-blink 1s ease-in-out infinite;
}}

/* Pre-seizure banner — amber, static */
.alert-pre {{
    background: rgba(255,159,10,.12);
    border: 0.5px solid rgba(255,159,10,.4);
    border-left: 3px solid var(--amber);
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 10px;
    color: #fff;
    font-weight: 600;
    font-size: 13px;
    backdrop-filter: blur(10px);
}}

/* Notification items */
.notif-seizure {{
    background: rgba(139,0,0,.20);
    border: 0.5px solid rgba(192,57,43,.5);
    border-radius: 10px;
    padding: 9px 12px;
    margin-bottom: 6px;
    font-size: 12px;
    color: var(--label);
    animation: seizure-blink 1s ease-in-out infinite;
}}
.notif-pre {{
    background: rgba(255,159,10,.10);
    border: 0.5px solid rgba(255,159,10,.3);
    border-radius: 10px;
    padding: 9px 12px;
    margin-bottom: 6px;
    font-size: 12px;
    color: var(--label);
}}
.notif-normal {{
    background: rgba(50,215,75,.08);
    border: 0.5px solid rgba(50,215,75,.2);
    border-radius: 10px;
    padding: 9px 12px;
    margin-bottom: 6px;
    font-size: 12px;
    color: var(--label);
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    background: var(--bg2) !important;
    border-radius: 12px !important;
    border: 0.5px solid var(--border) !important;
}}

/* Streamlit overrides */
#MainMenu{{visibility:hidden;}}
footer{{visibility:hidden;}}
header{{visibility:hidden;}}

.stButton > button {{
    background: rgba(10,132,255,.15) !important;
    border: 0.5px solid rgba(10,132,255,.4) !important;
    color: var(--blue) !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    transition: all .18s ease !important;
}}
.stButton > button:hover {{
    background: rgba(10,132,255,.28) !important;
    transform: scale(1.02) !important;
}}
[data-testid="stDownloadButton"] > button {{
    background: rgba(50,215,75,.12) !important;
    border: 0.5px solid rgba(50,215,75,.35) !important;
    color: var(--green) !important;
    border-radius: 8px !important;
    font-size: 12px !important;
}}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# JSON READERS
# =============================================================================
def read_json(path, default=None):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

@st.cache_data(ttl=REFRESH_INTERVAL)
def get_all_patients():
    data = read_json(PATIENTS_JSON, {})
    for pid, pdata in data.items():
        if not pdata.get("gps_lat") and pid in PATIENT_GPS_FALLBACK:
            pdata.update(PATIENT_GPS_FALLBACK[pid])
    return data

@st.cache_data(ttl=REFRESH_INTERVAL)
def get_metrics():
    return read_json(METRICS_JSON, {})

@st.cache_data(ttl=REFRESH_INTERVAL)
def get_status():
    return read_json(STATUS_JSON, {"phase": "WAITING", "message": "Waiting for Pi..."})

@st.cache_data(ttl=REFRESH_INTERVAL)
def get_patient_history(pid):
    path = os.path.join(_JSON_DIR, f"neurowatch_history_{pid}.json")
    data = read_json(path, [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

_live_pids = {p["id"] for p in PATIENT_REGISTRY if p.get("live")}
def is_live(pid): return pid in _live_pids


def get_patient_avatar_html(pid, patient_name, size=42):
    """Return HTML for a circular patient avatar (photo or coloured initials)."""
    photos_dir = os.path.join(BASE_DIR, "patient_photos")
    for ext in ("jpg", "jpeg", "png", "webp"):
        fpath = os.path.join(photos_dir, f"{pid}.{ext}")
        if os.path.exists(fpath):
            with open(fpath, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            return (f'<img src="data:{mime};base64,{b64}" style="width:{size}px;height:{size}px;'
                    f'border-radius:50%;object-fit:cover;flex-shrink:0;'
                    f'border:2px solid rgba(255,255,255,0.18);">')
    # Initials fallback
    palette = ["#0A84FF","#32D74B","#FF9F0A","#5E5CE6","#FF453A","#30D5C8","#BF5AF2"]
    bg      = palette[hash(pid) % len(palette)]
    initials = "".join(w[0].upper() for w in patient_name.split()[:2]) or "?"
    fs = size // 3
    return (f'<div style="width:{size}px;height:{size}px;border-radius:50%;background:{bg};'
            f'display:flex;align-items:center;justify-content:center;font-size:{fs}px;'
            f'font-weight:700;color:#fff;flex-shrink:0;border:2px solid rgba(255,255,255,0.18)">'
            f'{initials}</div>')


# =============================================================================
# HELPERS
# =============================================================================
def format_uptime(s):
    return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{int(s%60):02d}"

# =============================================================================
# MAP HELPERS
# =============================================================================
def make_osm_embed(lat, lng, label="Patient Location", zoom=17):
    google_url = f"https://maps.google.com/?q={lat},{lng}"
    return f"""<!DOCTYPE html><html><head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:#000;font-family:-apple-system,sans-serif;}}
  #map{{width:100%;height:290px;border-radius:10px;}}
  .foot{{background:{APPLE_BG2};padding:8px 14px;display:flex;
         justify-content:space-between;align-items:center;
         border-top:0.5px solid {APPLE_BORDER};}}
  .coord{{font-family:monospace;font-size:11px;color:{APPLE_GRAY};}}
  .btn{{background:rgba(10,132,255,.12);border:0.5px solid rgba(10,132,255,.4);
        color:{APPLE_BLUE};padding:5px 14px;border-radius:8px;font-size:12px;
        cursor:pointer;text-decoration:none;}}
  .btn:hover{{background:rgba(10,132,255,.22);}}
</style></head><body>
<div id="map"></div>
<div class="foot">
  <span class="coord"> {lat:.4f}, {lng:.4f} — {label}</span>
  <a class="btn" href="{google_url}" target="_blank">Open in Maps ↗</a>
</div>
<script>
  var map=L.map('map',{{zoomControl:true,attributionControl:false}}).setView([{lat},{lng}],{zoom});
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
    {{maxZoom:19,subdomains:'abcd'}}).addTo(map);
  var icon=L.divIcon({{
    html:'<div style="width:13px;height:13px;background:{APPLE_BLUE};border:2px solid #fff;border-radius:50%;box-shadow:0 0 8px {APPLE_BLUE};"></div>',
    iconSize:[13,13],iconAnchor:[6,6],className:''
  }});
  L.marker([{lat},{lng}],{{icon:icon}}).addTo(map).bindPopup('<b>{label}</b>');
  L.circle([{lat},{lng}],{{color:'{APPLE_BLUE}',fillColor:'{APPLE_BLUE}',fillOpacity:0.07,radius:15,weight:1}}).addTo(map);
</script></body></html>"""


def make_all_patients_map(all_patients, sorted_pids):
    cmap  = {"Normal": APPLE_GREEN, "Pre-Seizure": APPLE_AMBER, "Seizure": BLOOD_RED}
    valid = []
    mjs   = ""

    # Patient markers + distance lines from user
    for pid in sorted_pids:
        p   = all_patients.get(pid, {})
        lat = p.get("gps_lat", "")
        lng = p.get("gps_lng", "")
        if not lat or not lng:
            continue
        try:
            lat, lng = float(lat), float(lng)
        except (ValueError, TypeError):
            continue
        state = p.get("state", "Normal")
        col   = cmap.get(state, APPLE_GREEN)
        name  = p.get("patient_name", "—").replace("'", "\\'")
        ward  = p.get("ward", "—")
        bed   = p.get("bed", "—")
        conf  = int(p.get("confidence", 0) * 100)
        live  = "LIVE" if is_live(pid) else "DEMO"
        city  = p.get("city", "")
        dist  = haversine(USER_LAT, USER_LNG, lat, lng)
        valid.append((lat, lng))

        # Pulsing ring for seizure patients
        pulse = f"L.circle([{lat},{lng}],{{color:'{col}',fillColor:'{col}',fillOpacity:0.12,radius:8000,weight:1.5}}).addTo(map);" if state == "Seizure" else ""

        mjs += f"""
        L.marker([{lat},{lng}],{{icon:L.divIcon({{
          html:'<div style="width:16px;height:16px;background:{col};border:2.5px solid #fff;border-radius:50%;box-shadow:0 0 14px {col};"></div>',
          iconSize:[16,16],iconAnchor:[8,8],className:''
        }})}}).addTo(map)
        .bindPopup('<div style="font-family:-apple-system,sans-serif;min-width:160px">'
          +'<b style="font-size:13px">{name}</b> <span style="font-size:10px;color:#888">{live}</span><br>'
          +'<span style="color:#555;font-size:11px">{ward} · {bed}</span><br>'
          +'<b style="color:{col};font-size:12px">{state} ({conf}%)</b><br>'
          +'<span style="font-size:10px;color:#888">📍 {city}</span><br>'
          +'<span style="font-size:10px;color:#0A84FF">📏 {dist:.1f} km from Palapye</span>'
          +'</div>');
        L.polyline([[{USER_LAT},{USER_LNG}],[{lat},{lng}]],
          {{color:'{col}',weight:1.2,opacity:0.35,dashArray:'6,6'}}).addTo(map);
        {pulse}
        """

    if not valid:
        return None

    # Hospital markers — offset slightly if they land within 22 km of a patient
    # so the hospital diamond doesn't sit directly on a patient dot
    _OFFSET_ANGLES = [50,130,230,310,20,160,200,340,80,100,260,280,40,140,220]
    hosp_js = ""
    for hi, h in enumerate(BOTSWANA_HOSPITALS):
        hlat, hlng = h["lat"], h["lng"]
        for plat, plng in valid:
            if haversine(hlat, hlng, plat, plng) < 22:
                ang  = math.radians(_OFFSET_ANGLES[hi % len(_OFFSET_ANGLES)])
                hlat += 0.11 * math.sin(ang)
                hlng += 0.14 * math.cos(ang)
                break
        hname = h["name"].replace("'", "\\'")
        htype = h["type"]
        hdist = haversine(USER_LAT, USER_LNG, hlat, hlng)
        hosp_js += f"""
        L.marker([{hlat},{hlng}],{{icon:L.divIcon({{
          html:'<div style="width:10px;height:10px;background:#fff;border:2px solid #FF453A;border-radius:2px;transform:rotate(45deg)"></div>',
          iconSize:[10,10],iconAnchor:[5,5],className:''
        }})}}).addTo(map)
        .bindPopup('<div style="font-family:-apple-system,sans-serif">'
          +'<b style="color:#FF453A">🏥 {hname}</b><br>'
          +'<span style="color:#888;font-size:11px">{htype} Hospital</span><br>'
          +'<span style="font-size:10px;color:#0A84FF">📏 {hdist:.1f} km from Palapye</span>'
          +'</div>');
        """

    # User (you) marker
    user_js = f"""
    L.marker([{USER_LAT},{USER_LNG}],{{icon:L.divIcon({{
      html:'<div style="width:18px;height:18px;background:#5E5CE6;border:3px solid #fff;border-radius:50%;box-shadow:0 0 16px #5E5CE6;display:flex;align-items:center;justify-content:center;font-size:9px;color:#fff;font-weight:700">YOU</div>',
      iconSize:[18,18],iconAnchor:[9,9],className:''
    }})}}).addTo(map)
    .bindPopup('<b style="color:#5E5CE6">📍 Your Location</b><br><span style="font-size:11px">Palapye, Botswana</span>');
    L.circle([{USER_LAT},{USER_LNG}],{{color:'#5E5CE6',fillColor:'#5E5CE6',fillOpacity:0.06,radius:5000,weight:1}}).addTo(map);
    """

    # Map center = midpoint of all valid points
    all_lats = [v[0] for v in valid] + [USER_LAT]
    all_lngs = [v[1] for v in valid] + [USER_LNG]
    clat = sum(all_lats) / len(all_lats)
    clng = sum(all_lngs) / len(all_lngs)

    return f"""<!DOCTYPE html><html><head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}body{{background:#000;}}
  #map{{width:100%;height:440px;border-radius:10px;}}
  .legend{{position:absolute;bottom:20px;right:10px;z-index:999;
    background:rgba(28,28,30,.92);border:0.5px solid {APPLE_BORDER};
    border-radius:10px;padding:10px 14px;font-size:11px;color:{APPLE_GRAY};
    font-family:-apple-system,monospace;line-height:1.8;}}
  .li{{display:flex;align-items:center;gap:8px;margin-bottom:2px;}}
  .dot{{width:9px;height:9px;border-radius:50%;flex-shrink:0;}}
  .sq{{width:9px;height:9px;background:#fff;border:2px solid #FF453A;
       transform:rotate(45deg);flex-shrink:0;border-radius:1px;}}
</style></head><body>
<div id="map"></div>
<div class="legend">
  <div class="li"><div class="dot" style="background:{APPLE_GREEN}"></div>Normal</div>
  <div class="li"><div class="dot" style="background:{APPLE_AMBER}"></div>Pre-Seizure</div>
  <div class="li"><div class="dot" style="background:{BLOOD_RED}"></div>Seizure</div>
  <div class="li"><div class="dot" style="background:#5E5CE6"></div>You (Palapye)</div>
  <div class="li"><div class="sq"></div>Hospital / Clinic</div>
</div>
<script>
  var map=L.map('map',{{zoomControl:true,attributionControl:false}}).setView([{clat},{clng}],7);
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
    {{maxZoom:19,subdomains:'abcd'}}).addTo(map);
  {mjs}
  {hosp_js}
  {user_js}
</script></body></html>"""

# =============================================================================
# DISTANCE + BRAIN HELPERS
# =============================================================================
def haversine(lat1, lng1, lat2, lng2):
    """Return great-circle distance in km."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_hospitals(lat, lng, n=3):
    """Return n closest hospitals sorted by distance from (lat, lng)."""
    scored = [(haversine(lat, lng, h["lat"], h["lng"]), h)
              for h in BOTSWANA_HOSPITALS]
    scored.sort(key=lambda x: x[0])
    return [(d, h) for d, h in scored[:n]]


def make_brain_mri(state, band_powers, patient_name="Patient"):
    """
    Realistic MRI-style axial brain slice (frontal=top) with EEG activity overlay.
    Renders skull, gray/white matter, sulci, ventricles via numpy then uses
    go.Image so the result looks like a clinical scan with functional colour overlay.
    """
    N  = 300
    xi = np.linspace(-1.1,  1.1, N)
    yi = np.linspace( 1.1, -1.1, N)   # top→bottom: row 0 = frontal (north)
    gx, gy = np.meshgrid(xi, yi)
    r  = np.sqrt(gx**2 + gy**2)
    th = np.arctan2(gy, gx)

    # ── Anatomy (grayscale 0–1) ───────────────────────────────────────────────
    anat = np.full((N, N), 0.04, dtype=np.float32)

    anat[(r >= 0.82) & (r < 0.97)] = np.clip(
        0.88 - 0.10 * (r[(r >= 0.82) & (r < 0.97)] - 0.82) / 0.15, 0.70, 0.92)
    anat[(r >= 0.76) & (r < 0.82)] = 0.14          # thin CSF / meninges
    anat[(r >= 0.33) & (r < 0.76)] = 0.46          # cortical gray matter
    anat[r < 0.33]                  = 0.70          # white matter

    for k in range(16):                             # gyri + sulci pattern
        angle  = k * 2.0 * np.pi / 16
        dth    = np.abs(((th - angle + np.pi) % (2 * np.pi)) - np.pi)
        anat[(dth < 0.11) & (r >= 0.42) & (r < 0.76)] = np.minimum(
            anat[(dth < 0.11) & (r >= 0.42) & (r < 0.76)] + 0.17, 1.0)
        anat[(dth < 0.034) & (r >= 0.52) & (r < 0.74)] = 0.09

    lv_l = ((gx + 0.14)**2 / 0.043**2 + (gy - 0.07)**2 / 0.12**2) < 1
    lv_r = ((gx - 0.14)**2 / 0.043**2 + (gy - 0.07)**2 / 0.12**2) < 1
    v3   = (np.abs(gx) < 0.022) & (np.abs(gy - 0.02) < 0.08) & (r < 0.28)
    anat[lv_l | lv_r | v3] = 0.07                  # ventricles (dark CSF)

    cc  = (np.abs(gx) < 0.22) & (np.abs(gy - 0.05) < 0.034) & (r < 0.43)
    anat[cc] = 0.86                                 # corpus callosum

    th_l = ((gx + 0.09)**2 / 0.055**2 + (gy + 0.07)**2 / 0.07**2) < 1
    th_r = ((gx - 0.09)**2 / 0.055**2 + (gy + 0.07)**2 / 0.07**2) < 1
    anat[th_l | th_r] = np.minimum(anat[th_l | th_r] + 0.10, 1.0)

    anat = np.clip(anat, 0, 1)

    # ── EEG Activity (IDW interpolation on same grid) ─────────────────────────
    ELEC = {
        "Fp1": (-0.309,  0.951), "Fp2": ( 0.309,  0.951),
        "F7":  (-0.809,  0.588), "F3":  (-0.545,  0.588),
        "Fz":  ( 0.000,  0.672), "F4":  ( 0.545,  0.588), "F8":  ( 0.809,  0.588),
        "T7":  (-0.990,  0.000), "C3":  (-0.545,  0.000),
        "Cz":  ( 0.000,  0.000), "C4":  ( 0.545,  0.000), "T8":  ( 0.990,  0.000),
        "P7":  (-0.809, -0.588), "P3":  (-0.545, -0.588),
        "Pz":  ( 0.000, -0.672), "P4":  ( 0.545, -0.588), "P8":  ( 0.809, -0.588),
        "O1":  (-0.309, -0.951), "Oz":  ( 0.000, -0.951), "O2":  ( 0.309, -0.951),
    }
    ELEC_BAND = {
        "Fp1":"beta","Fp2":"beta","F7":"theta","F3":"beta","Fz":"beta",
        "F4":"beta","F8":"theta","T7":"delta","T8":"delta","C3":"alpha",
        "Cz":"alpha","C4":"alpha","P7":"theta","P3":"alpha","Pz":"alpha",
        "P4":"alpha","P8":"theta","O1":"alpha","Oz":"alpha","O2":"alpha",
    }
    HOTSPOT = {
        "Seizure":     {"T7":5.0,"T8":5.0,"F7":3.5,"F8":3.5,"Fp1":3.0,"Fp2":3.0,"C3":2.2,"C4":2.2},
        "Pre-Seizure": {"Fp1":3.0,"Fp2":3.0,"F3":2.5,"F4":2.5,"Fz":2.2,"F7":1.8,"F8":1.8},
    }

    bp   = {b: float(band_powers.get(b, 0.25)) for b in ("delta","theta","alpha","beta")}
    tot  = sum(bp.values()) or 1.0
    bp   = {k: v/tot for k, v in bp.items()}
    names = list(ELEC.keys())
    ex   = np.array([ELEC[e][0] for e in names])
    ey   = np.array([ELEC[e][1] for e in names])
    ev   = np.array([bp[ELEC_BAND[e]] for e in names], dtype=float)
    for i, nm in enumerate(names):
        ev[i] = min(ev[i] * HOTSPOT.get(state, {}).get(nm, 1.0), 1.0)
    ev_n = (ev - ev.min()) / (ev.max() - ev.min() + 1e-9)

    act = np.zeros((N, N)); wgt = np.zeros((N, N))
    for i in range(len(names)):
        d2  = np.maximum((gx - ex[i])**2 + (gy - ey[i])**2, 1e-4)
        w   = 1.0 / d2
        act += ev_n[i] * w; wgt += w
    act = np.clip(act / wgt, 0, 1)
    act[r > 0.76] = 0.0                            # restrict to cortex

    # ── Composite RGB image ───────────────────────────────────────────────────
    ACT_RGB = {
        "Seizure":     np.array([1.0, 0.04, 0.0]),
        "Pre-Seizure": np.array([1.0, 0.55, 0.0]),
        "Normal":      np.array([0.0, 0.48, 1.0]),
    }.get(state, np.array([0.5, 0.5, 0.5]))

    img   = np.stack([anat, anat, anat], axis=-1).astype(float)
    alpha = np.clip((act - 0.30) / 0.70, 0, 1) ** 0.65 * 0.80
    for ch in range(3):
        img[:, :, ch] = img[:, :, ch] * (1 - alpha) + ACT_RGB[ch] * alpha
    img[r >= 1.00] = 0.0
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    # ── Plotly figure (go.Image + scatter overlay) ────────────────────────────
    def n2p(xn, yn):
        return (xn + 1.1)/2.2*(N-1), (1.1 - yn)/2.2*(N-1)

    sc  = {"Seizure": BLOOD_RED, "Pre-Seizure": APPLE_AMBER,
           "Normal": APPLE_GREEN}.get(state, APPLE_GRAY)
    lbl = {"Seizure": "⚡ SEIZURE DETECTED",
           "Pre-Seizure": "⚠ PRE-SEIZURE ACTIVITY",
           "Normal": "● Normal Brain Activity"}.get(state, state)
    CS  = {
        "Seizure":     [[0,"#220000"],[0.5,"#cc0000"],[1,"#ff9900"]],
        "Pre-Seizure": [[0,"#221100"],[0.5,"#cc7700"],[1,"#fff176"]],
        "Normal":      [[0,"#001133"],[0.5,"#0066cc"],[1,"#99eeff"]],
    }.get(state, [[0,"#333"],[1,"#fff"]])

    fig = go.Figure()
    fig.add_trace(go.Image(z=img_u8, hoverinfo="skip"))

    epx = [(ELEC[e][0]+1.1)/2.2*(N-1) for e in names]
    epy = [(1.1-ELEC[e][1])/2.2*(N-1) for e in names]
    fig.add_trace(go.Scatter(
        x=epx, y=epy, mode="markers+text",
        marker=dict(color=ev_n.tolist(), colorscale=CS, cmin=0, cmax=1,
                    size=7, line=dict(color="rgba(255,255,255,0.85)", width=1.0)),
        text=names, textfont=dict(size=6, color="rgba(255,255,255,0.52)"),
        textposition="bottom center", showlegend=False,
        hovertemplate="<b>%{text}</b><br>Activity: %{marker.color:.2f}<extra></extra>",
    ))

    for rlbl, xn, yn in [("FRONTAL", 0.0, 0.71), ("L-TEMP", -0.68, 0.05),
                          ("R-TEMP", 0.68, 0.05), ("PARIETAL", 0.0, -0.47),
                          ("OCCIPITAL", 0.0, -0.71)]:
        px, py = n2p(xn, yn)
        fig.add_annotation(x=px, y=py, text=rlbl,
                           font=dict(size=7, color="rgba(255,255,255,0.24)"),
                           showarrow=False)

    if state == "Seizure":
        for sgn, ename in ((-1, "T7"), (1, "T8")):
            px, py = n2p(*ELEC[ename])
            fig.add_annotation(x=px, y=py, text="⚡ Focus",
                               font=dict(size=8, color=BLOOD_RED), showarrow=True,
                               arrowhead=2, arrowcolor=BLOOD_RED, ax=sgn*32, ay=-26)
    elif state == "Pre-Seizure":
        px, py = n2p(0.0, 0.82)
        fig.add_annotation(x=px, y=py, text="⚠ Elevated",
                           font=dict(size=8, color=APPLE_AMBER), showarrow=True,
                           arrowhead=2, arrowcolor=APPLE_AMBER, ax=0, ay=28)

    fig.update_layout(
        paper_bgcolor="#000000", plot_bgcolor="#000000",
        margin=dict(l=5, r=5, t=52, b=5), height=390,
        title=dict(
            text=(f"<b>EEG-fMRI Overlay</b>  ·  {patient_name}<br>"
                  f'<span style="color:{sc};font-size:11px">{lbl}</span>'),
            font=dict(size=12, color=APPLE_LABEL2), x=0.0, xanchor="left",
        ),
        xaxis=dict(range=[-0.5, N-0.5], showgrid=False, zeroline=False,
                   showticklabels=False, constrain="domain"),
        yaxis=dict(range=[N-0.5, -0.5], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x", scaleratio=1),
    )
    return fig


def make_brain_3d(state, band_powers, patient_name="Patient"):
    """
    Interactive 3D head + brain with EEG activity overlay.
    Drag to rotate, scroll to zoom. uirevision preserves camera between rerenders.
    """
    N_u, N_v = 64, 40
    u  = np.linspace(0, 2 * np.pi, N_u)
    v  = np.linspace(0, np.pi,     N_v)
    U, V = np.meshgrid(u, v)

    # ── Head ellipsoid (glass skull) ──────────────────────────────────────────
    Xh = 1.00 * np.sin(V) * np.cos(U)
    Yh = 0.82 * np.sin(V) * np.sin(U)
    Zh = 1.12 * np.cos(V)

    # ── Brain ellipsoid with subtle gyri texture ──────────────────────────────
    R    = 0.83
    gyri = 1.0 + 0.028 * np.sin(6*U) * np.cos(6*V) + 0.015 * np.sin(13*U) * np.cos(11*V)
    Xb = R * gyri * np.sin(V) * np.cos(U)
    Yb = R * 0.82 * gyri * np.sin(V) * np.sin(U)
    Zb = R * 1.12 * gyri * np.cos(V)

    # ── Standard 10-20 electrode positions (unit sphere: x=LR, y=AP, z=SI) ──
    ELEC3 = {
        "Fp1": (-0.30,  0.88,  0.36), "Fp2": ( 0.30,  0.88,  0.36),
        "F7":  (-0.82,  0.49,  0.29), "F3":  (-0.51,  0.62,  0.59),
        "Fz":  ( 0.00,  0.70,  0.72), "F4":  ( 0.51,  0.62,  0.59),
        "F8":  ( 0.82,  0.49,  0.29), "T7":  (-0.99,  0.00,  0.14),
        "C3":  (-0.67,  0.00,  0.74), "Cz":  ( 0.00,  0.00,  1.00),
        "C4":  ( 0.67,  0.00,  0.74), "T8":  ( 0.99,  0.00,  0.14),
        "P7":  (-0.82, -0.49,  0.29), "P3":  (-0.51, -0.62,  0.59),
        "Pz":  ( 0.00, -0.70,  0.72), "P4":  ( 0.51, -0.62,  0.59),
        "P8":  ( 0.82, -0.49,  0.29), "O1":  (-0.30, -0.88,  0.36),
        "Oz":  ( 0.00, -0.88,  0.36), "O2":  ( 0.30, -0.88,  0.36),
    }
    ELEC_BAND = {
        "Fp1":"beta","Fp2":"beta","F7":"theta","F3":"beta","Fz":"beta",
        "F4":"beta","F8":"theta","T7":"delta","T8":"delta","C3":"alpha",
        "Cz":"alpha","C4":"alpha","P7":"theta","P3":"alpha","Pz":"alpha",
        "P4":"alpha","P8":"theta","O1":"alpha","Oz":"alpha","O2":"alpha",
    }
    HOTSPOT = {
        "Seizure":     {"T7":5.0,"T8":5.0,"F7":3.5,"F8":3.5,"Fp1":3.0,"Fp2":3.0,"C3":2.2,"C4":2.2},
        "Pre-Seizure": {"Fp1":3.0,"Fp2":3.0,"F3":2.5,"F4":2.5,"Fz":2.2,"F7":1.8,"F8":1.8},
    }

    bp   = {b: float(band_powers.get(b, 0.25)) for b in ("delta","theta","alpha","beta")}
    tot  = sum(bp.values()) or 1.0
    bp   = {k: v / tot for k, v in bp.items()}
    names = list(ELEC3.keys())
    ev   = np.array([min(bp[ELEC_BAND[e]] * HOTSPOT.get(state, {}).get(e, 1.0), 1.0)
                     for e in names])
    ev_n = (ev - ev.min()) / (ev.max() - ev.min() + 1e-9)

    # Electrode 3D coords placed on brain surface
    ex3 = np.array([ELEC3[e][0] * R               for e in names])
    ey3 = np.array([ELEC3[e][1] * R * 0.82        for e in names])
    ez3 = np.array([ELEC3[e][2] * R * 1.12        for e in names])

    # ── Vectorised IDW: activity on brain surface ─────────────────────────────
    ex_n = np.array([ELEC3[e][0] for e in names])
    ey_n = np.array([ELEC3[e][1] for e in names])
    ez_n = np.array([ELEC3[e][2] for e in names])
    # Unit-sphere coordinates of brain surface (ignore gyri noise for IDW)
    Xbn = np.sin(V) * np.cos(U)
    Ybn = np.sin(V) * np.sin(U)
    Zbn = np.cos(V)
    d2  = np.maximum(
        (Xbn[np.newaxis] - ex_n[:, np.newaxis, np.newaxis])**2 +
        (Ybn[np.newaxis] - ey_n[:, np.newaxis, np.newaxis])**2 +
        (Zbn[np.newaxis] - ez_n[:, np.newaxis, np.newaxis])**2, 1e-4)   # (20, N_v, N_u)
    w   = 1.0 / d2
    act = np.einsum("e,eij->ij", ev_n, w) / w.sum(axis=0)
    act = np.clip(act, 0, 1)

    # ── Color scales (match clinical false-colour MRI palette) ───────────────
    CS = {
        "Seizure":     [[0,"#100000"],[0.30,"#700000"],[0.60,"#cc2200"],
                        [0.82,"#ff6600"],[1,"#ffcc00"]],
        "Pre-Seizure": [[0,"#100800"],[0.30,"#704000"],[0.60,"#cc8800"],
                        [0.82,"#ffaa00"],[1,"#fff176"]],
        "Normal":      [[0,"#000810"],[0.30,"#003070"],[0.60,"#0066cc"],
                        [0.82,"#00aaff"],[1,"#88ffee"]],
    }.get(state, [[0,"#111"],[1,"#fff"]])

    sc  = state_color(state)
    lbl = {"Seizure":    "⚡ SEIZURE DETECTED",
           "Pre-Seizure":"⚠ PRE-SEIZURE ACTIVITY",
           "Normal":     "● Normal Brain Activity"}.get(state, state)

    fig = go.Figure()

    # Glass skull — barely visible, gives the X-ray / transparent-head look
    fig.add_trace(go.Surface(
        x=Xh, y=Yh, z=Zh,
        colorscale=[[0,"rgba(160,190,220,0.03)"],[1,"rgba(180,210,240,0.03)"]],
        showscale=False, hoverinfo="skip", opacity=0.07,
        lighting=dict(ambient=0.9, diffuse=0.2, specular=1.2, roughness=0.04, fresnel=1.5),
        lightposition=dict(x=2, y=3, z=5),
    ))

    # Brain surface with EEG activity heatmap
    fig.add_trace(go.Surface(
        x=Xb, y=Yb, z=Zb,
        surfacecolor=act, colorscale=CS, cmin=0, cmax=1,
        showscale=True,
        colorbar=dict(
            title=dict(text="EEG Activity", font=dict(size=9, color=APPLE_GRAY)),
            tickfont=dict(size=8, color=APPLE_GRAY),
            len=0.45, thickness=10, x=1.01, xanchor="left",
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)",
        ),
        opacity=0.92,
        lighting=dict(ambient=0.35, diffuse=0.75, specular=0.55, roughness=0.38, fresnel=0.25),
        lightposition=dict(x=2, y=3, z=5),
        hovertemplate="Activity: %{surfacecolor:.2f}<extra></extra>",
    ))

    # Electrode scatter markers
    fig.add_trace(go.Scatter3d(
        x=ex3, y=ey3, z=ez3,
        mode="markers+text",
        marker=dict(color=ev_n.tolist(), colorscale=CS, cmin=0, cmax=1,
                    size=5, opacity=0.95,
                    line=dict(color="rgba(255,255,255,0.85)", width=1.2)),
        text=names,
        textfont=dict(size=6, color="rgba(255,255,255,0.50)"),
        textposition="top center",
        hovertemplate="<b>%{text}</b><br>Activity: %{marker.color:.2f}<extra></extra>",
        showlegend=False,
    ))

    # Focus markers for active states
    if state == "Seizure":
        fig.add_trace(go.Scatter3d(
            x=[ELEC3["T7"][0]*R, ELEC3["T8"][0]*R],
            y=[ELEC3["T7"][1]*R*0.82, ELEC3["T8"][1]*R*0.82],
            z=[ELEC3["T7"][2]*R*1.12, ELEC3["T8"][2]*R*1.12],
            mode="markers+text",
            marker=dict(size=13, color=BLOOD_RED, opacity=0.55,
                        line=dict(color=BLOOD_RED, width=2)),
            text=["⚡ Focus", "⚡ Focus"],
            textfont=dict(size=9, color=BLOOD_RED),
            textposition="top center", showlegend=False, hoverinfo="skip",
        ))
    elif state == "Pre-Seizure":
        fig.add_trace(go.Scatter3d(
            x=[ELEC3["Fp1"][0]*R], y=[ELEC3["Fp1"][1]*R*0.82], z=[ELEC3["Fp1"][2]*R*1.12],
            mode="markers+text",
            marker=dict(size=13, color=APPLE_AMBER, opacity=0.55,
                        line=dict(color=APPLE_AMBER, width=2)),
            text=["⚠ Elevated"], textfont=dict(size=9, color=APPLE_AMBER),
            textposition="top center", showlegend=False, hoverinfo="skip",
        ))

    fig.update_layout(
        paper_bgcolor="#000000",
        margin=dict(l=0, r=65, t=62, b=10),
        height=450,
        title=dict(
            text=(f"<b>3D Brain Activity</b>  ·  {patient_name}<br>"
                  f'<span style="color:{sc};font-size:11px">{lbl}</span><br>'
                  f'<span style="color:{APPLE_LABEL3};font-size:9px">'
                  f"Drag to rotate · Scroll to zoom · Right-click to pan</span>"),
            font=dict(size=12, color=APPLE_LABEL2), x=0.0, xanchor="left",
        ),
        scene=dict(
            bgcolor="#000000",
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            camera=dict(
                eye=dict(x=1.6, y=1.4, z=0.9),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            ),
            aspectmode="data",
        ),
        uirevision=f"3d_{patient_name}",
    )
    return fig


def make_head_brain_threejs(state: str, band_powers: dict,
                            patient_name: str = "Patient", height: int = 520) -> str:
    """Three.js 3D head with focal EEG vertex colouring — no external assets required."""
    _ELEC_BAND = {
        "Fp1":"beta",  "Fp2":"beta",  "F7":"theta", "F3":"beta",  "Fz":"beta",
        "F4":"beta",   "F8":"theta",  "T7":"delta", "T8":"delta", "C3":"alpha",
        "Cz":"alpha",  "C4":"alpha",  "P7":"theta", "P3":"alpha", "Pz":"alpha",
        "P4":"alpha",  "P8":"theta",  "O1":"alpha", "Oz":"alpha", "O2":"alpha",
    }
    _HOTSPOT = {
        "Seizure":     {"T7":5.0,"T8":5.0,"F7":3.5,"F8":3.5,"Fp1":3.0,"Fp2":3.0,"C3":2.2,"C4":2.2},
        "Pre-Seizure": {"Fp1":3.0,"Fp2":3.0,"F3":2.5,"F4":2.5,"Fz":2.2,"F7":1.8,"F8":1.8},
    }
    bp   = {b: float(band_powers.get(b, 0.25)) for b in ("delta","theta","alpha","beta")}
    tot  = sum(bp.values()) or 1.0
    bp   = {k: v/tot for k, v in bp.items()}
    names = list(_ELEC_BAND.keys())
    ev    = np.array([min(bp[_ELEC_BAND[e]] * _HOTSPOT.get(state, {}).get(e, 1.0), 1.0)
                      for e in names])
    ev_n  = (ev - ev.min()) / (ev.max() - ev.min() + 1e-9)
    elec_act = {e: round(float(ev_n[i]), 4) for i, e in enumerate(names)}

    srgb = {
        "Seizure":     [1.00, 0.18, 0.10],
        "Pre-Seizure": [1.00, 0.62, 0.04],
        "Normal":      [0.12, 0.84, 0.30],
    }.get(state, [0.50, 0.50, 0.50])

    cfg = json.dumps({"state": state, "patient": patient_name,
                      "act": elec_act, "srgb": srgb, "thr": 0.38})

    # --- HTML template (no f-string: JS braces kept literal) ---
    return ("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  *{margin:0;padding:0;box-sizing:border-box;}
  html,body{width:100%;height:100%;overflow:hidden;background:#000;}
  canvas{display:block;}
  #hud{position:absolute;top:8px;left:10px;
       font-family:-apple-system,'SF Pro Text',system-ui,sans-serif;
       pointer-events:none;}
  #ptname{font-size:13px;font-weight:600;color:#dde;}
  #stlbl{font-size:11px;margin-top:2px;}
  #hint{font-size:8px;color:#444;margin-top:3px;}
  #legend{position:absolute;bottom:6px;left:10px;font-size:8px;color:#333;
          font-family:system-ui,sans-serif;pointer-events:none;}
</style>
</head>
<body>
<div id="hud">
  <div id="ptname"></div>
  <div id="stlbl"></div>
  <div id="hint">Drag to rotate · Scroll to zoom</div>
</div>
<div id="legend">10-20 EEG · IDW Heatmap</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const CFG = __CFG__;

// HUD
const IC = {Seizure:'⚡','Pre-Seizure':'⚠',Normal:'●'};
const CC = {Seizure:'#FF3B30','Pre-Seizure':'#FF9F0A',Normal:'#32D74B'};
document.getElementById('ptname').textContent = CFG.patient;
const sl = document.getElementById('stlbl');
sl.textContent = (IC[CFG.state]||'●') + ' ' + CFG.state;
sl.style.color  = CC[CFG.state]||'#888';

// SCENE
const scene = new THREE.Scene();
const W = window.innerWidth, H = window.innerHeight;
const cam = new THREE.PerspectiveCamera(36, W/H, 0.05, 40);
cam.position.set(2.1, 0.35, 1.7);

const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
renderer.setSize(W, H);
document.body.appendChild(renderer.domElement);

// ORBIT CONTROLS
const ctrl = new THREE.OrbitControls(cam, renderer.domElement);
ctrl.enableDamping=true; ctrl.dampingFactor=0.07;
ctrl.enablePan=false; ctrl.minDistance=1.3; ctrl.maxDistance=5.5;
ctrl.autoRotate=true; ctrl.autoRotateSpeed=0.65;
ctrl.target.set(0,0.08,0);

// LIGHTS
scene.add(new THREE.AmbientLight(0x182038, 3.2));
const sun=new THREE.DirectionalLight(0xffffff,2.0);
sun.position.set(3,5,4); scene.add(sun);
const rim=new THREE.PointLight(0x3366cc,1.5,12);
rim.position.set(-3,2,-1); scene.add(rim);
const warm=new THREE.PointLight(0x331122,1.0,8);
warm.position.set(2,-3,3); scene.add(warm);

// ELECTRODE POSITIONS (Three.js x=LR, y=UP, z=FRONT)
// Mapped from 10-20 (Python: x=LR, y=AP, z=SI) → threeY=pyZ, threeZ=pyY
const EP = {
  Fp1:[-0.30,0.36, 0.88], Fp2:[ 0.30,0.36, 0.88],
  F7: [-0.82,0.29, 0.49], F3: [-0.51,0.59, 0.62],
  Fz: [ 0.00,0.72, 0.70], F4: [ 0.51,0.59, 0.62],
  F8: [ 0.82,0.29, 0.49], T7: [-0.99,0.14, 0.00],
  C3: [-0.67,0.74, 0.00], Cz: [ 0.00,1.00, 0.00],
  C4: [ 0.67,0.74, 0.00], T8: [ 0.99,0.14, 0.00],
  P7: [-0.82,0.29,-0.49], P3: [-0.51,0.59,-0.62],
  Pz: [ 0.00,0.72,-0.70], P4: [ 0.51,0.59,-0.62],
  P8: [ 0.82,0.29,-0.49], O1: [-0.30,0.36,-0.88],
  Oz: [ 0.00,0.36,-0.88], O2: [ 0.30,0.36,-0.88]
};
const EE = Object.entries(EP);
const THR=CFG.thr, SRGB=CFG.srgb;
const BASE=[0.05,0.07,0.14];

function idw(ox,oy,oz){
  let sw=0,sa=0;
  for(let k=0;k<EE.length;k++){
    const p=EE[k][1];
    const dx=ox-p[0],dy=oy-p[1],dz=oz-p[2];
    const d2=Math.max(dx*dx+dy*dy+dz*dz,0.004);
    const w=1/(d2*d2); sw+=w; sa+=(CFG.act[EE[k][0]]||0)*w;
  }
  return sa/sw;
}
function blendC(act){
  if(act>THR){
    const t=Math.min((act-THR)/(1-THR),1),s=t*t*(3-2*t);
    return [BASE[0]+(SRGB[0]-BASE[0])*s,
            BASE[1]+(SRGB[1]-BASE[1])*s,
            BASE[2]+(SRGB[2]-BASE[2])*s];
  }
  return BASE.slice();
}

// HEAD GEOMETRY — deformed unit sphere with vertex colours
const hGeo=new THREE.SphereGeometry(1.0,72,54);
const hPos=hGeo.attributes.position;
const nV=hPos.count;
const cBuf=new Float32Array(nV*3);

for(let i=0;i<nV;i++){
  let x=hPos.getX(i),y=hPos.getY(i),z=hPos.getZ(i);
  const ox=x,oy=y,oz=z;         // unit-sphere coords for IDW

  // Scale to head proportions: taller, narrower LR, shallower AP
  x*=0.90; y*=1.14; z*=0.87;

  // Nose bump (front-centre, mid-face)
  const nA=Math.max(0,oz-0.42)*Math.exp(-ox*ox*6)*Math.exp(-(oy+0.08)*(oy+0.08)*4.5);
  z+=nA*0.19; y-=nA*0.04;

  // Chin taper (lower front)
  const cA=Math.max(0,oz-0.15)*Math.max(0,-oy-0.52)*Math.exp(-ox*ox*5.5);
  z+=cA*0.06; y-=cA*0.10;

  // Flat occiput (rear of skull)
  const oA=Math.max(0,-oz-0.38)*Math.exp(-ox*ox*2)*Math.exp(-(oy-0.18)*(oy-0.18)*2.5);
  z-=oA*0.07;

  // Brow ridge (forehead-eye transition)
  const bA=Math.max(0,oz-0.58)*Math.exp(-ox*ox*3.5)*Math.exp(-(oy-0.60)*(oy-0.60)*14);
  z+=bA*0.055;

  hPos.setXYZ(i,x,y,z);

  // Per-vertex EEG colour
  const rgb=blendC(idw(ox,oy,oz));
  cBuf[i*3]=rgb[0]; cBuf[i*3+1]=rgb[1]; cBuf[i*3+2]=rgb[2];
}
hGeo.setAttribute('color',new THREE.BufferAttribute(cBuf,3));
hGeo.computeVertexNormals();

scene.add(new THREE.Mesh(hGeo, new THREE.MeshPhongMaterial({
  vertexColors:true, transparent:true, opacity:0.60,
  shininess:85, specular:new THREE.Color(0.28,0.35,0.55),
  side:THREE.DoubleSide
})));

// BRAIN — state-tinted inside the glass skull
const BC={
  Seizure:    {c:0x7a1010,e:0x3b0000},
  'Pre-Seizure':{c:0x6a4800,e:0x281a00},
  Normal:     {c:0x1a3060,e:0x000d28}
}[CFG.state]||{c:0x203050,e:0x000d1a};
const bMat=new THREE.MeshPhongMaterial({
  color:new THREE.Color(BC.c),emissive:new THREE.Color(BC.e),
  shininess:32,transparent:true,opacity:0.84
});

function buildHemi(scX,offX){
  const g=new THREE.SphereGeometry(0.58,52,36);
  const p=g.attributes.position;
  for(let i=0;i<p.count;i++){
    let x=p.getX(i)*scX*0.96,y=p.getY(i)*1.04,z=p.getZ(i)*0.88;
    // Gyri surface noise
    const n=1+0.033*Math.sin(x*27)*Math.cos(y*21)+0.019*Math.sin(z*30)*Math.cos(x*18);
    p.setXYZ(i,x*n+offX,y*n,z*n);
  }
  g.computeVertexNormals();
  return new THREE.Mesh(g,bMat);
}
scene.add(buildHemi(1,-0.09));  // left hemisphere
scene.add(buildHemi(1, 0.09));  // right hemisphere

// Cerebellum
const cGeo=new THREE.SphereGeometry(0.26,32,22);
const cpArr=cGeo.attributes.position;
for(let i=0;i<cpArr.count;i++){
  cpArr.setXYZ(i,cpArr.getX(i)*1.5,cpArr.getY(i)*0.72,cpArr.getZ(i)*0.65-0.50);
}
cGeo.computeVertexNormals();
const cMesh=new THREE.Mesh(cGeo,bMat);
cMesh.position.y=-0.32; scene.add(cMesh);

// Brainstem
const bsGeo=new THREE.CylinderGeometry(0.085,0.065,0.40,10);
bsGeo.computeVertexNormals();
const bsMesh=new THREE.Mesh(bsGeo,new THREE.MeshPhongMaterial({
  color:new THREE.Color(BC.c),emissive:new THREE.Color(BC.e),
  transparent:true,opacity:0.80
}));
bsMesh.position.set(0,-0.76,-0.08); scene.add(bsMesh);

// ELECTRODE DOTS on scalp surface
const dotGeo=new THREE.SphereGeometry(0.026,8,6);
const actC=new THREE.Color(SRGB[0],SRGB[1],SRGB[2]);
const idleC=new THREE.Color(0.5,0.5,0.65);
for(const [nm,p] of EE){
  const a=CFG.act[nm]||0;
  const dm=new THREE.Mesh(dotGeo,new THREE.MeshBasicMaterial({
    color:a>THR?actC.clone().lerp(idleC,1-(a-THR)/(1-THR+0.001)):idleC.clone(),
    transparent:true,opacity:a>THR?0.95:0.45
  }));
  // Place at head surface (apply same linear scaling as head deform)
  dm.position.set(p[0]*0.90,p[1]*1.14,p[2]*0.87);
  scene.add(dm);
}

// RESIZE
window.addEventListener('resize',()=>{
  cam.aspect=window.innerWidth/window.innerHeight;
  cam.updateProjectionMatrix();
  renderer.setSize(window.innerWidth,window.innerHeight);
});

// ANIMATE
(function loop(){requestAnimationFrame(loop);ctrl.update();renderer.render(scene,cam);})();
</script>
</body>
</html>""").replace("__CFG__", cfg)


@st.cache_data(ttl=30)
def _head_html(state: str, bp_tuple: tuple, patient_name: str, height: int = 520) -> str:
    return make_head_brain_threejs(state, dict(bp_tuple), patient_name, height)


def make_patient_map_embed(lat, lng, label, state):
    """Individual patient map with hospital overlays and distance from Palapye."""
    col        = {"Normal": APPLE_GREEN, "Pre-Seizure": APPLE_AMBER,
                  "Seizure": BLOOD_RED}.get(state, APPLE_GREEN)
    google_url = f"https://maps.google.com/?q={lat},{lng}"
    dist_you   = haversine(USER_LAT, USER_LNG, lat, lng)
    nearby     = nearest_hospitals(lat, lng, n=5)

    hosp_js = ""
    for d, h in nearby:
        hn = h["name"].replace("'", "\\'")
        ht = h["type"]
        hosp_js += f"""
        L.marker([{h['lat']},{h['lng']}],{{icon:L.divIcon({{
          html:'<div style="width:10px;height:10px;background:#fff;border:2px solid #FF453A;border-radius:2px;transform:rotate(45deg)"></div>',
          iconSize:[10,10],iconAnchor:[5,5],className:''
        }})}}).addTo(map)
        .bindPopup('<b style="color:#FF453A">🏥 {hn}</b><br>'
                  +'<span style="font-size:11px;color:#888">{ht}</span><br>'
                  +'<span style="font-size:10px;color:#0A84FF">📏 {d:.1f} km from patient</span>');
        """

    you_js = f"""
    L.marker([{USER_LAT},{USER_LNG}],{{icon:L.divIcon({{
      html:'<div style="width:16px;height:16px;background:#5E5CE6;border:2.5px solid #fff;border-radius:50%;box-shadow:0 0 12px #5E5CE6;display:flex;align-items:center;justify-content:center;font-size:8px;color:#fff;font-weight:700">YOU</div>',
      iconSize:[16,16],iconAnchor:[8,8],className:''
    }})}}).addTo(map)
    .bindPopup('📍 <b>Your Location</b> (Palapye)');
    L.polyline([[{lat},{lng}],[{USER_LAT},{USER_LNG}]],
      {{color:'{col}',weight:1.5,opacity:0.5,dashArray:'8,6'}}).addTo(map);
    """

    return f"""<!DOCTYPE html><html><head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:#000;font-family:-apple-system,sans-serif;}}
  #map{{width:100%;height:310px;border-radius:10px;}}
  .foot{{background:{APPLE_BG2};padding:8px 14px;display:flex;
         justify-content:space-between;align-items:center;
         border-top:0.5px solid {APPLE_BORDER};flex-wrap:wrap;gap:4px;}}
  .coord{{font-family:monospace;font-size:10px;color:{APPLE_GRAY};}}
  .dist{{font-size:10px;color:#5E5CE6;font-weight:600;}}
  .btn{{background:rgba(10,132,255,.12);border:0.5px solid rgba(10,132,255,.4);
        color:{APPLE_BLUE};padding:4px 12px;border-radius:8px;font-size:11px;
        cursor:pointer;text-decoration:none;}}
</style></head><body>
<div id="map"></div>
<div class="foot">
  <span class="coord">📍 {lat:.4f}, {lng:.4f} — {label}</span>
  <span class="dist">📏 {dist_you:.1f} km from you</span>
  <a class="btn" href="{google_url}" target="_blank">Google Maps ↗</a>
</div>
<script>
  var map=L.map('map',{{zoomControl:true,attributionControl:false}}).setView([{lat},{lng}],12);
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
    {{maxZoom:19,subdomains:'abcd'}}).addTo(map);
  var icon=L.divIcon({{
    html:'<div style="width:16px;height:16px;background:{col};border:3px solid #fff;border-radius:50%;box-shadow:0 0 14px {col};"></div>',
    iconSize:[16,16],iconAnchor:[8,8],className:''
  }});
  L.marker([{lat},{lng}],{{icon:icon}}).addTo(map).bindPopup('<b>{label}</b>').openPopup();
  L.circle([{lat},{lng}],{{color:'{col}',fillColor:'{col}',fillOpacity:0.06,radius:800,weight:1}}).addTo(map);
  {hosp_js}
  {you_js}
</script></body></html>"""


# =============================================================================
# SESSION STATE
# =============================================================================
for k, v in [
    ("notifications",[]), ("sms_last_states",{}), ("selected_pid",None),
    ("session_start",time.time()), ("session_seizures",0),
    ("state_entry_times",{}), ("sound_enabled",True),
    ("navigate_to_patient", False),
    ("refresh_interval", REFRESH_INTERVAL),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# Fragment auto-refresh interval — read from session state so slider changes
# take effect on the next full-page rerun (st.rerun(scope="app"))
_interval: int = int(st.session_state["refresh_interval"])

# =============================================================================
# PLOTLY DEFAULTS
# =============================================================================
_PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system,'SF Pro Display',Helvetica",
              color=APPLE_GRAY, size=10),
    margin=dict(l=10,r=10,t=28,b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)
_AX = dict(gridcolor=APPLE_BG3, zerolinecolor=APPLE_BORDER)

# =============================================================================
# FRAGMENT FUNCTIONS  (each auto-refreshes independently every REFRESH_INTERVAL)
# =============================================================================
def _sorted_pids_from(ap):
    return sorted(
        ap.keys(),
        key=lambda pid: (
            0 if is_live(pid) else 1,
            -PRIORITY.get(ap[pid].get("state", "Normal"), 0),
        ),
    )


@st.fragment(run_every=_interval)
def _render_sidebar():
    ap     = get_all_patients()
    status = get_status()
    spids  = _sorted_pids_from(ap)

    if st.session_state.selected_pid is None and spids:
        st.session_state.selected_pid = spids[0]

    # Navigate to Patient Detail tab if flagged by a button click
    if st.session_state.get("navigate_to_patient"):
        st.session_state.navigate_to_patient = False
        components.html("""<script>
        setTimeout(function(){
            var tabs=window.parent.document.querySelectorAll('button[role="tab"]');
            if(tabs.length>1) tabs[1].click();
        },200);
        </script>""", height=0)

    # Notification / seizure tracking
    for pid, pdata in ap.items():
        if not is_live(pid):
            continue
        new_state = pdata.get("state", "Normal")
        prev      = st.session_state.sms_last_states.get(pid)
        if new_state == prev:
            continue
        st.session_state.sms_last_states[pid]   = new_state
        st.session_state.state_entry_times[pid] = datetime.now()
        if new_state == "Seizure":
            st.session_state.session_seizures += 1
        st.session_state.notifications.insert(0, {
            "state":   new_state,
            "patient": pdata.get("patient_name", "—"),
            "ward":    pdata.get("ward", "—"),
            "bed":     pdata.get("bed", "—"),
            "conf":    int(pdata.get("confidence", 0) * 100),
            "ts":      datetime.now().strftime("%H:%M:%S"),
            "sms":     pdata.get("sms_sent", False),
            "gps_lat": pdata.get("gps_lat", ""),
            "gps_lng": pdata.get("gps_lng", ""),
            "pid":     pid,
        })
    st.session_state.notifications = st.session_state.notifications[:20]

    # Alert sound
    if st.session_state.sound_enabled and any(
            ap[pid].get("state") == "Seizure" for pid in ap if is_live(pid)):
        components.html("""<script>
        (function(){
          try{
            var ctx=new(window.AudioContext||window.webkitAudioContext)();
            function beep(f,d,v){
              var o=ctx.createOscillator(),g=ctx.createGain();
              o.connect(g);g.connect(ctx.destination);
              o.frequency.value=f;g.gain.value=v;
              o.start(ctx.currentTime);o.stop(ctx.currentTime+d);
              g.gain.exponentialRampToValueAtTime(0.001,ctx.currentTime+d);
            }
            beep(880,.18,.15);setTimeout(()=>beep(880,.18,.12),280);
          }catch(e){}
        })();
        </script>""", height=0)

    cur_ri = int(st.session_state.refresh_interval)
    st.caption(f"Refresh every {cur_ri}s · {datetime.now().strftime('%H:%M:%S')}")

    uptime_s = time.time() - st.session_state.session_start
    st.markdown(f"""<div class="uptime-box">
        Uptime <span style="color:{APPLE_BLUE};font-variant-numeric:tabular-nums">
        {format_uptime(uptime_s)}</span>&nbsp;·&nbsp;
        Seizures <span style="color:{BLOOD_RED};font-weight:700">
        {st.session_state.session_seizures}</span>
    </div>""", unsafe_allow_html=True)

    # ── Refresh interval slider ───────────────────────────────────────────────
    st.markdown('<div class="section-label">Refresh Interval</div>', unsafe_allow_html=True)
    new_ri = st.select_slider(
        "Refresh every",
        options=[2, 3, 5, 8, 10, 15, 20, 30, 60],
        value=cur_ri if cur_ri in [2,3,5,8,10,15,20,30,60] else 5,
        format_func=lambda x: f"{x}s",
        key="ri_slider",
        label_visibility="collapsed",
    )
    if new_ri != cur_ri:
        st.session_state.refresh_interval = new_ri
        st.rerun(scope="app")   # full rerun so fragments re-register with new interval

    # ── Alert sound toggle ────────────────────────────────────────────────────
    sc1, sc2 = st.columns([3, 1])
    with sc1: st.caption(" Alert sound")
    with sc2:
        st.session_state.sound_enabled = st.toggle(
            "Alert Sound", value=st.session_state.sound_enabled,
            key="sound_toggle", label_visibility="collapsed")

    st.markdown("---")

    # Pi status
    phase = status.get("phase", "WAITING")
    msg   = status.get("message", "")
    pc = {"DETECTING": APPLE_GREEN, "TRAINING": APPLE_BLUE,
          "PROCESSING": APPLE_AMBER, "LOADING": APPLE_AMBER,
          "ERROR": BLOOD_RED, "STOPPED": BLOOD_RED}.get(phase, APPLE_GRAY)
    st.markdown(f"""<div class="glass-panel" style="padding:10px 14px;margin-bottom:8px;">
      <span style="font-size:10px;font-weight:700;
          background:rgba(255,255,255,0.06);color:{pc};
          padding:2px 8px;border-radius:6px;letter-spacing:.06em;">{phase}</span>
      <div style="font-size:11px;color:{APPLE_AMBER};margin-top:6px;font-weight:500">{msg}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Patients</div>', unsafe_allow_html=True)
    for pid in spids:
        pdata  = ap[pid]
        state  = pdata.get("state", "Normal")
        col    = state_color(state)
        icon   = CLASS_ICONS[state]
        conf   = int(pdata.get("confidence", 0) * 100)
        name   = pdata.get("patient_name", "—")
        is_sel = pid == st.session_state.selected_pid
        badge  = "LIVE" if is_live(pid) else "DEMO"
        blink  = "animation:seizure-blink 1s ease-in-out infinite;" if state == "Seizure" else ""

        row_l, row_r = st.columns([5, 1])
        with row_l:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;
                 background:{'rgba(10,132,255,.10)' if is_sel else 'transparent'};
                 border-left:3px solid {col};border-radius:7px;
                 padding:7px 10px;margin-bottom:1px;{blink}">
              <div style="width:8px;height:8px;border-radius:50%;
                          background:{col};flex-shrink:0;"></div>
              <div style="flex:1;min-width:0">
                <span style="font-size:12px;font-weight:600;color:{APPLE_LABEL};
                             white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                  {name}
                </span>
                <span style="font-size:9px;color:{APPLE_GRAY};margin-left:5px">{badge}</span><br>
                <span style="font-size:10px;color:{col}">{icon} {state}</span>
                <span style="font-size:10px;color:{APPLE_LABEL3};margin-left:5px">{conf}%</span>
              </div>
            </div>""", unsafe_allow_html=True)
        with row_r:
            if st.button("→", key=f"sel_{pid}", use_container_width=True,
                         help=f"View {name} in Patient Detail"):
                st.session_state.selected_pid = pid
                st.session_state.navigate_to_patient = True
                st.rerun(scope="fragment")

    # Alerts
    if st.session_state.notifications:
        st.markdown('<div class="section-label">Alerts</div>', unsafe_allow_html=True)
        for n in st.session_state.notifications[:8]:
            s   = n["state"]
            css = ("notif-seizure" if s == "Seizure"
                   else "notif-pre" if s == "Pre-Seizure" else "notif-normal")
            ico = "⚡" if s == "Seizure" else "⚠️" if s == "Pre-Seizure" else "✅"
            sms = (f' <span style="font-size:10px;color:{APPLE_GREEN}">(SMS ✓)</span>'
                   if n.get("sms") else "")
            gps = ""
            if n.get("gps_lat") and n.get("gps_lng"):
                gm  = f"https://maps.google.com/?q={n['gps_lat']},{n['gps_lng']}"
                gps = f'<br><a href="{gm}" target="_blank" style="font-size:10px;color:{APPLE_BLUE}">📍 Maps</a>'
            st.markdown(f"""<div class="{css}">
              <b>{ico} {s.upper()}</b>{sms}<br>
              {n['patient']} — {n['ward']}, {n['bed']}<br>
              <span style="font-size:10px;color:{APPLE_LABEL3}">
                {n['conf']}% · {n['ts']}</span>{gps}
            </div>""", unsafe_allow_html=True)


@st.fragment(run_every=_interval)
def _render_header():
    ap    = get_all_patients()
    spids = _sorted_pids_from(ap)

    for p in (p for pid, p in ap.items() if p.get("state") == "Seizure" and is_live(pid)):
        sms = " — Doctor notified via SMS ✓" if p.get("sms_sent") else ""
        st.markdown(
            f'<div class="alert-seizure">⚡ ACTIVE SEIZURE — {p["patient_name"]}'
            f'{sms} — Immediate attention required</div>',
            unsafe_allow_html=True)

    for p in (p for pid, p in ap.items() if p.get("state") == "Pre-Seizure" and is_live(pid)):
        st.markdown(
            f'<div class="alert-pre">⚠️ Pre-Seizure Warning — {p["patient_name"]}'
            f' — Administer medication</div>',
            unsafe_allow_html=True)

    n_total   = len(ap)
    n_live    = sum(1 for pid in ap if is_live(pid))
    n_demo    = n_total - n_live
    n_normal  = sum(1 for p in ap.values() if p.get("state") == "Normal")
    n_pre     = sum(1 for p in ap.values() if p.get("state") == "Pre-Seizure")
    n_seizure = sum(1 for pid, p in ap.items()
                    if p.get("state") == "Seizure" and is_live(pid))

    c1, c2, c3, c4, c5 = st.columns(5)

    def _card(col, label, val, color, sub):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{val}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    _card(c1, "Patients",    n_total,   APPLE_BLUE,  f"{n_live} live · {n_demo} demo")
    _card(c2, "Normal",      n_normal,  APPLE_GREEN, "No alerts")
    _card(c3, "Pre-Seizure", n_pre,     APPLE_AMBER, "Warning state")

    if n_seizure > 0:
        c4.markdown(f"""<div class="metric-card"
            style="animation:seizure-blink 1s ease-in-out infinite;
                   border-left:3px solid {BLOOD_RED};">
            <div class="metric-label">Active Seizure</div>
            <div class="metric-value" style="color:{BLOOD_HOT}">{n_seizure}</div>
            <div class="metric-sub" style="color:{BLOOD_RED}">⚡ Live patient</div>
        </div>""", unsafe_allow_html=True)
    else:
        _card(c4, "Active Seizure", 0, APPLE_GRAY, "None active")

    _card(c5, "Session Total",
          st.session_state.session_seizures, BLOOD_RED,
          f"{format_uptime(time.time()-st.session_state.session_start)} uptime")

    st.markdown("<br>", unsafe_allow_html=True)


@st.fragment(run_every=_interval)
def _render_overview():
    ap    = get_all_patients()
    spids = _sorted_pids_from(ap)

    if not ap:
        st.info(" Waiting for main_pi.py to start detection…")
        return

    live_pids = [pid for pid in spids if is_live(pid)]
    demo_pids = [pid for pid in spids if not is_live(pid)]

    _TAB_JS = """<script>setTimeout(function(){
        var t=window.parent.document.querySelectorAll('button[role="tab"]');
        if(t.length>1)t[1].click();},200);</script>"""

    if live_pids:
        st.markdown('<div class="section-label">Live Patients</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(live_pids), 3))
        for i, pid in enumerate(live_pids[:3]):
            p      = ap[pid]
            state  = p.get("state", "Normal")
            col    = state_color(state)
            conf   = int(p.get("confidence", 0) * 100)
            name   = p.get("patient_name", "—")
            blink  = "animation:seizure-blink 1s ease-in-out infinite;" if state == "Seizure" else ""
            avatar = get_patient_avatar_html(pid, name, size=44)
            et     = st.session_state.state_entry_times.get(pid)
            dur    = (f'<span style="font-size:10px;color:{APPLE_LABEL3};">'
                      f'In {state} {(datetime.now()-et).seconds}s</span>' if et else "")
            with cols[i]:
                st.markdown(f"""
                <div class="glass-panel" style="border-left:4px solid {col};{blink}padding:14px 16px;">
                  <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
                    {avatar}
                    <div style="flex:1;min-width:0">
                      <div style="font-size:14px;font-weight:700;color:{APPLE_LABEL};
                                  white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                        {name} <span class="badge-live">LIVE</span>
                      </div>
                      <div style="font-size:11px;color:{APPLE_LABEL2};">
                        {p.get('ward','—')} · {p.get('bed','—')}
                      </div>
                      {dur}
                    </div>
                  </div>
                  <div style="font-size:24px;font-weight:700;color:{col};
                              letter-spacing:-.5px;">{CLASS_ICONS[state]} {state}</div>
                  <div style="font-size:11px;color:{APPLE_LABEL3};margin-top:4px;">
                    Confidence: <span style="color:{col};font-weight:600">{conf}%</span>
                  </div>
                </div>""", unsafe_allow_html=True)
                if st.button("View Patient →", key=f"ov_view_{pid}",
                             use_container_width=True):
                    st.session_state.selected_pid = pid
                    st.session_state.navigate_to_patient = True
                    components.html(_TAB_JS, height=0)

    if demo_pids:
        st.markdown('<div class="section-label">Demo Patients</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(demo_pids), 4))
        for i, pid in enumerate(demo_pids):
            p      = ap[pid]
            state  = p.get("state", "Normal")
            col    = state_color(state)
            conf   = int(p.get("confidence", 0) * 100)
            name   = p.get("patient_name", "—")
            avatar = get_patient_avatar_html(pid, name, size=36)
            with cols[i % 4]:
                st.markdown(f"""
                <div class="glass-panel" style="border-left:3px solid {col};opacity:.80;padding:12px 14px;">
                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                    {avatar}
                    <div style="flex:1;min-width:0">
                      <div style="font-size:12px;font-weight:600;color:{APPLE_LABEL};
                                  white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                        {name} <span class="badge-demo">DEMO</span>
                      </div>
                      <div style="font-size:10px;color:{APPLE_LABEL3};">
                        {p.get('ward','—')} · {p.get('bed','—')}
                      </div>
                    </div>
                  </div>
                  <div style="font-size:15px;font-weight:700;color:{col};">
                    {CLASS_ICONS[state]} {state}
                    <span style="font-size:10px;opacity:.6">({conf}%)</span>
                  </div>
                </div>""", unsafe_allow_html=True)
                if st.button("View Patient →", key=f"ov_view_{pid}",
                             use_container_width=True):
                    st.session_state.selected_pid = pid
                    st.session_state.navigate_to_patient = True
                    components.html(_TAB_JS, height=0)


@st.fragment(run_every=_interval)
def _render_patient():
    ap    = get_all_patients()
    spids = _sorted_pids_from(ap)

    if not spids:
        st.info("⏳ No patient data yet.")
        return

    pid   = st.session_state.selected_pid or spids[0]
    pdata = ap.get(pid, {})
    state = pdata.get("state", "Normal")
    col   = state_color(state)
    conf  = int(pdata.get("confidence", 0) * 100)
    blink = "animation:seizure-blink 1s ease-in-out infinite;" if state == "Seizure" else ""

    st.markdown(f"""
    <div class="glass-panel" style="border-left:5px solid {col};
         margin-bottom:18px;{blink}">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
          <span style="font-size:20px;font-weight:700;letter-spacing:-.4px;
                       color:{APPLE_LABEL};">
            {CLASS_ICONS[state]} {pdata.get('patient_name','—')}
          </span>
          {'<span class="badge-live">LIVE</span>' if is_live(pid)
           else '<span class="badge-demo">DEMO</span>'}
          <div style="font-size:12px;color:{APPLE_LABEL2};margin-top:4px;">
            {pdata.get('ward','—')} · {pdata.get('bed','—')}
          </div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:26px;font-weight:700;letter-spacing:-.5px;
                      color:{col};">{state}</div>
          <div style="font-size:11px;color:{APPLE_LABEL3};">
            Confidence: <span style="color:{col};font-weight:600">{conf}%</span>
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    history = get_patient_history(pid)

    # ── Live Brain Topomap ────────────────────────────────────────────────────
    # Pull band powers from latest history row, or use state-specific defaults
    latest_bp: dict = {}
    if not history.empty:
        row       = history.iloc[-1]
        have_cols = [b for b in ("delta", "theta", "alpha", "beta") if b in history.columns]
        latest_bp = {b: float(row.get(b, 0.25)) for b in have_cols}
    if not latest_bp:
        latest_bp = {
            "Seizure":     {"delta": 0.40, "theta": 0.30, "alpha": 0.15, "beta": 0.15},
            "Pre-Seizure": {"delta": 0.25, "theta": 0.35, "alpha": 0.20, "beta": 0.20},
            "Normal":      {"delta": 0.20, "theta": 0.20, "alpha": 0.40, "beta": 0.20},
        }.get(state, {"delta": 0.25, "theta": 0.25, "alpha": 0.25, "beta": 0.25})

    # ── 50/50 left (charts) / right (brain MRI) ──────────────────────────────
    left_col, right_col = st.columns(2)

    with right_col:
        _html = _head_html(
            state,
            tuple(sorted(latest_bp.items())),
            pdata.get("patient_name", "Patient"),
        )
        components.html(_html, height=520, scrolling=False)

    with left_col:
        # EEG State Distribution donut
        if not history.empty:
            counts = history["state"].value_counts()
            fig_p  = go.Figure(go.Pie(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                marker=dict(colors=[
                    {"Normal": APPLE_GREEN, "Pre-Seizure": APPLE_AMBER,
                     "Seizure": BLOOD_RED}.get(lbl, APPLE_GRAY)
                    for lbl in counts.index]),
                hole=0.58,
                textfont=dict(size=11, color=APPLE_LABEL),
            ))
            fig_p.update_layout(**_PL, height=210,
                                title=dict(text="EEG State Distribution",
                                           font=dict(size=12, color=APPLE_LABEL2)))
            st.plotly_chart(fig_p, use_container_width=True,
                            config={"displayModeBar": False}, key=f"pie_{pid}")
        else:
            st.markdown('<div class="section-label">EEG State Distribution</div>',
                        unsafe_allow_html=True)
            st.caption("Awaiting history data…")

        # Band power bars
        st.markdown('<div class="section-label">Band Power</div>', unsafe_allow_html=True)
        total_bp    = sum(latest_bp.values()) or 1.0
        band_colors = {"delta": APPLE_BLUE, "theta": APPLE_AMBER,
                       "alpha": APPLE_GREEN, "beta": BLOOD_RED}
        for band, bval in latest_bp.items():
            pct = bval / total_bp * 100
            bc  = band_colors.get(band, APPLE_GRAY)
            st.markdown(f"""
            <div style="margin-bottom:10px">
              <div style="display:flex;justify-content:space-between;
                          font-size:11px;color:{APPLE_LABEL2};margin-bottom:3px;">
                <span style="font-weight:600;color:{bc}">{band.capitalize()}</span>
                <span>{pct:.1f}%</span>
              </div>
              <div style="background:{APPLE_BG3};border-radius:4px;height:6px;">
                <div style="width:{min(pct,100):.1f}%;height:6px;background:{bc};
                     border-radius:4px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

    if history.empty:
        st.info("⏳ Recording history… check back in a few seconds.")
    else:
        tail      = history.tail(300)
        state_num = tail["state"].map({"Normal": 0, "Pre-Seizure": 1, "Seizure": 2})
        cseq      = [APPLE_GREEN if s == "Normal" else APPLE_AMBER if s == "Pre-Seizure"
                     else BLOOD_RED for s in tail["state"]]
        fig_s = go.Figure(go.Scatter(
            x=tail["timestamp"], y=state_num,
            mode="lines+markers",
            marker=dict(color=cseq, size=4),
            line=dict(color="rgba(255,255,255,.07)", width=1),
            hovertemplate="%{text}<extra></extra>", text=tail["state"],
        ))
        fig_s.update_layout(**_PL, height=175, uirevision=f"state_{pid}",
                            title=dict(text="EEG State Timeline",
                                       font=dict(size=12, color=APPLE_LABEL2)),
                            yaxis=dict(tickvals=[0, 1, 2],
                                       ticktext=["Normal", "Pre-Sz", "Seizure"], **_AX),
                            xaxis=dict(**_AX))
        st.plotly_chart(fig_s, use_container_width=True,
                        config={"displayModeBar": False}, key=f"state_{pid}")

        band_cols = [c for c in ["delta", "theta", "alpha", "beta"] if c in history.columns]
        if band_cols:
            t200  = history.tail(200)
            fig_b = go.Figure()
            bc    = {"delta": APPLE_BLUE, "theta": APPLE_AMBER,
                     "alpha": APPLE_GREEN, "beta": BLOOD_RED}
            for b in band_cols:
                fig_b.add_trace(go.Scatter(
                    x=t200["timestamp"], y=t200[b], name=b.capitalize(),
                    mode="lines", line=dict(color=bc.get(b, "#fff"), width=1.5)))
            fig_b.update_layout(**_PL, height=210, uirevision=f"band_{pid}",
                                title=dict(text="EEG Band Power",
                                           font=dict(size=12, color=APPLE_LABEL2)),
                                xaxis=dict(**_AX), yaxis=dict(**_AX))
            st.plotly_chart(fig_b, use_container_width=True,
                            config={"displayModeBar": False}, key=f"band_{pid}")

        fig_c = go.Figure(go.Scatter(
            x=history.tail(200)["timestamp"],
            y=history.tail(200)["confidence"] * 100,
            mode="lines", fill="tozeroy",
            line=dict(color=APPLE_BLUE, width=1.5),
            fillcolor="rgba(10,132,255,0.07)",
        ))
        fig_c.update_layout(**_PL, height=155, uirevision=f"conf_{pid}",
                            title=dict(text="Classifier Confidence (%)",
                                       font=dict(size=12, color=APPLE_LABEL2)),
                            xaxis=dict(**_AX), yaxis=dict(range=[0, 105], **_AX))
        st.plotly_chart(fig_c, use_container_width=True,
                        config={"displayModeBar": False}, key=f"conf_{pid}")



@st.fragment(run_every=_interval)
def _render_map():
    ap    = get_all_patients()
    spids = _sorted_pids_from(ap)

    # ── Overview map (all patients + hospitals + you) ─────────────────────────
    st.markdown('<div class="section-label">Ward Map — Botswana</div>',
                unsafe_allow_html=True)
    map_html = make_all_patients_map(ap, spids)
    if map_html:
        components.html(map_html, height=468, scrolling=False)
        st.caption(
            "🔴 Seizure · 🟡 Pre-Seizure · 🟢 Normal · 🟣 You (Palapye) · ◇ Hospital/Clinic  "
            "— Click any marker for details · Distance lines from your location")
    else:
        st.info("No GPS coordinates found.")

    # ── Distance-from-Palapye summary cards ───────────────────────────────────
    dist_rows = []
    for pid in spids:
        p = ap[pid]
        try:
            d = haversine(USER_LAT, USER_LNG, float(p["gps_lat"]), float(p["gps_lng"]))
            dist_rows.append((d, pid, p))
        except (KeyError, TypeError, ValueError):
            pass
    dist_rows.sort(key=lambda x: x[0])

    if dist_rows:
        st.markdown('<div class="section-label">Distance from You (Palapye)</div>',
                    unsafe_allow_html=True)
        dcols = st.columns(min(len(dist_rows), 4))
        for i, (d, pid, p) in enumerate(dist_rows[:4]):
            s  = p.get("state", "Normal")
            c  = state_color(s)
            city = p.get("city", p.get("ward", "—"))
            dcols[i].markdown(f"""
            <div class="glass-panel" style="border-left:3px solid {c};
                 padding:10px 12px;text-align:center;margin-bottom:0;">
              <div style="font-size:11px;font-weight:600;color:{APPLE_LABEL}">
                {p.get('patient_name','—')}
              </div>
              <div style="font-size:10px;color:{APPLE_LABEL3};margin-bottom:4px">{city}</div>
              <div style="font-size:22px;font-weight:700;color:{APPLE_BLUE}">{d:.0f} km</div>
              <div style="font-size:10px;color:{c};margin-top:2px">
                {CLASS_ICONS[s]} {s}
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Individual patient deep-dive ──────────────────────────────────────────
    st.markdown('<div class="section-label">Patient Location Detail</div>',
                unsafe_allow_html=True)
    opts = {ap[p].get("patient_name", "—") + (" 🔴" if is_live(p) else " ⬜"): p
            for p in spids if ap[p].get("gps_lat") and ap[p].get("gps_lng")}

    if not opts:
        st.info("No GPS coordinates configured for any patient.")
        return

    sel = st.selectbox("Patient", list(opts.keys()),
                       label_visibility="collapsed", key="map_patient_sel")
    mp  = ap[opts[sel]]
    mc  = state_color(mp.get("state", "Normal"))
    mlat, mlng = float(mp["gps_lat"]), float(mp["gps_lng"])
    mdist      = haversine(USER_LAT, USER_LNG, mlat, mlng)

    # Patient info banner
    st.markdown(f"""
    <div class="glass-panel" style="border-left:4px solid {mc};
         padding:10px 16px;margin-bottom:8px;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
          {CLASS_ICONS[mp.get('state','Normal')]}
          <b style="color:{APPLE_LABEL};font-size:13px">{mp.get('patient_name','—')}</b>
          &nbsp;·&nbsp;
          <span style="color:{APPLE_LABEL2};font-size:12px">
            {mp.get('ward','—')} · {mp.get('bed','—')}
          </span>
        </div>
        <div style="text-align:right;">
          <span style="color:{mc};font-size:13px;font-weight:700">
            {mp.get('state','Normal')}
          </span><br>
          <span style="color:#5E5CE6;font-size:11px">📏 {mdist:.1f} km from you</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Map embed with hospitals + distance line to you
    components.html(
        make_patient_map_embed(mlat, mlng, mp.get("patient_name", "—"),
                               mp.get("state", "Normal")),
        height=390, scrolling=False)

    # Nearby hospitals table
    st.markdown('<div class="section-label">Nearby Hospitals & Clinics</div>',
                unsafe_allow_html=True)
    nearby = nearest_hospitals(mlat, mlng, n=5)
    htype_color = {
        "General": APPLE_BLUE, "Private": APPLE_GREEN, "Mission": APPLE_AMBER,
        "Primary": APPLE_INDIGO, "Clinic": APPLE_GREEN,
        "Psychiatric": APPLE_RED, "Referral": APPLE_BLUE,
    }
    for d, h in nearby:
        dist_you = haversine(USER_LAT, USER_LNG, h["lat"], h["lng"])
        hc       = htype_color.get(h["type"], APPLE_GRAY)
        st.markdown(f"""
        <div class="glass-panel" style="padding:9px 14px;margin-bottom:5px;
             border-left:3px solid {hc};">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:12px;font-weight:600;color:{APPLE_LABEL}">
              🏥 {h['name']}
            </span>
            <span style="font-size:10px;background:rgba(255,255,255,.06);
                 color:{hc};padding:2px 8px;border-radius:6px">{h['type']}</span>
          </div>
          <div style="font-size:11px;color:{APPLE_LABEL2};margin-top:3px;">
            📏 <b style="color:{APPLE_BLUE}">{d:.1f} km</b> from patient
            &nbsp;·&nbsp;
            <span style="color:#5E5CE6">{dist_you:.1f} km from you (Palapye)</span>
            &nbsp;·&nbsp;
            <span style="color:{APPLE_LABEL3}">{h['city']}</span>
          </div>
        </div>""", unsafe_allow_html=True)


@st.fragment(run_every=_interval)
def _render_log():
    ap    = get_all_patients()
    spids = _sorted_pids_from(ap)

    opts = {ap[p].get("patient_name", "—") + (" 🔴" if is_live(p) else " ⬜"): p
            for p in spids}
    sel  = st.selectbox("Patient", list(opts.keys()),
                        label_visibility="collapsed", key="log_patient_sel")
    lpid = opts.get(sel, spids[0] if spids else None)

    if lpid:
        history = get_patient_history(lpid)
        if history.empty:
            st.info("No history yet for this patient.")
        else:
            if not is_live(lpid):
                st.caption(" Demo patient — simulated data")
            f1, f2, f3 = st.columns(3)
            with f1: sf = st.multiselect("State", CLASSES, default=CLASSES,
                                         label_visibility="collapsed", key="log_sf")
            with f2: nr = st.slider("Rows", 10, 200, 50,
                                    label_visibility="collapsed", key="log_nr")
            with f3: so = st.checkbox("Seizures only", value=False, key="log_so")
            if so:
                sf = ["Seizure"]

            disp  = history[history["state"].isin(sf)].tail(nr).reset_index(drop=True)
            bands = [c for c in ["delta", "theta", "alpha", "beta"] if c in history.columns]
            scols = (["timestamp", "state", "confidence"] + bands + ["sms_sent"]
                     if is_live(lpid) else ["timestamp", "state", "confidence", "sms_sent"])
            show  = disp[[c for c in scols if c in disp.columns]].copy()
            show["timestamp"]  = show["timestamp"].dt.strftime("%H:%M:%S")
            show["confidence"] = (show["confidence"] * 100).round(1).astype(str) + "%"
            st.dataframe(show, use_container_width=True, height=360)

            sz = len(history[history["state"] == "Seizure"])
            ps = len(history[history["state"] == "Pre-Seizure"])
            st.caption(f"Records: {len(history)} · Seizure: {sz} · "
                       f"Pre-Seizure: {ps} · Rate: {sz/len(history)*100:.1f}%")
            st.download_button(
                "⬇ Download CSV",
                data=history.to_csv(index=False).encode("utf-8"),
                file_name=f"neurowatch_{lpid}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key=f"dl_{lpid}")


@st.fragment(run_every=_interval)
def _render_metrics():
    metrics = get_metrics()
    if not metrics:
        st.info("⏳ Metrics available after first training cycle.")
        return

    ts = metrics.get("timestamp", "")
    try:
        ts = datetime.fromisoformat(ts).strftime("%d %b %Y  %H:%M")
    except Exception:
        pass
    st.caption(f"Trained: {ts}  ·  Bonn University Dataset · SVM / Random Forest")

    ks  = ["accuracy", "precision", "recall", "f1", "specificity", "seizure_recall"]
    lbs = ["Accuracy", "Precision", "Recall", "F1 Score", "Specificity", "Seiz. Recall"]
    for col, label, key in zip(st.columns(6), lbs, ks):
        v = metrics.get(key, 0)
        c = APPLE_GREEN if v >= .9 else APPLE_AMBER if v >= .7 else BLOOD_RED
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{c}">{v*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    vals   = [metrics.get(k, 0) for k in ks]
    colors = [APPLE_GREEN if v >= .9 else APPLE_AMBER if v >= .7 else BLOOD_RED for v in vals]
    fig_m  = go.Figure(go.Bar(
        x=vals, y=lbs, orientation="h", marker_color=colors,
        text=[f"{v*100:.1f}%" for v in vals], textposition="outside",
        textfont=dict(size=11, color=APPLE_LABEL2)))
    fig_m.update_layout(**_PL, height=250, showlegend=False,
                        xaxis=dict(range=[0, 1.2], **_AX), yaxis=dict(**_AX))
    st.plotly_chart(fig_m, use_container_width=True,
                    config={"displayModeBar": False}, key="metrics_bar")

    cm = metrics.get("confusion_matrix")
    if cm:
        st.markdown('<div class="section-label">Confusion Matrix</div>',
                    unsafe_allow_html=True)
        ca  = np.array(cm)
        lb  = CLASSES[:ca.shape[0]]
        fig_cm = go.Figure(go.Heatmap(
            z=ca, x=lb, y=lb,
            colorscale=[[0, APPLE_BG2], [1, APPLE_BLUE]],
            text=ca, texttemplate="%{text}",
            textfont=dict(size=14, color=APPLE_LABEL), showscale=False))
        fig_cm.update_layout(**_PL, height=290,
                             xaxis=dict(title="Predicted", side="bottom", **_AX),
                             yaxis=dict(title="True", autorange="reversed", **_AX))
        st.plotly_chart(fig_cm, use_container_width=True,
                        config={"displayModeBar": False}, key="metrics_cm")


# =============================================================================
# STATIC LAYOUT  (runs once — tabs are never re-created so selection persists)
# =============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title"> NeuroWatch</div>', unsafe_allow_html=True)
    st.markdown('<div class="live-badge"><span class="live-dot"></span> Live Monitoring</div>',
                unsafe_allow_html=True)
    _render_sidebar()

_render_header()

tab_overview, tab_patient, tab_map, tab_log, tab_metrics_tab = st.tabs([
    "  Ward Overview",
    "  Patient Detail",
    "  Ward Map",
    "  Event Log",
    "  Model Metrics",
])

with tab_overview:    _render_overview()
with tab_patient:     _render_patient()
with tab_map:         _render_map()
with tab_log:         _render_log()
with tab_metrics_tab: _render_metrics()