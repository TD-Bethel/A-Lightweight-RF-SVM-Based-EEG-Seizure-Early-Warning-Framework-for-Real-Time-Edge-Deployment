# =============================================================================
# NEUROWATCH — Multi-Patient EEG Seizure Monitor
# Apple HIG colour system — Dark Mode
# Seizure state: blood-red (#C0392B / #FF0000) with CSS blink animation
# =============================================================================

import os
import json
import time
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
PATIENTS_JSON    = os.path.join(BASE_DIR, "neurowatch_patients.json")
METRICS_JSON     = os.path.join(BASE_DIR, "neurowatch_metrics.json")
STATUS_JSON      = os.path.join(BASE_DIR, "neurowatch_status.json")
REFRESH_INTERVAL = 3

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

def get_all_patients():  return read_json(PATIENTS_JSON, {})
def get_metrics():       return read_json(METRICS_JSON, {})
def get_status():        return read_json(STATUS_JSON, {"phase":"WAITING","message":"Waiting for Pi..."})

def get_patient_history(pid):
    path = os.path.join(BASE_DIR, f"neurowatch_history_{pid}.json")
    data = read_json(path, [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

_live_pids = {p["id"] for p in PATIENT_REGISTRY if p.get("live")}
def is_live(pid): return pid in _live_pids

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
    cmap  = {"Normal":APPLE_GREEN,"Pre-Seizure":APPLE_AMBER,"Seizure":BLOOD_RED}
    valid = []
    mjs   = ""
    for pid in sorted_pids:
        p   = all_patients.get(pid, {})
        lat = p.get("gps_lat","")
        lng = p.get("gps_lng","")
        if not lat or not lng:
            continue
        try:
            lat, lng = float(lat), float(lng)
        except (ValueError, TypeError):
            continue
        state = p.get("state","Normal")
        col   = cmap.get(state, APPLE_GREEN)
        name  = p.get("patient_name","—").replace("'","\\'")
        ward  = p.get("ward","—")
        bed   = p.get("bed","—")
        conf  = int(p.get("confidence",0)*100)
        live  = "LIVE" if is_live(pid) else "DEMO"
        valid.append((lat,lng))
        mjs += f"""
        L.marker([{lat},{lng}],{{icon:L.divIcon({{
          html:'<div style="width:15px;height:15px;background:{col};border:2px solid #fff;border-radius:50%;box-shadow:0 0 10px {col};"></div>',
          iconSize:[15,15],iconAnchor:[7,7],className:''
        }})}}).addTo(map)
        .bindPopup('<b style="color:#111">{name}</b><br><span style="color:#555">{ward}·{bed}</span><br><b style="color:{col}">{state}({conf}%)</b><br><small>{live}</small>');
        L.circle([{lat},{lng}],{{color:'{col}',fillColor:'{col}',fillOpacity:0.08,radius:12,weight:1}}).addTo(map);
        """
    if not valid:
        return None
    clat = sum(v[0] for v in valid)/len(valid)
    clng = sum(v[1] for v in valid)/len(valid)
    return f"""<!DOCTYPE html><html><head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}body{{background:#000;}}
  #map{{width:100%;height:420px;border-radius:10px;}}
  .legend{{position:absolute;bottom:20px;right:10px;z-index:999;
    background:rgba(28,28,30,.9);border:0.5px solid {APPLE_BORDER};
    border-radius:10px;padding:10px 14px;font-size:11px;color:{APPLE_GRAY};
    font-family:-apple-system,monospace;}}
  .li{{display:flex;align-items:center;gap:8px;margin-bottom:4px;}}
  .dot{{width:9px;height:9px;border-radius:50%;flex-shrink:0;}}
</style></head><body>
<div id="map"></div>
<div class="legend">
  <div class="li"><div class="dot" style="background:{APPLE_GREEN}"></div>Normal</div>
  <div class="li"><div class="dot" style="background:{APPLE_AMBER}"></div>Pre-Seizure</div>
  <div class="li"><div class="dot" style="background:{BLOOD_RED}"></div>Seizure</div>
</div>
<script>
  var map=L.map('map',{{zoomControl:true,attributionControl:false}}).setView([{clat},{clng}],18);
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
    {{maxZoom:19,subdomains:'abcd'}}).addTo(map);
  {mjs}
</script></body></html>"""

# =============================================================================
# SESSION STATE
# =============================================================================
for k, v in [
    ("notifications",[]), ("sms_last_states",{}), ("selected_pid",None),
    ("session_start",time.time()), ("session_seizures",0),
    ("state_entry_times",{}), ("sound_enabled",True),
]:
    if k not in st.session_state:
        st.session_state[k] = v

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
# READ DATA
# =============================================================================
all_patients = get_all_patients()
metrics      = get_metrics()
status       = get_status()

sorted_pids = sorted(
    all_patients.keys(),
    key=lambda pid: (
        0 if is_live(pid) else 1,
        -PRIORITY.get(all_patients[pid].get("state","Normal"),0),
    ),
)
if st.session_state.selected_pid is None and sorted_pids:
    st.session_state.selected_pid = sorted_pids[0]

# =============================================================================
# NOTIFICATION / SEIZURE COUNTER
# =============================================================================
for pid, pdata in all_patients.items():
    if not is_live(pid): continue
    new_state = pdata.get("state","Normal")
    prev      = st.session_state.sms_last_states.get(pid)
    if new_state == prev: continue
    st.session_state.sms_last_states[pid]   = new_state
    st.session_state.state_entry_times[pid] = datetime.now()
    if new_state == "Seizure":
        st.session_state.session_seizures += 1
    st.session_state.notifications.insert(0, {
        "state":   new_state,
        "patient": pdata.get("patient_name","—"),
        "ward":    pdata.get("ward","—"),
        "bed":     pdata.get("bed","—"),
        "conf":    int(pdata.get("confidence",0)*100),
        "ts":      datetime.now().strftime("%H:%M:%S"),
        "sms":     pdata.get("sms_sent",False),
        "gps_lat": pdata.get("gps_lat",""),
        "gps_lng": pdata.get("gps_lng",""),
        "pid":     pid,
    })
st.session_state.notifications = st.session_state.notifications[:20]

# =============================================================================
# ALERT SOUND
# =============================================================================
if st.session_state.sound_enabled and any(
        all_patients[pid].get("state")=="Seizure"
        for pid in all_patients if is_live(pid)):
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

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title"> NeuroWatch</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="live-badge"><span class="live-dot"></span> Live Monitoring</div>',
                unsafe_allow_html=True)
    st.caption(f"Refresh every {REFRESH_INTERVAL}s · {datetime.now().strftime('%H:%M:%S')}")

    uptime_s = time.time() - st.session_state.session_start
    st.markdown(f"""<div class="uptime-box">
        Uptime <span style="color:{APPLE_BLUE};font-variant-numeric:tabular-nums">
        {format_uptime(uptime_s)}</span>&nbsp;·&nbsp;
        Seizures <span style="color:{BLOOD_RED};font-weight:700">
        {st.session_state.session_seizures}</span>
    </div>""", unsafe_allow_html=True)

    sc1, sc2 = st.columns([3,1])
    with sc1: st.caption(" Alert sound")
    with sc2:
        st.session_state.sound_enabled = st.toggle(
            "", value=st.session_state.sound_enabled,
            key="sound_toggle", label_visibility="collapsed")

    st.markdown("---")

    # Pi status
    phase = status.get("phase","WAITING")
    msg   = status.get("message","")
    pc = {"DETECTING":APPLE_GREEN,"TRAINING":APPLE_BLUE,
          "PROCESSING":APPLE_AMBER,"LOADING":APPLE_AMBER,
          "ERROR":BLOOD_RED,"STOPPED":BLOOD_RED}.get(phase, APPLE_GRAY)
    st.markdown(f"""<div class="glass-panel" style="padding:10px 14px;margin-bottom:8px;">
      <span style="font-size:10px;font-weight:700;
          background:rgba(255,255,255,0.06);color:{pc};
          padding:2px 8px;border-radius:6px;letter-spacing:.06em;">{phase}</span>
      <div style="font-size:11px;color:{APPLE_AMBER};margin-top:6px;font-weight:500">{msg}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Patients</div>', unsafe_allow_html=True)
    for pid in sorted_pids:
        pdata  = all_patients[pid]
        state  = pdata.get("state","Normal")
        col    = state_color(state)
        icon   = CLASS_ICONS[state]
        conf   = int(pdata.get("confidence",0)*100)
        name   = pdata.get("patient_name","—")
        ward   = pdata.get("ward","—")
        bed    = pdata.get("bed","—")
        is_sel = pid == st.session_state.selected_pid
        lt     = ('<span class="badge-live">LIVE</span>'
                  if is_live(pid) else '<span class="badge-demo">DEMO</span>')
        et     = st.session_state.state_entry_times.get(pid)
        timer  = (f'<div class="state-timer">In {state} for {(datetime.now()-et).seconds}s</div>'
                  if et else "")
        blink  = "animation:seizure-blink 1s ease-in-out infinite;" if state=="Seizure" else ""

        st.markdown(f"""
        <div style="background:{'rgba(10,132,255,.08)' if is_sel else 'rgba(44,44,46,.5)'};
             border:0.5px solid {'rgba(10,132,255,.45)' if is_sel else 'rgba(255,255,255,.08)'};
             border-left:3px solid {col};border-radius:10px;
             padding:10px 13px;margin-bottom:4px;{blink}">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:13px;font-weight:600;color:{APPLE_LABEL}">
              {icon} {name}{lt}
            </span>
            <span style="font-size:10px;color:{col};font-weight:700;
                font-variant-numeric:tabular-nums">{conf}%</span>
          </div>
          <div style="font-size:11px;color:{APPLE_LABEL2};margin-top:2px">{ward} · {bed}</div>
          {timer}
        </div>""", unsafe_allow_html=True)

        if st.button(f"View {name}", key=f"sel_{pid}", use_container_width=True):
            st.session_state.selected_pid = pid
            st.rerun()

    # Alerts
    if st.session_state.notifications:
        st.markdown('<div class="section-label">Alerts</div>', unsafe_allow_html=True)
        for n in st.session_state.notifications[:8]:
            s   = n["state"]
            css = ("notif-seizure" if s=="Seizure"
                   else "notif-pre" if s=="Pre-Seizure" else "notif-normal")
            ico = "⚡" if s=="Seizure" else "⚠️" if s=="Pre-Seizure" else "✅"
            sms = f' <span style="font-size:10px;color:{APPLE_GREEN}">(SMS ✓)</span>' if n.get("sms") else ""
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

# =============================================================================
# ALERT BANNERS (main area)
# =============================================================================
for p in (p for pid,p in all_patients.items()
          if p.get("state")=="Seizure" and is_live(pid)):
    sms = " — Doctor notified via SMS ✓" if p.get("sms_sent") else ""
    st.markdown(
        f'<div class="alert-seizure">⚡ ACTIVE SEIZURE — {p["patient_name"]}'
        f'{sms} — Immediate attention required</div>',
        unsafe_allow_html=True)

for p in (p for pid,p in all_patients.items()
          if p.get("state")=="Pre-Seizure" and is_live(pid)):
    st.markdown(
        f'<div class="alert-pre">⚠️ Pre-Seizure Warning — {p["patient_name"]}'
        f' — Administer medication</div>',
        unsafe_allow_html=True)

# =============================================================================
# METRIC CARDS
# =============================================================================
n_total   = len(all_patients)
n_live    = sum(1 for pid in all_patients if is_live(pid))
n_demo    = n_total - n_live
n_normal  = sum(1 for p in all_patients.values() if p.get("state")=="Normal")
n_pre     = sum(1 for p in all_patients.values() if p.get("state")=="Pre-Seizure")
n_seizure = sum(1 for pid,p in all_patients.items()
                if p.get("state")=="Seizure" and is_live(pid))

c1,c2,c3,c4,c5 = st.columns(5)

def _card(col, label, val, color, sub):
    col.markdown(f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{val}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

_card(c1,"Patients",     n_total,   APPLE_BLUE,  f"{n_live} live · {n_demo} demo")
_card(c2,"Normal",       n_normal,  APPLE_GREEN, "No alerts")
_card(c3,"Pre-Seizure",  n_pre,     APPLE_AMBER, "Warning state")

if n_seizure > 0:
    c4.markdown(f"""<div class="metric-card"
        style="animation:seizure-blink 1s ease-in-out infinite;
               border-left:3px solid {BLOOD_RED};">
        <div class="metric-label">Active Seizure</div>
        <div class="metric-value" style="color:{BLOOD_HOT}">{n_seizure}</div>
        <div class="metric-sub" style="color:{BLOOD_RED}">⚡ Live patient</div>
    </div>""", unsafe_allow_html=True)
else:
    _card(c4,"Active Seizure", 0, APPLE_GRAY, "None active")

_card(c5,"Session Total",
      st.session_state.session_seizures, BLOOD_RED,
      f"{format_uptime(time.time()-st.session_state.session_start)} uptime")

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# TABS
# =============================================================================
tab_overview, tab_patient, tab_map, tab_log, tab_metrics_tab = st.tabs([
    "  Ward Overview",
    "  Patient Detail",
    "  Ward Map",
    "  Event Log",
    "  Model Metrics",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — WARD OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:
    if not all_patients:
        st.info(" Waiting for main_pi.py to start detection…")
    else:
        live_pids = [pid for pid in sorted_pids if is_live(pid)]
        demo_pids = [pid for pid in sorted_pids if not is_live(pid)]

        if live_pids:
            st.markdown('<div class="section-label">Live Patients</div>',
                        unsafe_allow_html=True)
            cols = st.columns(min(len(live_pids),3))
            for i, pid in enumerate(live_pids[:3]):
                p     = all_patients[pid]
                state = p.get("state","Normal")
                col   = state_color(state)
                conf  = int(p.get("confidence",0)*100)
                blink = "animation:seizure-blink 1s ease-in-out infinite;" if state=="Seizure" else ""
                with cols[i]:
                    st.markdown(f"""
                    <div class="glass-panel" style="border-left:4px solid {col};{blink}">
                      <div style="font-size:13px;font-weight:700;
                                  color:{APPLE_LABEL};margin-bottom:4px;">
                        {CLASS_ICONS[state]} {p.get('patient_name','—')}
                        <span class="badge-live">LIVE</span>
                      </div>
                      <div style="font-size:11px;color:{APPLE_LABEL2};margin-bottom:8px;">
                        {p.get('ward','—')} · {p.get('bed','—')}
                      </div>
                      <div style="font-size:24px;font-weight:700;color:{col};
                          letter-spacing:-.5px;">{state}</div>
                      <div style="font-size:11px;color:{APPLE_LABEL3};margin-top:4px;">
                        Confidence: <span style="color:{col};font-weight:600">{conf}%</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

        if demo_pids:
            st.markdown('<div class="section-label">Demo Patients</div>',
                        unsafe_allow_html=True)
            cols = st.columns(min(len(demo_pids),4))
            for i, pid in enumerate(demo_pids):
                p     = all_patients[pid]
                state = p.get("state","Normal")
                col   = state_color(state)
                conf  = int(p.get("confidence",0)*100)
                with cols[i%4]:
                    st.markdown(f"""
                    <div class="glass-panel" style="border-left:3px solid {col};opacity:.72;">
                      <div style="font-size:12px;font-weight:600;color:{APPLE_LABEL}">
                        {CLASS_ICONS[state]} {p.get('patient_name','—')}
                        <span class="badge-demo">DEMO</span>
                      </div>
                      <div style="font-size:10px;color:{APPLE_LABEL3};">
                        {p.get('ward','—')} · {p.get('bed','—')}
                      </div>
                      <div style="font-size:15px;font-weight:700;color:{col};margin-top:5px;">
                        {state} <span style="font-size:10px;opacity:.6">({conf}%)</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PATIENT DETAIL
# ════════════════════════════════════════════════════════════════════════════
with tab_patient:
    if not sorted_pids:
        st.info("⏳ No patient data yet.")
    else:
        pid   = st.session_state.selected_pid or sorted_pids[0]
        pdata = all_patients.get(pid,{})
        state = pdata.get("state","Normal")
        col   = state_color(state)
        conf  = int(pdata.get("confidence",0)*100)
        blink = "animation:seizure-blink 1s ease-in-out infinite;" if state=="Seizure" else ""

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
        if history.empty:
            st.info("No EEG history recorded yet.")
        else:
            tail = history.tail(300)
            state_num = tail["state"].map({"Normal":0,"Pre-Seizure":1,"Seizure":2})
            cseq = [APPLE_GREEN if s=="Normal" else APPLE_AMBER if s=="Pre-Seizure"
                    else BLOOD_RED for s in tail["state"]]
            fig_s = go.Figure(go.Scatter(
                x=tail["timestamp"], y=state_num,
                mode="lines+markers",
                marker=dict(color=cseq, size=4),
                line=dict(color="rgba(255,255,255,.07)",width=1),
                hovertemplate="%{text}<extra></extra>", text=tail["state"],
            ))
            fig_s.update_layout(**_PL, height=175,
                                title=dict(text="EEG State Timeline",
                                           font=dict(size=12,color=APPLE_LABEL2)),
                                yaxis=dict(tickvals=[0,1,2],
                                           ticktext=["Normal","Pre-Sz","Seizure"],**_AX),
                                xaxis=dict(**_AX))
            st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar":False})

            band_cols = [c for c in ["delta","theta","alpha","beta"] if c in history.columns]
            if band_cols:
                t200 = history.tail(200)
                fig_b = go.Figure()
                bc = {"delta":APPLE_BLUE,"theta":APPLE_AMBER,
                      "alpha":APPLE_GREEN,"beta":BLOOD_RED}
                for b in band_cols:
                    fig_b.add_trace(go.Scatter(
                        x=t200["timestamp"], y=t200[b], name=b.capitalize(),
                        mode="lines", line=dict(color=bc.get(b,"#fff"),width=1.5)))
                fig_b.update_layout(**_PL, height=210,
                                    title=dict(text="EEG Band Power",
                                               font=dict(size=12,color=APPLE_LABEL2)),
                                    xaxis=dict(**_AX), yaxis=dict(**_AX))
                st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar":False})

            fig_c = go.Figure(go.Scatter(
                x=history.tail(200)["timestamp"],
                y=history.tail(200)["confidence"]*100,
                mode="lines", fill="tozeroy",
                line=dict(color=APPLE_BLUE,width=1.5),
                fillcolor="rgba(10,132,255,0.07)",
            ))
            fig_c.update_layout(**_PL, height=155,
                                title=dict(text="Classifier Confidence (%)",
                                           font=dict(size=12,color=APPLE_LABEL2)),
                                xaxis=dict(**_AX),
                                yaxis=dict(range=[0,105],**_AX))
            st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar":False})

            counts = history["state"].value_counts()
            fig_p  = go.Figure(go.Pie(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                marker=dict(colors=[
                    {"Normal":APPLE_GREEN,"Pre-Seizure":APPLE_AMBER,
                     "Seizure":BLOOD_RED}.get(l,APPLE_GRAY)
                    for l in counts.index]),
                hole=0.58,
                textfont=dict(size=11,color=APPLE_LABEL),
            ))
            fig_p.update_layout(**_PL, height=230)
            st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar":False})

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — WARD MAP
# ════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.markdown('<div class="section-label">All Patients Map</div>',
                unsafe_allow_html=True)
    map_html = make_all_patients_map(all_patients, sorted_pids)
    if map_html:
        components.html(map_html, height=460, scrolling=False)
        st.caption("Tap a marker for patient details. 🔴 Seizure · 🟡 Pre-Seizure · 🟢 Normal")
    else:
        st.info("No GPS coordinates found. Add gps_lat / gps_lng to patient data.")

    st.markdown('<div class="section-label">Individual Patient</div>',
                unsafe_allow_html=True)
    opts = {all_patients[p].get("patient_name","—")+(" 🔴" if is_live(p) else " ⬜"):p
            for p in sorted_pids
            if all_patients[p].get("gps_lat") and all_patients[p].get("gps_lng")}
    if opts:
        sel = st.selectbox("", list(opts.keys()), label_visibility="collapsed")
        mp  = all_patients[opts[sel]]
        mc  = state_color(mp.get("state","Normal"))
        st.markdown(f"""
        <div class="glass-panel" style="border-left:4px solid {mc};
             padding:10px 16px;margin-bottom:8px;">
          {CLASS_ICONS[mp.get('state','Normal')]}
          <b style="color:{APPLE_LABEL}">{mp.get('patient_name','—')}</b> ·
          {mp.get('ward','—')} · {mp.get('bed','—')} ·
          <span style="color:{mc}">{mp.get('state','Normal')}</span>
        </div>""", unsafe_allow_html=True)
        components.html(
            make_osm_embed(float(mp["gps_lat"]),float(mp["gps_lng"]),
                           mp.get("patient_name","—"),zoom=18),
            height=360, scrolling=False)
    else:
        st.info("No GPS coordinates configured.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVENT LOG
# ════════════════════════════════════════════════════════════════════════════
with tab_log:
    opts = {all_patients[p].get("patient_name","—")+(" 🔴" if is_live(p) else " ⬜"):p
            for p in sorted_pids}
    sel  = st.selectbox("", list(opts.keys()), label_visibility="collapsed")
    lpid = opts.get(sel, sorted_pids[0] if sorted_pids else None)

    if lpid:
        history = get_patient_history(lpid)
        if history.empty:
            st.info("No history yet for this patient.")
        else:
            if not is_live(lpid):
                st.caption(" Demo patient — simulated data")
            f1,f2,f3 = st.columns(3)
            with f1: sf = st.multiselect("State", CLASSES, default=CLASSES,
                                         label_visibility="collapsed")
            with f2: nr = st.slider("Rows",10,200,50,label_visibility="collapsed")
            with f3: so = st.checkbox("Seizures only",value=False)
            if so: sf = ["Seizure"]

            disp  = history[history["state"].isin(sf)].tail(nr).reset_index(drop=True)
            bands = [c for c in ["delta","theta","alpha","beta"] if c in history.columns]
            scols = (["timestamp","state","confidence"]+bands+["sms_sent"]
                     if is_live(lpid) else ["timestamp","state","confidence","sms_sent"])
            show  = disp[[c for c in scols if c in disp.columns]].copy()
            show["timestamp"]  = show["timestamp"].dt.strftime("%H:%M:%S")
            show["confidence"] = (show["confidence"]*100).round(1).astype(str)+"%"
            st.dataframe(show, use_container_width=True, height=360)

            sz = len(history[history["state"]=="Seizure"])
            ps = len(history[history["state"]=="Pre-Seizure"])
            st.caption(f"Records: {len(history)} · Seizure: {sz} · "
                       f"Pre-Seizure: {ps} · Rate: {sz/len(history)*100:.1f}%")
            st.download_button(
                "⬇ Download CSV",
                data=history.to_csv(index=False).encode("utf-8"),
                file_name=f"neurowatch_{lpid}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv")

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL METRICS
# ════════════════════════════════════════════════════════════════════════════
with tab_metrics_tab:
    if not metrics:
        st.info("⏳ Metrics available after first training cycle.")
    else:
        ts = metrics.get("timestamp","")
        try:  ts = datetime.fromisoformat(ts).strftime("%d %b %Y  %H:%M")
        except Exception: pass
        st.caption(f"Trained: {ts}  ·  Bonn University Dataset · SVM / Random Forest")

        ks  = ["accuracy","precision","recall","f1","specificity","seizure_recall"]
        lbs = ["Accuracy","Precision","Recall","F1 Score","Specificity","Seiz. Recall"]
        for col, label, key in zip(st.columns(6), lbs, ks):
            v = metrics.get(key,0)
            c = APPLE_GREEN if v>=.9 else APPLE_AMBER if v>=.7 else BLOOD_RED
            col.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{c}">{v*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        vals   = [metrics.get(k,0) for k in ks]
        colors = [APPLE_GREEN if v>=.9 else APPLE_AMBER if v>=.7 else BLOOD_RED for v in vals]
        fig_m  = go.Figure(go.Bar(
            x=vals, y=lbs, orientation="h", marker_color=colors,
            text=[f"{v*100:.1f}%" for v in vals], textposition="outside",
            textfont=dict(size=11,color=APPLE_LABEL2)))
        fig_m.update_layout(**_PL, height=250, showlegend=False,
                            xaxis=dict(range=[0,1.2],**_AX), yaxis=dict(**_AX))
        st.plotly_chart(fig_m, use_container_width=True, config={"displayModeBar":False})

        cm = metrics.get("confusion_matrix")
        if cm:
            st.markdown('<div class="section-label">Confusion Matrix</div>',
                        unsafe_allow_html=True)
            ca  = np.array(cm)
            lb  = CLASSES[:ca.shape[0]]
            fig_cm = go.Figure(go.Heatmap(
                z=ca, x=lb, y=lb,
                colorscale=[[0,APPLE_BG2],[1,APPLE_BLUE]],
                text=ca, texttemplate="%{text}",
                textfont=dict(size=14,color=APPLE_LABEL), showscale=False))
            fig_cm.update_layout(**_PL, height=290,
                                 xaxis=dict(title="Predicted",side="bottom",**_AX),
                                 yaxis=dict(title="True",autorange="reversed",**_AX))
            st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar":False})

# =============================================================================
# AUTO-REFRESH
# =============================================================================
time.sleep(REFRESH_INTERVAL)
st.rerun()