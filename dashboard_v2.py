# =============================================================================
# dashboard.py  —  NeuroWatch Multi-Patient Streamlit Dashboard  (v3)
#
# Run:  streamlit run dashboard.py
# Reads live JSON files written by main_pi.py every few seconds.
#
# Improvements over v2:
#   1. Embedded Google Maps (iframe) inline — no redirect needed
#   2. Interactive Ward Map tab — all patients pinned on one map
#   3. Uptime clock + session seizure counter in sidebar
#   4. Sparkline mini-trend in each patient card
#   5. EEG waveform simulation overlay on patient detail
#   6. Alert sound toggle (HTML5 audio pulse on seizure)
#   7. Per-patient "Time in state" timer
#   8. Dark-themed OpenStreetMap fallback (no API key needed)
# =============================================================================

import os, json, time
from datetime import datetime, timedelta
import numpy as np
try:
    from PIL import Image as _PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

_PHOTO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patient_photos")

def _load_patient_photo_pil(pid):
    """Returns a PIL Image (110x110, circular crop) or None."""
    if not _PIL_OK:
        return None
    for ext in ("jpg", "jpeg", "png", "webp"):
        path = os.path.join(_PHOTO_DIR, f"{pid}.{ext}")
        if os.path.isfile(path):
            try:
                from PIL import ImageDraw
                img = _PILImage.open(path).convert("RGBA")
                img = img.resize((110, 110), _PILImage.LANCZOS)
                mask = _PILImage.new("L", (110, 110), 0)
                ImageDraw.Draw(mask).ellipse([0, 0, 109, 109], fill=255)
                img.putalpha(mask)
                return img
            except Exception:
                pass
    return None
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

try:
    from sms_notifier import send_sms_alert_async
    SMS_AVAILABLE = True
except ImportError:
    SMS_AVAILABLE = False

try:
    from patients import PATIENT_REGISTRY
except ImportError:
    PATIENT_REGISTRY = []

# =============================================================================
# CONFIG
# =============================================================================
# =============================================================================
# CONFIG - UNIVERSAL PATHS
# =============================================================================
# This finds the folder where your script is saved
# This finds the EXACT folder where your dashboard.py is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# These MUST match the filenames in your main_pi_bios_v15.py script
PATIENTS_JSON    = os.path.join(BASE_DIR, "neurowatch_patients.json")
METRICS_JSON     = os.path.join(BASE_DIR, "neurowatch_metrics.json")
STATUS_JSON      = os.path.join(BASE_DIR, "neurowatch_status.json")

# Debugging: Uncomment the line below to see EXACTLY where it's looking in your terminal
# print(f"DEBUG: Looking for status file at: {STATUS_JSON}")

REFRESH_INTERVAL = 3

CLASSES      = ["Normal", "Pre-Seizure", "Seizure"]
CLASS_COLORS = {"Normal":"#00e88f","Pre-Seizure":"#ffc93c","Seizure":"#ff3b5c"}
CLASS_ICONS  = {"Normal":"🟢","Pre-Seizure":"🟡","Seizure":"🔴"}
PRIORITY     = {"Seizure":2,"Pre-Seizure":1,"Normal":0}

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="NeuroWatch — Multi-Patient Monitor",
    page_icon="🧠", layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background-color:#080e1a;}
section[data-testid="stSidebar"]{background-color:#0d1525!important;border-right:1px solid #1e2e48;}
.sidebar-title{font-family:'Space Mono',monospace;font-size:17px;color:#00d4ff;
  letter-spacing:.08em;padding-bottom:10px;border-bottom:1px solid #1e2e48;margin-bottom:14px;}
.live-badge{display:inline-flex;align-items:center;gap:8px;
  background:rgba(0,232,143,.1);border:1px solid rgba(0,232,143,.3);
  border-radius:20px;padding:4px 14px;
  font-family:'Space Mono',monospace;font-size:11px;color:#00e88f;}
.live-dot{width:8px;height:8px;border-radius:50%;background:#00e88f;
  animation:blink 1.2s ease-in-out infinite;display:inline-block;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.badge-live{display:inline-block;font-family:'Space Mono',monospace;font-size:9px;
  background:rgba(255,59,92,.2);color:#ff3b5c;border:1px solid rgba(255,59,92,.5);
  padding:1px 7px;border-radius:4px;letter-spacing:.06em;vertical-align:middle;}
.badge-demo{display:inline-block;font-family:'Space Mono',monospace;font-size:9px;
  background:rgba(74,96,128,.2);color:#4a6080;border:1px solid #1e2e48;
  padding:1px 7px;border-radius:4px;letter-spacing:.06em;vertical-align:middle;}
.metric-card{background:#111d30;border:1px solid #1e2e48;
  border-radius:14px;padding:14px 16px;text-align:center;}
.metric-label{font-family:'Space Mono',monospace;font-size:10px;
  letter-spacing:.12em;color:#4a6080;text-transform:uppercase;margin-bottom:5px;}
.metric-value{font-family:'Space Mono',monospace;font-size:28px;font-weight:700;line-height:1;}
.metric-sub{font-size:11px;color:#4a6080;margin-top:4px;}
.alert-seizure{background:rgba(255,59,92,.12);border:1px solid rgba(255,59,92,.5);
  border-radius:12px;padding:12px 18px;font-size:14px;font-weight:600;
  color:#ff3b5c;margin-bottom:10px;animation:pulse-border 1.5s ease-in-out infinite;}
@keyframes pulse-border{0%,100%{border-color:rgba(255,59,92,.5)}50%{border-color:rgba(255,59,92,1)}}
.alert-pre{background:rgba(255,201,60,.1);border:1px solid rgba(255,201,60,.4);
  border-radius:12px;padding:12px 18px;font-size:14px;font-weight:600;
  color:#ffc93c;margin-bottom:10px;}
.section-label{font-family:'Space Mono',monospace;font-size:10px;
  letter-spacing:.14em;color:#4a6080;text-transform:uppercase;
  margin:16px 0 8px;border-bottom:1px solid #1e2e48;padding-bottom:5px;}
.patient-card{background:#111d30;border:1px solid #1e2e48;
  border-radius:12px;padding:14px 16px;margin-bottom:8px;}
.notif-seizure{background:rgba(255,59,92,.12);border:1px solid rgba(255,59,92,.4);
  border-radius:10px;padding:10px 14px;margin-bottom:6px;font-size:12px;color:#ff3b5c;}
.notif-pre{background:rgba(255,201,60,.1);border:1px solid rgba(255,201,60,.35);
  border-radius:10px;padding:10px 14px;margin-bottom:6px;font-size:12px;color:#ffc93c;}
.notif-normal{background:rgba(0,232,143,.08);border:1px solid rgba(0,232,143,.3);
  border-radius:10px;padding:10px 14px;margin-bottom:6px;font-size:12px;color:#00e88f;}
.map-container{border-radius:12px;overflow:hidden;border:1px solid #1e2e48;margin-top:8px;}
.uptime-box{background:#0d1a2e;border:1px solid #1e2e48;border-radius:8px;
  padding:8px 12px;font-family:'Space Mono',monospace;font-size:11px;color:#4a6080;
  margin-bottom:10px;}
.uptime-box span{color:#00d4ff;}
.state-timer{font-family:'Space Mono',monospace;font-size:10px;color:#4a6080;margin-top:3px;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# JSON READERS
# =============================================================================
def read_json(path, default=None):
    try:
        with open(path,"r") as f: return json.load(f)
    except Exception: return default

def get_all_patients():
    return read_json(PATIENTS_JSON, {})

def get_patient_history_path(pid):
    return os.path.join(BASE_DIR, f"neurowatch_history_{pid}.json")

def get_patient_history(pid):
    path = get_patient_history_path(pid)
    data = read_json(path, [])
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def get_metrics():
    return read_json(METRICS_JSON, {})

def get_status():
    return read_json(STATUS_JSON, {"phase":"WAITING","message":"Waiting for Pi..."})

_live_pids = {p["id"] for p in PATIENT_REGISTRY if p.get("live")}

def is_live(pid):
    return pid in _live_pids

# =============================================================================
# HELPERS
# =============================================================================
def format_uptime(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def time_in_state(history_df, state):
    """Estimate how long patient has been in current state continuously."""
    if history_df.empty: return "—"
    recent = history_df.tail(200)
    # Count from end going backwards while state matches
    count = 0
    for s in reversed(recent["state"].tolist()):
        if s == state: count += 1
        else: break
    seconds = count * 0.12  # STEP_SLEEP estimate
    if seconds < 60: return f"{int(seconds)}s"
    return f"{int(seconds//60)}m {int(seconds%60)}s"

def make_osm_embed(lat, lng, label="Patient Location", zoom=17):
    """
    Returns an HTML string with an embedded OpenStreetMap (no API key needed).
    Dark-styled via CartoDB dark_matter tiles.
    Includes a marker and a 'Open in Google Maps' button.
    """
    google_url = f"https://maps.google.com/?q={lat},{lng}"
    osm_url    = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lng}#map={zoom}/{lat}/{lng}"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
      <style>
        * {{margin:0;padding:0;box-sizing:border-box;}}
        body {{background:#080e1a;font-family:'DM Sans',sans-serif;}}
        #map {{width:100%;height:320px;border-radius:10px;}}
        .map-footer {{
          background:#0d1525;padding:8px 14px;
          display:flex;justify-content:space-between;align-items:center;
          border-top:1px solid #1e2e48;
        }}
        .coord-text {{font-family:monospace;font-size:11px;color:#4a6080;}}
        .open-btn {{
          background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.4);
          color:#00d4ff;padding:5px 14px;border-radius:6px;font-size:12px;
          cursor:pointer;text-decoration:none;
        }}
        .open-btn:hover{{background:rgba(0,212,255,.2);}}
      </style>
    </head>
    <body>
      <div id="map"></div>
      <div class="map-footer">
        <span class="coord-text">📍 {lat:.4f}, {lng:.4f} — {label}</span>
        <a class="open-btn" href="{google_url}" target="_blank">Open in Google Maps ↗</a>
      </div>
      <script>
        var map = L.map('map', {{zoomControl:true, attributionControl:false}})
                   .setView([{lat},{lng}], {zoom});

        // Dark tiles — CartoDB Dark Matter
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
          maxZoom: 19,
          subdomains: 'abcd'
        }}).addTo(map);

        // Custom marker
        var icon = L.divIcon({{
          html: '<div style="width:14px;height:14px;background:#ff3b5c;border:2px solid #fff;border-radius:50%;box-shadow:0 0 8px #ff3b5c;"></div>',
          iconSize: [14,14], iconAnchor:[7,7], className:''
        }});
        L.marker([{lat},{lng}], {{icon:icon}})
         .addTo(map)
         .bindPopup('<b style="color:#111">{label}</b><br><small>{lat:.4f}, {lng:.4f}</small>')
         .openPopup();

        // Pulsing circle
        L.circle([{lat},{lng}], {{
          color:'#ff3b5c', fillColor:'#ff3b5c',
          fillOpacity:0.08, radius:15, weight:1.5
        }}).addTo(map);
      </script>
    </body>
    </html>
    """
    return html

def make_all_patients_map(patients_dict, sorted_pids):
    """
    Renders all patients as coloured markers on one shared dark map.
    """
    color_map = {"Normal":"#00e88f","Pre-Seizure":"#ffc93c","Seizure":"#ff3b5c"}
    markers_js = ""
    valid = []
    for pid in sorted_pids:
        p   = patients_dict[pid]
        lat = p.get("gps_lat","")
        lng = p.get("gps_lng","")
        if not lat or not lng: continue
        state = p.get("state","Normal")
        col   = color_map.get(state,"#00e88f")
        name  = p.get("patient_name","—").replace("'","\\'")
        ward  = p.get("ward","—")
        bed   = p.get("bed","—")
        conf  = int(p.get("confidence",0)*100)
        live  = "LIVE" if is_live(pid) else "DEMO"
        valid.append((lat, lng))
        markers_js += f"""
        L.marker([{lat},{lng}], {{icon: L.divIcon({{
          html: '<div style="width:16px;height:16px;background:{col};border:2px solid #fff;border-radius:50%;box-shadow:0 0 10px {col};"></div>',
          iconSize:[16,16],iconAnchor:[8,8],className:''
        }})}}).addTo(map)
         .bindPopup('<b style="color:#111">{name}</b><br><span style="color:#555">{ward} · {bed}</span><br><b style="color:{col}">{state} ({conf}%)</b><br><small>{live}</small>');
        L.circle([{lat},{lng}], {{color:'{col}',fillColor:'{col}',fillOpacity:0.07,radius:12,weight:1}}).addTo(map);
        """

    if not valid: return None

    center_lat = sum(v[0] for v in valid) / len(valid)
    center_lng = sum(v[1] for v in valid) / len(valid)

    html = f"""
    <!DOCTYPE html><html><head>
    <meta charset="utf-8"/>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
      *{{margin:0;padding:0;box-sizing:border-box;}}
      body{{background:#080e1a;}}
      #map{{width:100%;height:420px;border-radius:10px;}}
      .legend{{
        position:absolute;bottom:20px;right:10px;z-index:999;
        background:rgba(8,14,26,.85);border:1px solid #1e2e48;
        border-radius:8px;padding:10px 14px;font-size:11px;color:#4a6080;
        font-family:monospace;
      }}
      .legend-item{{display:flex;align-items:center;gap:8px;margin-bottom:4px;}}
      .dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0;}}
    </style>
    </head><body>
    <div id="map"></div>
    <div class="legend">
      <div class="legend-item"><div class="dot" style="background:#00e88f"></div>Normal</div>
      <div class="legend-item"><div class="dot" style="background:#ffc93c"></div>Pre-Seizure</div>
      <div class="legend-item"><div class="dot" style="background:#ff3b5c"></div>Seizure</div>
    </div>
    <script>
      var map = L.map('map',{{zoomControl:true,attributionControl:false}})
                 .setView([{center_lat},{center_lng}], 18);
      L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
        {{maxZoom:19,subdomains:'abcd'}}).addTo(map);
      {markers_js}
    </script>
    </body></html>
    """
    return html

# =============================================================================
# SESSION STATE
# =============================================================================
if "notifications"      not in st.session_state: st.session_state.notifications      = []
if "sms_last_states"    not in st.session_state: st.session_state.sms_last_states     = {}
if "selected_pid"       not in st.session_state: st.session_state.selected_pid        = None
if "last_refresh"       not in st.session_state: st.session_state.last_refresh        = 0
if "session_start"      not in st.session_state: st.session_state.session_start       = time.time()
if "session_seizures"   not in st.session_state: st.session_state.session_seizures    = 0
if "state_entry_times"  not in st.session_state: st.session_state.state_entry_times   = {}
if "sound_enabled"      not in st.session_state: st.session_state.sound_enabled       = True

# =============================================================================
# PLOTLY DEFAULTS
# =============================================================================
_PL = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
           font=dict(family="Space Mono, monospace",color="#4a6080",size=10),
           margin=dict(l=10,r=10,t=28,b=10),
           legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10)))
_AX = dict(gridcolor="#1e2e48",zerolinecolor="#1e2e48")

# =============================================================================
# READ DATA
# =============================================================================
all_patients = get_all_patients()
metrics      = get_metrics()
status       = get_status()

sorted_pids  = sorted(
    all_patients.keys(),
    key=lambda pid: (
        0 if is_live(pid) else 1,
        -PRIORITY.get(all_patients[pid].get("state","Normal"),0),
    ),
)

if st.session_state.selected_pid is None and sorted_pids:
    st.session_state.selected_pid = sorted_pids[0]

# =============================================================================
# NOTIFICATION CHECK + SESSION SEIZURE COUNTER
# =============================================================================
for pid, pdata in all_patients.items():
    if not is_live(pid): continue
    new_state = pdata.get("state","Normal")
    prev      = st.session_state.sms_last_states.get(pid)
    if new_state == prev: continue
    st.session_state.sms_last_states[pid] = new_state

    # Track state entry time
    st.session_state.state_entry_times[pid] = datetime.now()

    # Count seizures this session
    if new_state == "Seizure":
        st.session_state.session_seizures += 1

    ts   = datetime.now().strftime("%H:%M:%S")
    conf = int(pdata.get("confidence",0)*100)
    st.session_state.notifications.insert(0, {
        "state":   new_state,
        "patient": pdata.get("patient_name","—"),
        "ward":    pdata.get("ward","—"),
        "bed":     pdata.get("bed","—"),
        "conf":    conf,
        "ts":      ts,
        "sms":     pdata.get("sms_sent",False),
        "gps_lat": pdata.get("gps_lat",""),
        "gps_lng": pdata.get("gps_lng",""),
        "pid":     pid,
        "is_live": True,
    })

st.session_state.notifications = st.session_state.notifications[:20]

# =============================================================================
# ALERT SOUND — inject HTML5 audio ping on active seizure (live only)
# =============================================================================
live_seizure_active = any(
    all_patients[pid].get("state") == "Seizure"
    for pid in all_patients if is_live(pid)
)

if live_seizure_active and st.session_state.sound_enabled:
    components.html("""
    <audio autoplay>
      <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAA..." type="audio/wav">
    </audio>
    <script>
      // Gentle beep using Web Audio API
      (function() {
        try {
          var ctx = new (window.AudioContext || window.webkitAudioContext)();
          function beep(freq, dur, vol) {
            var o = ctx.createOscillator(), g = ctx.createGain();
            o.connect(g); g.connect(ctx.destination);
            o.frequency.value = freq; g.gain.value = vol;
            o.start(ctx.currentTime); o.stop(ctx.currentTime + dur);
            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + dur);
          }
          beep(880, 0.15, 0.15);
          setTimeout(function(){ beep(880, 0.15, 0.12); }, 250);
        } catch(e) {}
      })();
    </script>
    """, height=0)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">🧠 NEUROWATCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="live-badge"><span class="live-dot"></span> LIVE MONITORING</div>',
                unsafe_allow_html=True)
    st.caption(f"Auto-refresh every {REFRESH_INTERVAL}s · {datetime.now().strftime('%H:%M:%S')}")

    # ── Uptime + session stats ────────────────────────────────────────────────
    uptime_s = time.time() - st.session_state.session_start
    st.markdown(f"""
    <div class="uptime-box">
      Uptime <span>{format_uptime(uptime_s)}</span> &nbsp;·&nbsp;
      Seizures this session <span style="color:#ff3b5c">{st.session_state.session_seizures}</span>
    </div>""", unsafe_allow_html=True)

    # ── Sound toggle ──────────────────────────────────────────────────────────
    sound_col1, sound_col2 = st.columns([3,1])
    with sound_col1:
        st.caption("🔔 Alert sound")
    with sound_col2:
        st.session_state.sound_enabled = st.toggle("", value=st.session_state.sound_enabled,
                                                    key="sound_toggle", label_visibility="collapsed")

    st.markdown("---")

    # ── Pi status ─────────────────────────────────────────────────────────────
    phase = status.get("phase","WAITING")
    msg   = status.get("message","")
    pc    = {"DETECTING":"#00e88f","TRAINING":"#00d4ff","PROCESSING":"#ffc93c",
             "LOADING":"#ffc93c","ERROR":"#ff3b5c","STOPPED":"#ff3b5c"}.get(phase,"#4a6080")
    st.markdown(f"""
    <div style="background:#111d30;border:1px solid #1e2e48;border-radius:10px;
         padding:10px 14px;margin-bottom:12px;">
      <span style="font-family:'Space Mono',monospace;font-size:10px;
           background:{pc}22;color:{pc};padding:2px 8px;border-radius:4px;">{phase}</span>
      <div style="font-size:11px;color:#4a6080;margin-top:4px">{msg}</div>
    </div>""", unsafe_allow_html=True)

    # ── Patient list ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">SELECT PATIENT</div>', unsafe_allow_html=True)
    for pid in sorted_pids:
        pdata  = all_patients[pid]
        state  = pdata.get("state","Normal")
        col    = CLASS_COLORS[state]
        icon   = CLASS_ICONS[state]
        conf   = int(pdata.get("confidence",0)*100)
        name   = pdata.get("patient_name","—")
        ward   = pdata.get("ward","—")
        bed    = pdata.get("bed","—")
        is_sel = pid == st.session_state.selected_pid
        live_tag = '<span class="badge-live">LIVE</span>' if is_live(pid) else '<span class="badge-demo">DEMO</span>'

        # Mini state timer
        entry_t = st.session_state.state_entry_times.get(pid)
        if entry_t:
            elapsed = (datetime.now() - entry_t).seconds
            timer_txt = f"In {state} for {elapsed}s"
        else:
            timer_txt = ""

        st.markdown(f"""
        <div style="background:{'rgba(0,212,255,.06)' if is_sel else '#111d30'};
             border:1px solid {'#00d4ff' if is_sel else '#1e2e48'};
             border-left:4px solid {col};border-radius:10px;
             padding:10px 14px;margin-bottom:4px;">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:13px;font-weight:600;color:#e8f0ff">
              {icon} {name} {live_tag}
            </span>
            <span style="font-family:'Space Mono',monospace;font-size:10px;color:{col}">{conf}%</span>
          </div>
          <div style="font-size:11px;color:#4a6080;margin-top:2px">{ward} · {bed}</div>
          {f'<div class="state-timer">{timer_txt}</div>' if timer_txt else ''}
        </div>""", unsafe_allow_html=True)

        if st.button(f"View {name}", key=f"sel_{pid}", use_container_width=True):
            st.session_state.selected_pid = pid
            st.rerun()

    # ── Live alerts with inline mini-map link ─────────────────────────────────
    if st.session_state.notifications:
        st.markdown('<div class="section-label">LIVE ALERTS</div>', unsafe_allow_html=True)
        for n in st.session_state.notifications[:8]:
            s   = n["state"]
            css = ("notif-seizure" if s=="Seizure" else "notif-pre" if s=="Pre-Seizure" else "notif-normal")
            ico = "⚡" if s=="Seizure" else "⚠️" if s=="Pre-Seizure" else "✅"
            sms_tag = ' <b style="font-size:10px">(SMS ✓)</b>' if n.get("sms") else ""

            # Map link — now clearly labelled
            gps = ""
            if n.get("gps_lat") and n.get("gps_lng"):
                gmaps = f"https://maps.google.com/?q={n['gps_lat']},{n['gps_lng']}"
                osm   = f"https://www.openstreetmap.org/?mlat={n['gps_lat']}&mlon={n['gps_lng']}#map=18/{n['gps_lat']}/{n['gps_lng']}"
                gps   = (f'<br>'
                         f'<a href="{gmaps}" target="_blank" style="font-size:10px;color:#00d4ff;margin-right:10px">📍 Google Maps</a>'
                         f'<a href="{osm}"   target="_blank" style="font-size:10px;color:#4a6080">OSM ↗</a>')
            st.markdown(f"""
            <div class="{css}">
              <b>{ico} {s.upper()}</b>{sms_tag}<br>
              {n['patient']} — {n['ward']}, {n['bed']}<br>
              <span style="font-family:'Space Mono',monospace;font-size:10px;opacity:.7">
                Conf:{n['conf']}% · {n['ts']}
              </span>{gps}
            </div>""", unsafe_allow_html=True)

# =============================================================================
# ACTIVE ALERT BANNERS
# =============================================================================
live_seizure_pts = [p for pid,p in all_patients.items()
                    if p.get("state")=="Seizure" and is_live(pid)]
live_pre_pts     = [p for pid,p in all_patients.items()
                    if p.get("state")=="Pre-Seizure" and is_live(pid)]

if live_seizure_pts:
    for p in live_seizure_pts:
        st.markdown(
            f'<div class="alert-seizure">⚡ ACTIVE SEIZURE — {p["patient_name"]}'
            f'{" — DR NOTIFIED via SMS ✓" if p.get("sms_sent") else ""}'
            f' — Immediate attention required!</div>',
            unsafe_allow_html=True)
if live_pre_pts:
    for p in live_pre_pts:
        st.markdown(
            f'<div class="alert-pre">⚠️ PRE-SEIZURE WARNING — {p["patient_name"]} — Please take medication</div>',
            unsafe_allow_html=True)

# =============================================================================
# TOP SUMMARY METRICS
# =============================================================================
n_total   = len(all_patients)
n_live    = sum(1 for pid in all_patients if is_live(pid))
n_demo    = n_total - n_live
n_normal  = sum(1 for p in all_patients.values() if p.get("state")=="Normal")
n_pre     = sum(1 for p in all_patients.values() if p.get("state")=="Pre-Seizure")
n_seizure = sum(1 for pid,p in all_patients.items()
                if p.get("state")=="Seizure" and is_live(pid))

c1,c2,c3,c4,c5 = st.columns(5)
with c1:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Total Patients</div>
      <div class="metric-value" style="color:#00d4ff">{n_total}</div>
      <div class="metric-sub">{n_live} live · {n_demo} demo</div></div>""",
      unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Normal</div>
      <div class="metric-value" style="color:#00e88f">{n_normal}</div>
      <div class="metric-sub">No alerts</div></div>""",unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Pre-Seizure</div>
      <div class="metric-value" style="color:#ffc93c">{n_pre}</div>
      <div class="metric-sub">Warning state</div></div>""",unsafe_allow_html=True)
with c4:
    brd = "border:1px solid rgba(255,59,92,.5);" if n_seizure>0 else ""
    bg  = "background:rgba(255,59,92,.06);" if n_seizure>0 else ""
    st.markdown(f"""<div class="metric-card" style="{brd}{bg}">
      <div class="metric-label">Active Seizure</div>
      <div class="metric-value" style="color:#ff3b5c">{n_seizure}</div>
      <div class="metric-sub">{'⚡ Live patient!' if n_seizure>0 else 'None'}</div>
    </div>""",unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-label">Session Seizures</div>
      <div class="metric-value" style="color:#ff3b5c">{st.session_state.session_seizures}</div>
      <div class="metric-sub">{format_uptime(time.time()-st.session_state.session_start)} uptime</div>
    </div>""",unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# TABS  — added Ward Map tab
# =============================================================================
tab_overview, tab_patient, tab_map, tab_log, tab_metrics_tab = st.tabs([
    "🏥  Ward Overview",
    "👤  Patient Detail",
    "🗺️  Ward Map",
    "📋  Event Log",
    "📊  Model Metrics",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — WARD OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:

    if not all_patients:
        st.info("⏳ Waiting for main_pi.py to start detection...")
    else:
        # Live patient featured card
        live_pids = [pid for pid in sorted_pids if is_live(pid)]
        if live_pids:
            st.markdown('<div class="section-label">🔴 LIVE MONITORED PATIENT</div>',
                        unsafe_allow_html=True)
            pid    = live_pids[0]
            pdata  = all_patients[pid]
            state  = pdata.get("state","Normal")
            col    = CLASS_COLORS[state]
            icon   = CLASS_ICONS[state]
            conf   = int(pdata.get("confidence",0)*100)
            name   = pdata.get("patient_name","—")
            age    = pdata.get("age","—")
            diag   = pdata.get("diagnosis","—")
            ward   = pdata.get("ward","—")
            bed    = pdata.get("bed","—")
            gps_lat= pdata.get("gps_lat","")
            gps_lng= pdata.get("gps_lng","")
            sms_ok = pdata.get("sms_sent",False)
            sz_extra = "box-shadow:0 0 24px rgba(255,59,92,.25);" if state=="Seizure" else ""
            sms_badge = ('<span style="font-size:11px;color:#00e88f">✓ SMS sent — DR notified</span>'
                         if sms_ok else "")

            # Split: info left, map right
            ov_col1, ov_col2 = st.columns([1, 1])
            with ov_col1:
                st.markdown(f"""
                <div style="background:#111d30;border:2px solid {col};border-left:6px solid {col};
                     border-radius:14px;padding:18px 22px;{sz_extra}height:100%;">
                  <div style="font-size:20px;font-weight:700;color:#e8f0ff;margin-bottom:6px">
                    {icon} {name}
                    <span style="font-family:'Space Mono',monospace;font-size:10px;
                         background:rgba(255,59,92,.2);color:#ff3b5c;border:1px solid rgba(255,59,92,.4);
                         padding:2px 8px;border-radius:4px;margin-left:8px;vertical-align:middle;">LIVE</span>
                  </div>
                  <div style="font-size:12px;color:#4a6080;margin-bottom:4px">
                    {diag} · Age {age} · {ward} · {bed}
                  </div>
                  {f'<div style="font-size:11px;margin-bottom:6px">{sms_badge}</div>' if sms_ok else ''}
                  <div style="display:flex;align-items:center;gap:16px;margin-top:12px">
                    <span style="font-family:'Space Mono',monospace;font-size:13px;
                         background:{col}22;color:{col};padding:4px 14px;border-radius:6px;">
                      {state.upper()}
                    </span>
                    <span style="font-family:'Space Mono',monospace;font-size:26px;
                         font-weight:700;color:{col}">{conf}%</span>
                    <span style="font-size:11px;color:#4a6080">confidence</span>
                  </div>
                </div>""", unsafe_allow_html=True)

            with ov_col2:
                if gps_lat and gps_lng:
                    gps_lbl = pdata.get("gps_label","Patient Location")
                    components.html(
                        make_osm_embed(float(gps_lat), float(gps_lng), gps_lbl, zoom=18),
                        height=370, scrolling=False
                    )
                else:
                    st.markdown("""
                    <div style="background:#111d30;border:1px solid #1e2e48;border-radius:12px;
                         height:340px;display:flex;align-items:center;justify-content:center;
                         color:#4a6080;font-size:13px;">
                      📍 No GPS coordinates configured
                    </div>""", unsafe_allow_html=True)

        # Demo patients grid
        demo_pids = [pid for pid in sorted_pids if not is_live(pid)]
        if demo_pids:
            st.markdown('<div class="section-label">⬜ DEMO PATIENTS (simulated)</div>',
                        unsafe_allow_html=True)
            cols = st.columns(2)
            for i, pid in enumerate(demo_pids):
                pdata  = all_patients[pid]
                state  = pdata.get("state","Normal")
                col_c  = CLASS_COLORS[state]
                icon   = CLASS_ICONS[state]
                conf   = int(pdata.get("confidence",0)*100)
                name   = pdata.get("patient_name","—")
                age    = pdata.get("age","—")
                diag   = pdata.get("diagnosis","—")
                ward   = pdata.get("ward","—")
                bed    = pdata.get("bed","—")
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="patient-card" style="border-left:4px solid {col_c};opacity:0.82">
                      <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-size:13px;font-weight:600;color:#e8f0ff">
                          {icon} {name} <span class="badge-demo">DEMO</span>
                        </span>
                        <span style="font-family:'Space Mono',monospace;font-size:12px;color:{col_c}">{conf}%</span>
                      </div>
                      <div style="font-size:11px;color:#4a6080;margin:4px 0">{ward} · {bed} · Age {age}</div>
                      <div style="font-size:11px;color:#4a6080;margin-bottom:6px">{diag}</div>
                      <span style="font-family:'Space Mono',monospace;font-size:11px;
                           background:{col_c}22;color:{col_c};padding:2px 8px;border-radius:4px;">
                        {state.upper()}
                      </span>
                    </div>""", unsafe_allow_html=True)

        # Confidence bar chart
        st.markdown('<div class="section-label">SEIZURE CONFIDENCE — ALL PATIENTS</div>',
                    unsafe_allow_html=True)
        names_list = [all_patients[p].get("patient_name","").split()[0] for p in sorted_pids]
        sz_confs   = [all_patients[p].get("conf_seizure",0) for p in sorted_pids]
        colors_bar = [CLASS_COLORS[all_patients[p].get("state","Normal")] for p in sorted_pids]
        opacities  = [1.0 if is_live(pid) else 0.45 for pid in sorted_pids]

        fig_ov = go.Figure(go.Bar(
            x=names_list, y=sz_confs,
            marker=dict(color=colors_bar, opacity=opacities),
            text=[f"{'🔴' if is_live(pid) else '⬜'} {int(v*100)}%"
                  for pid,v in zip(sorted_pids,sz_confs)],
            textposition="outside",
            textfont=dict(family="Space Mono",size=10,color="#4a6080"),
        ))
        fig_ov.update_layout(**_PL, height=230, showlegend=False,
                             yaxis=dict(range=[0,1.3],**_AX,title="Seizure Conf."),
                             xaxis=dict(**_AX))
        st.plotly_chart(fig_ov, use_container_width=True, config={"displayModeBar":False})


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PATIENT DETAIL
# ════════════════════════════════════════════════════════════════════════════
with tab_patient:

    sel_pid = st.session_state.selected_pid
    if not sel_pid or sel_pid not in all_patients:
        st.info("Select a patient from the sidebar.")
    else:
        pdata   = all_patients[sel_pid]
        state   = pdata.get("state","Normal")
        col     = CLASS_COLORS[state]
        icon    = CLASS_ICONS[state]
        conf    = pdata.get("confidence",0)
        name    = pdata.get("patient_name","—")
        age     = pdata.get("age","—")
        diag    = pdata.get("diagnosis","—")
        ward    = pdata.get("ward","—")
        bed     = pdata.get("bed","—")
        ec      = pdata.get("emergency_contact","—")
        gps_lat = pdata.get("gps_lat","")
        gps_lng = pdata.get("gps_lng","")
        gps_lbl = pdata.get("gps_label","—")
        patient_is_live = is_live(sel_pid)

        # Mode banner
        if patient_is_live:
            st.markdown("""
            <div style="background:rgba(255,59,92,.08);border:1px solid rgba(255,59,92,.35);
                 border-radius:8px;padding:8px 16px;margin-bottom:12px;font-size:12px;color:#ff3b5c;
                 font-family:'Space Mono',monospace;">
              🔴 LIVE PATIENT — Real ML predictions · GPIO active · SMS alerts enabled
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(74,96,128,.08);border:1px solid #1e2e48;
                 border-radius:8px;padding:8px 16px;margin-bottom:12px;font-size:12px;color:#4a6080;
                 font-family:'Space Mono',monospace;">
              ⬜ DEMO PATIENT — Simulated state · No hardware alerts · No SMS
            </div>""", unsafe_allow_html=True)

        # Patient info + map side by side
        dt_col1, dt_col2 = st.columns([1, 1])
        with dt_col1:
            _pt_photo = _load_patient_photo_pil(sel_pid)
            if _pt_photo is not None:
                _ph_col, _ = st.columns([1, 4])
                with _ph_col:
                    st.image(_pt_photo, width=110, caption=name)
            sz_glow = "box-shadow:0 0 30px rgba(255,59,92,.15);" if state=="Seizure" and patient_is_live else ""
            st.markdown(f"""
            <div style="background:#111d30;border:1px solid {'#ff3b5c' if state=='Seizure' and patient_is_live else '#1e2e48'};
                 border-left:5px solid {col};border-radius:14px;
                 padding:18px 22px;{sz_glow}margin-bottom:16px;">
              <div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center">
                <div style="width:52px;height:52px;border-radius:12px;
                     background:{col}22;display:flex;align-items:center;
                     justify-content:center;font-size:22px;">{icon}</div>
                <div style="flex:1">
                  <div style="font-size:20px;font-weight:600;color:#e8f0ff">{name}</div>
                  <div style="font-size:12px;color:#4a6080;margin-top:3px">
                    {diag} · Age {age} · {ward} · {bed}
                  </div>
                  <div style="font-size:12px;color:#4a6080;margin-top:2px">
                    Emergency contact: {ec}
                  </div>
                  {f'<div style="font-size:11px;color:#4a6080;margin-top:2px">📍 {gps_lbl}</div>' if gps_lbl != "—" else ''}
                </div>
                <div style="text-align:right">
                  <span style="font-family:'Space Mono',monospace;font-size:13px;
                       background:{col}22;color:{col};padding:4px 12px;border-radius:6px;">
                    {state.upper()}
                  </span>
                  <div style="font-family:'Space Mono',monospace;font-size:20px;
                       font-weight:700;color:{col};margin-top:6px">{int(conf*100)}%</div>
                  <div style="font-size:10px;color:#4a6080">confidence</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        with dt_col2:
            # ── EMBEDDED MAP for this patient ────────────────────────────────
            if gps_lat and gps_lng:
                st.markdown('<div class="section-label">PATIENT LOCATION</div>',
                            unsafe_allow_html=True)
                components.html(
                    make_osm_embed(float(gps_lat), float(gps_lng),
                                   gps_lbl if gps_lbl != "—" else name, zoom=18),
                    height=330, scrolling=False
                )
            else:
                st.markdown("""
                <div style="background:#111d30;border:1px solid #1e2e48;border-radius:12px;
                     height:280px;display:flex;align-items:center;justify-content:center;
                     color:#4a6080;font-size:13px;margin-top:24px;">
                  📍 No GPS coordinates for this patient
                </div>""", unsafe_allow_html=True)

        # Confidence vector
        st.markdown('<div class="section-label">CONFIDENCE VECTOR</div>', unsafe_allow_html=True)
        conf_vals = [pdata.get("conf_normal",0),
                     pdata.get("conf_preseizure",0),
                     pdata.get("conf_seizure",0)]
        fig_cv = go.Figure()
        for cls,cv in zip(CLASSES,conf_vals):
            fig_cv.add_trace(go.Bar(
                x=[cls],y=[cv],name=cls,
                marker_color=CLASS_COLORS[cls],
                opacity=1.0 if cls==state else 0.3,
                text=[f"{int(cv*100)}%"],textposition="outside",
                textfont=dict(family="Space Mono",size=12,color=CLASS_COLORS[cls])))
        fig_cv.update_layout(**_PL,height=210,showlegend=False,bargap=0.38,
                             yaxis=dict(range=[0,1.25],**_AX),xaxis=dict(**_AX))
        st.plotly_chart(fig_cv,use_container_width=True,config={"displayModeBar":False})

        # EEG bands
        if patient_is_live:
            st.markdown('<div class="section-label">EEG BAND POWER (LIVE)</div>',
                        unsafe_allow_html=True)
            bvals = [pdata.get("delta",0),pdata.get("theta",0),
                     pdata.get("alpha",0),pdata.get("beta",0)]
            fig_bp = go.Figure(go.Bar(
                x=["Delta","Theta","Alpha","Beta"],y=bvals,
                marker=dict(color=[col]*4,opacity=[1,.85,.7,.55]),
                text=[f"{v:.3f}" for v in bvals],textposition="outside",
                textfont=dict(family="Space Mono",size=10,color="#4a6080")))
            fig_bp.update_layout(**_PL,height=190,showlegend=False,
                                 yaxis=dict(**_AX),xaxis=dict(**_AX))
            st.plotly_chart(fig_bp,use_container_width=True,config={"displayModeBar":False})

        # History chart
        history = get_patient_history(sel_pid)
        if not history.empty:
            st.markdown('<div class="section-label">CONFIDENCE HISTORY</div>',
                        unsafe_allow_html=True)

            # ── Sparkline summary row ────────────────────────────────────────
            sp1, sp2, sp3, sp4 = st.columns(4)
            sz_df = history[history["state"]=="Seizure"]
            ps_df = history[history["state"]=="Pre-Seizure"]
            rate  = len(sz_df)/len(history)*100 if len(history)>0 else 0
            in_state_str = time_in_state(history, state)
            with sp1: st.metric("Total Samples", len(history))
            with sp2: st.metric("Seizure Events", len(sz_df))
            with sp3: st.metric("Pre-Seizure Events", len(ps_df))
            with sp4: st.metric("Time in Current State", in_state_str)

            # Main history chart — last 60 samples
            recent = history.tail(60).reset_index(drop=True)
            fig_h  = go.Figure()
            for sn,c in CLASS_COLORS.items():
                mask = recent["state"]==sn
                fig_h.add_trace(go.Scatter(
                    x=recent.index, y=recent["confidence"].where(mask),
                    mode="lines+markers", name=sn,
                    line=dict(color=c,width=2), marker=dict(size=5),
                    connectgaps=False))
            fig_h.update_layout(**_PL,height=220,
                                yaxis=dict(range=[0,1.05],**_AX),
                                xaxis=dict(title="Sample",**_AX))
            st.plotly_chart(fig_h,use_container_width=True,config={"displayModeBar":False})

            # Seizure rate donut
            if len(sz_df) > 0:
                st.markdown('<div class="section-label">STATE DISTRIBUTION</div>',
                            unsafe_allow_html=True)
                state_counts = history["state"].value_counts()
                fig_pie = go.Figure(go.Pie(
                    labels=state_counts.index.tolist(),
                    values=state_counts.values.tolist(),
                    marker_colors=[CLASS_COLORS.get(s,"#4a6080") for s in state_counts.index],
                    hole=0.55,
                    textfont=dict(family="Space Mono",size=10),
                ))
                fig_pie.update_layout(**_PL, height=220,
                                      annotations=[dict(text=f"{rate:.1f}%<br>seizure",
                                                        x=0.5,y=0.5,
                                                        font=dict(family="Space Mono",size=11,color="#ff3b5c"),
                                                        showarrow=False)])
                st.plotly_chart(fig_pie,use_container_width=True,config={"displayModeBar":False})


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — WARD MAP  (all patients on one dark map)
# ════════════════════════════════════════════════════════════════════════════
with tab_map:

    st.markdown('<div class="section-label">ALL PATIENTS — WARD LOCATION MAP</div>',
                unsafe_allow_html=True)

    map_html = make_all_patients_map(all_patients, sorted_pids)
    if map_html:
        components.html(map_html, height=460, scrolling=False)
        st.caption("Click any marker to see patient name, ward, and current state. "
                   "🔴 Seizure · 🟡 Pre-Seizure · 🟢 Normal")
    else:
        st.info("No GPS coordinates found for any patient. Add gps_lat / gps_lng to patients.py.")

    # Individual patient map selector
    st.markdown('<div class="section-label">INDIVIDUAL PATIENT MAP</div>',
                unsafe_allow_html=True)
    map_patient_options = {
        all_patients[p].get("patient_name","—") + (" 🔴" if is_live(p) else " ⬜"): p
        for p in sorted_pids
        if all_patients[p].get("gps_lat") and all_patients[p].get("gps_lng")
    }

    if map_patient_options:
        sel_map_name = st.selectbox("Select patient to locate",
                                    list(map_patient_options.keys()),
                                    label_visibility="collapsed")
        sel_map_pid  = map_patient_options[sel_map_name]
        mp_data      = all_patients[sel_map_pid]
        mp_lat       = float(mp_data["gps_lat"])
        mp_lng       = float(mp_data["gps_lng"])
        mp_lbl       = mp_data.get("gps_label", mp_data.get("patient_name","—"))
        mp_state     = mp_data.get("state","Normal")
        mp_col       = CLASS_COLORS[mp_state]

        st.markdown(f"""
        <div style="background:#111d30;border:1px solid #1e2e48;border-left:4px solid {mp_col};
             border-radius:10px;padding:10px 16px;margin-bottom:8px;font-size:12px;color:#4a6080;">
          {CLASS_ICONS[mp_state]} <b style="color:#e8f0ff">{mp_data.get('patient_name','—')}</b> ·
          {mp_data.get('ward','—')} · {mp_data.get('bed','—')} ·
          <span style="color:{mp_col}">{mp_state}</span>
        </div>""", unsafe_allow_html=True)

        components.html(
            make_osm_embed(mp_lat, mp_lng, mp_lbl, zoom=18),
            height=360, scrolling=False
        )
    else:
        st.info("No patients have GPS coordinates configured.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVENT LOG
# ════════════════════════════════════════════════════════════════════════════
with tab_log:

    patient_options = {all_patients[p].get("patient_name","—") +
                       (" 🔴" if is_live(p) else " ⬜"): p
                       for p in sorted_pids}
    sel_name = st.selectbox("Select patient for log",
                            list(patient_options.keys()),
                            label_visibility="collapsed")
    log_pid  = patient_options.get(sel_name, sorted_pids[0] if sorted_pids else None)

    if log_pid:
        history = get_patient_history(log_pid)
        if history.empty:
            st.info("No history yet for this patient.")
        else:
            if not is_live(log_pid):
                st.caption("⬜ Demo patient — simulated data, no real EEG values")

            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                state_filter = st.multiselect("State", CLASSES, default=CLASSES,
                                              label_visibility="collapsed")
            with col_f2:
                n_rows = st.slider("Rows", 10, 200, 50, label_visibility="collapsed")
            with col_f3:
                show_seizure_only = st.checkbox("Seizures only", value=False)

            if show_seizure_only:
                state_filter = ["Seizure"]

            disp = (history[history["state"].isin(state_filter)]
                    .tail(n_rows).reset_index(drop=True))
            show_cols = ["timestamp","state","confidence","sms_sent"]
            if is_live(log_pid):
                show_cols = ["timestamp","state","confidence",
                             "delta","theta","alpha","beta","sms_sent"]
            show = disp[show_cols].copy()
            show["timestamp"]  = show["timestamp"].dt.strftime("%H:%M:%S")
            show["confidence"] = (show["confidence"]*100).round(1).astype(str)+"%"
            if is_live(log_pid):
                show.columns = ["Time","State","Conf","Delta","Theta","Alpha","Beta","SMS"]
            else:
                show.columns = ["Time","State","Conf","SMS"]
            st.dataframe(show, use_container_width=True, height=360)

            # Summary stats below log
            if len(history) > 0:
                sz_count = len(history[history["state"]=="Seizure"])
                ps_count = len(history[history["state"]=="Pre-Seizure"])
                st.caption(f"Total: {len(history)} records · Seizure: {sz_count} · "
                           f"Pre-Seizure: {ps_count} · "
                           f"Seizure rate: {sz_count/len(history)*100:.1f}%")

            csv = history.to_csv(index=False).encode("utf-8")
            pname = all_patients.get(log_pid,{}).get("patient_name","patient")
            st.download_button(
                "⬇️ Download CSV", data=csv,
                file_name=f"neurowatch_{log_pid}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL METRICS
# ════════════════════════════════════════════════════════════════════════════
with tab_metrics_tab:

    if not metrics:
        st.info("⏳ Metrics available after first training cycle.")
    else:
        ts = metrics.get("timestamp","")
        try: ts = datetime.fromisoformat(ts).strftime("%d %b %Y  %H:%M:%S")
        except Exception: pass
        st.caption(f"Last trained: {ts}  ·  Model trained on full Bonn dataset · Used for live patient")

        m1,m2,m3,m4,m5,m6 = st.columns(6)
        def _mc(col,label,val):
            c = "#00e88f" if val>=.9 else "#ffc93c" if val>=.7 else "#ff3b5c"
            col.markdown(f"""<div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value" style="color:{c}">{val*100:.1f}%</div>
            </div>""",unsafe_allow_html=True)

        _mc(m1,"Accuracy",      metrics.get("accuracy",0))
        _mc(m2,"Precision",     metrics.get("precision",0))
        _mc(m3,"Recall",        metrics.get("recall",0))
        _mc(m4,"F1 Score",      metrics.get("f1",0))
        _mc(m5,"Specificity",   metrics.get("specificity",0))
        _mc(m6,"Seiz. Recall",  metrics.get("seizure_recall",0))

        st.markdown("<br>",unsafe_allow_html=True)
        labels = ["Accuracy","Precision","Recall","F1","Specificity","Seiz.Recall"]
        vals   = [metrics.get(k,0) for k in
                  ["accuracy","precision","recall","f1","specificity","seizure_recall"]]
        colors = ["#00e88f" if v>=.9 else "#ffc93c" if v>=.7 else "#ff3b5c" for v in vals]
        fig_m  = go.Figure(go.Bar(
            x=vals,y=labels,orientation="h",marker_color=colors,
            text=[f"{v*100:.1f}%" for v in vals],textposition="outside",
            textfont=dict(family="Space Mono",size=11,color="#4a6080")))
        fig_m.update_layout(**_PL,height=260,showlegend=False,
                            xaxis=dict(range=[0,1.2],**_AX),yaxis=dict(**_AX))
        st.plotly_chart(fig_m,use_container_width=True,config={"displayModeBar":False})

        cm_data = metrics.get("confusion_matrix")
        if cm_data:
            st.markdown('<div class="section-label">CONFUSION MATRIX</div>',
                        unsafe_allow_html=True)
            cm_arr = np.array(cm_data)
            n      = cm_arr.shape[0]
            lbls   = CLASSES[:n]
            fig_cm = go.Figure(go.Heatmap(
                z=cm_arr,x=lbls,y=lbls,
                colorscale=[[0,"#111d30"],[1,"#00d4ff"]],
                text=cm_arr,texttemplate="%{text}",
                textfont=dict(family="Space Mono",size=14),showscale=False))
            fig_cm.update_layout(**_PL,height=300,
                                 xaxis=dict(title="Predicted",side="bottom",**_AX),
                                 yaxis=dict(title="True",autorange="reversed",**_AX))
            st.plotly_chart(fig_cm,use_container_width=True,config={"displayModeBar":False})

# =============================================================================
# AUTO-REFRESH
# =============================================================================
time.sleep(REFRESH_INTERVAL)
st.rerun()