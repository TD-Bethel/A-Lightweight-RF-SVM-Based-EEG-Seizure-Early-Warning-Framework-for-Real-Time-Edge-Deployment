"""
NeuroWatch --- Patient Mobile UI
Mobile-friendly Streamlit version of the patient watch UI.
Run: streamlit run src/patient_mobile_ui.py
"""

import json
import os
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go

# -- Paths ---------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATIENTS_JSON   = os.path.join(BASE_DIR, "json", "neurowatch_patients.json")
HISTORY_PREFIX  = os.path.join(BASE_DIR, "json", "neurowatch_history_")
HOME_PATIENT_ID = "P001"
HISTORY_DOTS    = 30

# -- Palette -------------------------------------------------------------------
BG      = "#0d0f1a"
CARD    = "#181c30"
BORDER  = "#252a45"
GREEN   = "#00e676"
AMBER   = "#ffca28"
RED     = "#ff5252"
MUTED   = "#5a6080"
TEXT    = "#e8ecf4"
SUBTEXT = "#8a93b0"

STATE_COLORS = {"Seizure": RED, "Pre-Seizure": AMBER, "Normal": GREEN}
STATE_LABELS = {
    "Seizure":     "🔴  Alert — Please stay calm",
    "Pre-Seizure": "⚡  Stay relaxed",
    "Normal":      "✅  You're doing great",
}
STATE_ADVICE = {
    "Seizure":     "Your caregiver has been notified.",
    "Pre-Seizure": "Find a safe, comfortable place to sit or lie down and take medication.",
    "Normal":      "Everything looks normal. Keep resting.",
}
STATE_PILL = {"Seizure": "ALERT", "Pre-Seizure": "HEADS UP", "Normal": "NORMAL"}

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="NeuroWatch",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
  html, body,
  [data-testid="stAppViewContainer"],
  [data-testid="stApp"],
  section.main,
  .block-container {{
    background-color: {BG} !important;
    color: {TEXT} !important;
    font-family: 'Segoe UI', sans-serif;
    max-width: 480px !important;
    margin: 0 auto !important;
    padding: 0 !important;
  }}
  .block-container {{
    padding-top: 0.5rem !important;
    padding-bottom: 1rem !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
  }}
  #MainMenu, header, footer,
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stStatusWidget"] {{ display: none !important; }}
  .js-plotly-plot .plotly {{ background: transparent !important; }}
</style>
""", unsafe_allow_html=True)


# -- Data helpers --------------------------------------------------------------
def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_patient():
    raw = _read_json(PATIENTS_JSON, {})
    patients = list(raw.values()) if isinstance(raw, dict) else (raw or [])
    match = next((p for p in patients if p.get("patient_id") == HOME_PATIENT_ID), None)
    return match or (patients[0] if patients else None)


def load_history(pid):
    raw = _read_json(f"{HISTORY_PREFIX}{pid}.json", [])
    return raw[-HISTORY_DOTS * 2:] if isinstance(raw, list) else []


def fmt_time(iso):
    if not iso:
        return "---"
    try:
        dt = (datetime.fromisoformat(iso) if "T" in iso
              else datetime.strptime(iso, "%Y-%m-%d %H:%M:%S"))
        return dt.strftime("%H:%M")
    except Exception:
        return "---"


# -- Ring chart ----------------------------------------------------------------
def make_ring(state, confidence):
    color = STATE_COLORS.get(state, GREEN)
    pill  = STATE_PILL.get(state, state)
    pct   = int(confidence * 100)

    fig = go.Figure()

    # Background ring (full circle, dim colour)
    fig.add_trace(go.Pie(
        values=[1], hole=0.72,
        marker_colors=[BORDER],
        textinfo="none", hoverinfo="skip",
        showlegend=False,
    ))

    # Confidence arc (clockwise from top)
    fig.add_trace(go.Pie(
        values=[confidence, max(0.001, 1 - confidence)],
        hole=0.72,
        marker_colors=[color, "rgba(0,0,0,0)"],
        textinfo="none", hoverinfo="skip",
        showlegend=False,
        direction="clockwise",
        rotation=90,
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=240,
        annotations=[dict(
            text=(
                f"<b><span style='font-size:36px;color:{color}'>{pct}%</span></b>"
                f"<br><span style='font-size:12px;color:{SUBTEXT}'>brain activity</span>"
                f"<br><br><b><span style='font-size:13px;color:{color}'>{pill}</span></b>"
            ),
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, align="center",
        )],
    )
    return fig


# -- History dots HTML ---------------------------------------------------------
def history_html(states):
    recent = states[-HISTORY_DOTS:]
    dots = ""
    for i, s in enumerate(recent):
        col  = STATE_COLORS.get(s, MUTED)
        size = "22px" if i == len(recent) - 1 else "16px"
        glow = f"box-shadow:0 0 6px 2px {col}88;" if i == len(recent) - 1 else ""
        dots += (
            f"<span style='display:inline-block;width:{size};height:{size};"
            f"border-radius:5px;background:{col};margin:2px;{glow}'></span>"
        )
    legend = "".join([
        f"<span style='display:inline-flex;align-items:center;gap:4px;'>"
        f"<span style='display:inline-block;width:10px;height:10px;border-radius:3px;"
        f"background:{c};'></span>"
        f"<span style='color:{SUBTEXT};font-size:11px;'>{lbl}</span></span>"
        for c, lbl in [(GREEN, "Normal"), (AMBER, "Heads-up"), (RED, "Alert")]
    ])
    return f"""
<div style='background:{CARD};border:1px solid {BORDER};border-radius:12px;
            padding:12px 14px;'>
  <div style='color:{SUBTEXT};font-size:11px;font-weight:600;
              letter-spacing:0.05em;margin-bottom:8px;'>RECENT ACTIVITY</div>
  <div style='line-height:1.6;min-height:24px;'>
    {dots or f"<span style='color:{MUTED}'>No data yet</span>"}
  </div>
  <div style='margin-top:10px;display:flex;gap:16px;flex-wrap:wrap;'>{legend}</div>
</div>"""


# -- Main fragment (auto-refresh every 2 s) ------------------------------------
@st.fragment(run_every=2)
def patient_view():
    patient = load_patient()

    if not patient:
        st.markdown(
            f"<p style='color:{SUBTEXT};text-align:center;padding:40px;'>"
            "Waiting for monitoring data...</p>",
            unsafe_allow_html=True,
        )
        return

    pid      = patient.get("patient_id", "")
    name     = patient.get("patient_name", "Patient")
    state    = patient.get("state", "Normal")
    conf     = float(patient.get("confidence", 0.75))
    ts       = patient.get("timestamp", "")
    emer     = patient.get("emergency_contact", "---")
    emer_no  = str(patient.get("emergency_phone", "---"))
    sms_sent = bool(patient.get("sms_sent", False))
    color    = STATE_COLORS.get(state, GREEN)

    history     = [r.get("state", "Normal") for r in load_history(pid)]
    first_name  = name.split()[0]
    now_str     = datetime.now().strftime("%H:%M")

    # Consecutive normal streak
    streak = 0
    for s in reversed(history):
        if s == "Normal":
            streak += 1
        else:
            break

    # -- Top bar
    st.markdown(f"""
<div style='display:flex;justify-content:space-between;align-items:center;
            padding:14px 20px 4px 20px;'>
  <span style='color:{SUBTEXT};font-size:13px;'>Hello, {first_name} 👋</span>
  <span style='color:{MUTED};font-size:13px;'>{now_str}</span>
</div>
<div style='text-align:center;padding:2px 0 0 0;'>
  <span style='color:{TEXT};font-size:20px;font-weight:700;'>{name}</span>
</div>
""", unsafe_allow_html=True)

    # -- Ring
    st.plotly_chart(make_ring(state, conf),
                    use_container_width=True,
                    config={"displayModeBar": False})

    # -- Status text
    status_text = STATE_LABELS.get(state, state)
    if state == "Seizure" and sms_sent:
        status_text = "🔴  Alert — Doctor contacted"
    advice_text = STATE_ADVICE.get(state, "")
    if state == "Seizure":
        advice_text = (
            "Doctor has been contacted. Please stay calm and wait for assistance."
            if sms_sent else "CONTACT DOCTOR / EMERGENCY"
        )

    st.markdown(f"""
<div style='text-align:center;padding:0 20px 4px 20px;'>
  <div style='color:{color};font-size:16px;font-weight:700;'>{status_text}</div>
  <div style='color:{SUBTEXT};font-size:12px;margin-top:4px;'>{advice_text}</div>
</div>
""", unsafe_allow_html=True)

    # -- Cards row
    streak_color = GREEN if streak >= 5 else TEXT
    st.markdown(f"""
<div style='display:flex;gap:10px;padding:14px 20px 0 20px;'>
  <div style='flex:1;background:{CARD};border:1px solid {BORDER};border-radius:12px;
              padding:12px 14px;'>
    <div style='color:{SUBTEXT};font-size:10px;font-weight:600;
                letter-spacing:0.05em;'>✅ GOOD STREAK</div>
    <div style='color:{streak_color};font-size:18px;font-weight:700;margin-top:4px;'>
      {streak} reading{'s' if streak != 1 else ''}
    </div>
  </div>
  <div style='flex:1;background:{CARD};border:1px solid {BORDER};border-radius:12px;
              padding:12px 14px;'>
    <div style='color:{SUBTEXT};font-size:10px;font-weight:600;
                letter-spacing:0.05em;'>🕐 LAST CHECKED</div>
    <div style='color:{TEXT};font-size:18px;font-weight:700;margin-top:4px;'>
      {fmt_time(ts)}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # -- History dots
    st.markdown(
        f"<div style='padding:14px 20px 0 20px;'>{history_html(history)}</div>",
        unsafe_allow_html=True,
    )

    # -- Emergency contact
    st.markdown(f"""
<div style='margin:14px 20px 0 20px;background:{CARD};border:1px solid {BORDER};
            border-radius:12px;padding:14px;'>
  <div style='color:{SUBTEXT};font-size:10px;font-weight:600;
              letter-spacing:0.05em;'>🚨 EMERGENCY CONTACT</div>
  <div style='color:{TEXT};font-size:15px;font-weight:700;margin-top:6px;'>{emer}</div>
  <div style='color:{SUBTEXT};font-size:13px;margin-top:2px;'>{emer_no}</div>
</div>
""", unsafe_allow_html=True)

    # -- Live indicator
    st.markdown(f"""
<div style='display:flex;align-items:center;padding:14px 20px 20px 20px;gap:8px;'>
  <span style='display:inline-block;width:8px;height:8px;border-radius:50%;
               background:{GREEN};box-shadow:0 0 6px {GREEN};'></span>
  <span style='color:{MUTED};font-size:11px;'>Live monitoring active</span>
</div>
""", unsafe_allow_html=True)


patient_view()
