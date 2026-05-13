"""
NeuroWatch --- Patient Mobile UI
Mobile-friendly Streamlit version of the patient watch UI.
Run: streamlit run src/patient_mobile_ui.py

HOW THIS FILE FITS INTO THE PROJECT:
  - This is the patient-facing screen. It shows ONE patient's current state
    (Normal / Pre-Seizure / Seizure) in a phone-sized layout.
  - Data is read from JSON files written by main_pi_bios_v15.py running on
    the Raspberry Pi. This script does NOT talk to the Pi directly; it just
    reads the shared JSON files.
  - Auto-refreshes every 2 seconds via Streamlit's @st.fragment(run_every=2).

TO RUN:
  streamlit run src/patient_mobile_ui.py
"""

# =============================================================================
# SECTION 1 — IMPORTS
#
# WHY: Streamlit drives the web UI. Plotly draws the confidence ring chart.
# json/os handle file paths and data loading. datetime formats timestamps.
#
# TO CHANGE: If you want to add data manipulation (e.g., pandas DataFrames for
# trend analysis), import pandas here. If you want SMS from the UI side, add
# twilio imports here (though SMS is already handled by the Pi script).
# =============================================================================
import json
import os
from datetime import datetime

import streamlit as st          # Core UI framework — every st.* call renders something
import plotly.graph_objects as go  # Used only for the ring/donut confidence chart


# =============================================================================
# SECTION 2 — FILE PATHS AND CONSTANTS
#
# WHY: Centralising paths here means you only need to change one line if you
# move the JSON files. Using os.path.join makes the code cross-platform
# (Windows backslash vs Linux forward slash).
#
# HOW TO CHANGE:
#   - PATIENTS_JSON: points to the file the Pi writes with current patient
#     state. Change this if you rename or move the file.
#   - HISTORY_PREFIX: the Pi appends history records to
#     neurowatch_history_P001.json (one file per patient ID). The prefix is
#     the path up to the patient-ID part.
#   - HOME_PATIENT_ID: which patient this mobile app is showing. Change to
#     "P002" etc. to show a different patient, or make it dynamic (see
#     patient_view() below).
#   - HISTORY_DOTS: how many past readings to show in the dot timeline at
#     the bottom. 30 dots = 30 readings × ~2 s/reading = ~1 minute of history.
# =============================================================================
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATIENTS_JSON   = os.path.join(BASE_DIR, "json", "neurowatch_patients.json")
HISTORY_PREFIX  = os.path.join(BASE_DIR, "json", "neurowatch_history_")
HOME_PATIENT_ID = "P001"   # Change this to show a different patient's data
HISTORY_DOTS    = 30       # Number of history dots shown in the activity strip


# =============================================================================
# SECTION 3 — COLOUR PALETTE
#
# WHY: All colours are defined once here so you can restyle the whole app by
# editing this block. The dark navy theme (#0d0f1a) was chosen to reduce eye
# strain for patients monitoring at night.
#
# HOW TO CHANGE: Replace any hex value to change a colour globally.
#   - BG    : the outermost background (the "page" colour).
#   - CARD  : slightly lighter panel colour used for info cards.
#   - BORDER: subtle border around cards.
#   - GREEN / AMBER / RED: traffic-light colours for Normal/Pre-Seizure/Seizure.
#   - MUTED / TEXT / SUBTEXT: text hierarchy — TEXT is the main body,
#     SUBTEXT is secondary labels, MUTED is dimmed decorative elements.
# =============================================================================
BG      = "#0d0f1a"   # Near-black dark navy — main page background
CARD    = "#181c30"   # Slightly lighter than BG — used for info card panels
BORDER  = "#252a45"   # Subtle card border colour
GREEN   = "#00e676"   # Bright green — Normal state
AMBER   = "#ffca28"   # Warm amber/yellow — Pre-Seizure state
RED     = "#ff5252"   # Bright red — Seizure / Alert state
MUTED   = "#5a6080"   # Dimmed grey-blue — decorative / less important text
TEXT    = "#e8ecf4"   # Near-white — primary body text
SUBTEXT = "#8a93b0"   # Mid-tone blue-grey — secondary labels


# =============================================================================
# SECTION 4 — STATE-DEPENDENT TEXT AND COLOUR MAPS
#
# WHY: The three dictionaries below translate a state string ("Normal",
# "Pre-Seizure", "Seizure") into the colour, short pill badge, headline
# status text, and advice sentence shown to the patient. Keeping these as
# dictionaries makes it trivial to update copy or add new states.
#
# HOW TO CHANGE:
#   - STATE_COLORS : change which colour maps to which state.
#   - STATE_LABELS : change the headline text shown under the ring chart.
#   - STATE_ADVICE : change the advice sentence shown beneath the headline.
#   - STATE_PILL   : change the short badge text in the centre of the ring.
#
# NOTE: If you add a new state (e.g., "Unknown"), add a key to all four dicts
# so none of them return None unexpectedly.
# =============================================================================
STATE_COLORS = {"Seizure": RED, "Pre-Seizure": AMBER, "Normal": GREEN}

# Short headline shown directly under the confidence ring
STATE_LABELS = {
    "Seizure":     "🔴  Alert — Please stay calm",
    "Pre-Seizure": "⚡  Stay relaxed",
    "Normal":      "✅  You're doing great",
}

# One-sentence actionable advice shown beneath the headline
STATE_ADVICE = {
    "Seizure":     "Your caregiver has been notified.",
    "Pre-Seizure": "Find a safe, comfortable place to sit or lie down and take medication.",
    "Normal":      "Everything looks normal. Keep resting.",
}

# Badge text rendered in the centre of the confidence ring
STATE_PILL = {"Seizure": "ALERT", "Pre-Seizure": "HEADS UP", "Normal": "NORMAL"}


# =============================================================================
# SECTION 5 — STREAMLIT PAGE CONFIGURATION
#
# WHY: st.set_page_config() MUST be called before any other Streamlit command.
# It sets the browser tab title, icon, layout width, and sidebar default.
# "centered" layout + the CSS below constrains the app to 480 px wide,
# which mimics a phone screen when opened on desktop browsers.
#
# HOW TO CHANGE:
#   - page_title : changes the browser tab text.
#   - page_icon  : any emoji or URL to a .png favicon.
#   - layout     : "centered" or "wide". "wide" uses the full browser width.
#   - initial_sidebar_state: "collapsed" hides the sidebar by default;
#     change to "expanded" if you add sidebar controls.
# =============================================================================
st.set_page_config(
    page_title="NeuroWatch",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# SECTION 6 — GLOBAL CSS INJECTION
#
# WHY: Streamlit does not expose direct styling for its outer container, so we
# inject a <style> block via st.markdown(unsafe_allow_html=True). This block:
#   1. Applies the dark background to every Streamlit wrapper div.
#   2. Caps the content width at 480 px and centres it (phone layout).
#   3. Removes Streamlit's built-in toolbar, menu, footer, and decorations so
#      the app looks like a native app rather than a data science dashboard.
#   4. Makes Plotly chart backgrounds transparent so the ring blends with the
#      dark page.
#
# HOW TO CHANGE:
#   - To widen the layout for tablets: change 480px to 768px.
#   - To restore the Streamlit header/menu: remove the display:none rule for
#     #MainMenu, header, footer, and [data-testid="stToolbar"].
#   - To change the font: replace 'Segoe UI' with any Google Font name after
#     adding a @import url(...) at the top of the style block.
#   - Note: f-string interpolation (e.g., {BG}) injects the hex colour values
#     from the palette defined above.
# =============================================================================
st.markdown(f"""
<style>
  /* Apply dark background and font to ALL Streamlit container divs */
  html, body,
  [data-testid="stAppViewContainer"],
  [data-testid="stApp"],
  section.main,
  .block-container {{
    background-color: {BG} !important;
    color: {TEXT} !important;
    font-family: 'Segoe UI', sans-serif;
    max-width: 480px !important;   /* Phone-width cap */
    margin: 0 auto !important;     /* Centre horizontally */
    padding: 0 !important;
  }}
  /* Fine-tune the inner content block padding */
  .block-container {{
    padding-top: 0.5rem !important;
    padding-bottom: 1rem !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
  }}
  /* Hide Streamlit chrome: hamburger menu, header bar, footer, toolbar, decorations */
  #MainMenu, header, footer,
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stStatusWidget"] {{ display: none !important; }}
  /* Make Plotly chart backgrounds transparent so the ring blends with the page */
  .js-plotly-plot .plotly {{ background: transparent !important; }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 7 — DATA HELPER FUNCTIONS
#
# WHY: These three small functions isolate all file I/O so that the main
# rendering function (patient_view) stays clean and readable. If the file
# format or location changes, you only need to update these helpers.
# =============================================================================

def _read_json(path, default):
    """
    Safely read a JSON file and return its contents.

    WHY: Wrapping json.load in a try/except prevents the whole app from
    crashing if the Pi hasn't written the file yet, or if it writes a
    partial/corrupt JSON mid-update. Instead, the caller gets `default`.

    ARGS:
      path    — absolute path to the JSON file.
      default — value to return if the file is missing or unreadable.

    RETURNS: parsed Python object (dict, list, etc.) or `default`.

    HOW TO CHANGE: If you switch from JSON to SQLite or a REST API,
    replace the body of this function — all callers stay the same.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_patient():
    """
    Load the current patient record for HOME_PATIENT_ID from PATIENTS_JSON.

    WHY: The patients JSON can be either a dict keyed by patient ID or a
    plain list of patient dicts (different Pi firmware versions write it
    differently). This function handles both shapes.

    RETURNS: a dict with keys patient_id, patient_name, state, confidence,
    timestamp, emergency_contact, emergency_phone, sms_sent — or None if
    no data is available yet.

    HOW TO CHANGE:
      - To support multiple patients on screen, return the full `patients`
        list instead of a single match.
      - To change which patient is shown, edit HOME_PATIENT_ID at the top
        of this file.
    """
    raw = _read_json(PATIENTS_JSON, {})
    # Normalise: handle both dict-of-dicts and list-of-dicts formats
    patients = list(raw.values()) if isinstance(raw, dict) else (raw or [])
    # Find the patient whose ID matches HOME_PATIENT_ID
    match = next((p for p in patients if p.get("patient_id") == HOME_PATIENT_ID), None)
    # Fall back to the first patient in the file if HOME_PATIENT_ID is not found
    return match or (patients[0] if patients else None)


def load_history(pid):
    """
    Load the recent history entries for a given patient ID.

    WHY: The Pi appends one record per detection cycle to
    neurowatch_history_<pid>.json. We only need the most recent HISTORY_DOTS
    readings for the dot timeline. Loading the tail slice also keeps memory
    usage bounded even if the file grows very large.

    ARGS:
      pid — patient ID string (e.g., "P001").

    RETURNS: list of history dicts (each has at least a "state" key), or []
    if the file does not exist.

    HOW TO CHANGE:
      - Increase HISTORY_DOTS (top of file) to show more dots.
      - If you change the history file naming scheme on the Pi, update
        HISTORY_PREFIX at the top of this file.
    """
    raw = _read_json(f"{HISTORY_PREFIX}{pid}.json", [])
    # Only keep the last (HISTORY_DOTS * 2) entries to limit memory, then
    # slice to exactly HISTORY_DOTS in history_html() below
    return raw[-HISTORY_DOTS * 2:] if isinstance(raw, list) else []


def fmt_time(iso):
    """
    Format an ISO-8601 or 'YYYY-MM-DD HH:MM:SS' timestamp string to 'HH:MM'.

    WHY: The Pi writes timestamps in two possible formats depending on the
    firmware version. This function handles both, and returns "---" gracefully
    if the string is empty or malformed.

    ARGS:
      iso — timestamp string from the patient JSON.

    RETURNS: "HH:MM" string, or "---" on failure.

    HOW TO CHANGE: To show seconds as well, change the strftime format to
    "%H:%M:%S". To show a date, use "%d %b %H:%M".
    """
    if not iso:
        return "---"
    try:
        # fromisoformat handles "2024-01-15T14:32:00" style
        # strptime handles "2024-01-15 14:32:00" style (no T separator)
        dt = (datetime.fromisoformat(iso) if "T" in iso
              else datetime.strptime(iso, "%Y-%m-%d %H:%M:%S"))
        return dt.strftime("%H:%M")
    except Exception:
        return "---"


# =============================================================================
# SECTION 8 — CONFIDENCE RING CHART
#
# WHY: A donut/ring chart conveys the model's confidence in the current
# prediction at a glance — patients don't need a bar chart or a number, just
# a visual dial. Plotly is used because Streamlit's native charts don't
# support donut-style charts with custom centre annotations.
#
# HOW THIS WORKS:
#   - Two overlapping Pie traces share the same hole (72% → only the outer
#     ring is visible).
#   - Trace 1: a single-segment ring in BORDER colour = the "empty" background.
#   - Trace 2: a two-segment ring where the coloured segment covers `confidence`
#     fraction of the circle (clockwise from the top).
#   - A text annotation is placed at the centre (x=0.5, y=0.5 in paper coords)
#     to show the percentage, "brain activity" label, and the state pill badge.
# =============================================================================

def make_ring(state, confidence):
    """
    Build and return a Plotly Figure for the confidence ring chart.

    ARGS:
      state      — string: "Normal", "Pre-Seizure", or "Seizure".
      confidence — float in [0.0, 1.0] representing model confidence.

    RETURNS: plotly.graph_objects.Figure.

    HOW TO CHANGE:
      - Ring thickness: change hole=0.72 (higher = thinner ring, lower = thicker).
      - Ring height: change height=240 in update_layout.
      - Centre text: edit the f-string in the annotations dict.
        font-size values control how large each line of text appears.
      - To show two rings (e.g., confidence + historical accuracy), add a
        third Pie trace with a smaller hole value.
    """
    color = STATE_COLORS.get(state, GREEN)  # Pick the state colour; fallback to green
    pill  = STATE_PILL.get(state, state)    # Short badge text for the ring centre
    pct   = int(confidence * 100)           # Convert 0.87 → 87 for display

    fig = go.Figure()

    # -- Background ring (full 360° circle in dim border colour) ----------------
    # This is the "track" that the confidence arc sits on top of.
    # values=[1] means one segment = the entire circle.
    fig.add_trace(go.Pie(
        values=[1], hole=0.72,
        marker_colors=[BORDER],
        textinfo="none", hoverinfo="skip",
        showlegend=False,
    ))

    # -- Confidence arc (the coloured fill) ------------------------------------
    # values=[confidence, 1-confidence] splits the circle into:
    #   - coloured segment: the model's confidence fraction
    #   - transparent segment: the remainder (makes the arc look like a gauge)
    # rotation=90 starts the arc from 12 o'clock (top), clockwise.
    # max(0.001, ...) prevents Plotly from throwing an error when confidence=1.0
    # because a 0-value segment is invalid.
    fig.add_trace(go.Pie(
        values=[confidence, max(0.001, 1 - confidence)],
        hole=0.72,
        marker_colors=[color, "rgba(0,0,0,0)"],  # Second segment is invisible
        textinfo="none", hoverinfo="skip",
        showlegend=False,
        direction="clockwise",
        rotation=90,   # Start at 12 o'clock
    ))

    # -- Layout: transparent background, no margins, centre annotation ----------
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent so the card BG shows through
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=240,
        annotations=[dict(
            # HTML rendered in the centre of the donut hole
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


# =============================================================================
# SECTION 9 — HISTORY DOT STRIP (HTML builder)
#
# WHY: Streamlit doesn't have a built-in "dot timeline" widget. We build the
# HTML string manually and inject it with st.markdown(unsafe_allow_html=True).
# Each dot is a coloured <span> styled as a rounded square. The most recent
# dot is slightly larger and has a glow effect to draw the eye.
#
# HOW TO CHANGE:
#   - Dot size: change "22px" (latest dot) and "16px" (older dots).
#   - Glow intensity: change "6px 2px" in the box-shadow value. Set to "" to
#     remove the glow entirely.
#   - Dot shape: change border-radius to "50%" for circles, "0" for squares.
#   - Number of dots: change HISTORY_DOTS at the top of this file.
# =============================================================================

def history_html(states):
    """
    Build an HTML string for the history dot strip.

    ARGS:
      states — list of state strings in chronological order (oldest first).

    RETURNS: HTML string containing the card div with dots and legend.

    HOW TO CHANGE: To add a tooltip on hover (showing the exact time), wrap
    each <span> in a <div title="HH:MM"> element and pass time data alongside
    the states list.
    """
    # Take only the last HISTORY_DOTS states (most recent window)
    recent = states[-HISTORY_DOTS:]
    dots = ""
    for i, s in enumerate(recent):
        col  = STATE_COLORS.get(s, MUTED)   # Dot colour based on state
        # The last dot (most recent reading) is larger to indicate "current"
        size = "22px" if i == len(recent) - 1 else "16px"
        # Glow effect only on the latest dot
        glow = f"box-shadow:0 0 6px 2px {col}88;" if i == len(recent) - 1 else ""
        dots += (
            f"<span style='display:inline-block;width:{size};height:{size};"
            f"border-radius:5px;background:{col};margin:2px;{glow}'></span>"
        )

    # Build the legend row (Green = Normal, Amber = Heads-up, Red = Alert)
    legend = "".join([
        f"<span style='display:inline-flex;align-items:center;gap:4px;'>"
        f"<span style='display:inline-block;width:10px;height:10px;border-radius:3px;"
        f"background:{c};'></span>"
        f"<span style='color:{SUBTEXT};font-size:11px;'>{lbl}</span></span>"
        for c, lbl in [(GREEN, "Normal"), (AMBER, "Heads-up"), (RED, "Alert")]
    ])

    # Wrap dots and legend in a styled card div
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


# =============================================================================
# SECTION 10 — MAIN PATIENT VIEW (auto-refreshing fragment)
#
# WHY: @st.fragment(run_every=2) tells Streamlit to re-run ONLY this function
# every 2 seconds without refreshing the whole page. This is more efficient
# than using st_autorefresh (which reloads the entire app) and avoids
# flickering. The function reads fresh JSON on every run.
#
# HOW TO CHANGE:
#   - Refresh rate: change run_every=2 to any number of seconds (e.g., 5).
#     Lower values give faster updates but more file I/O.
#   - To support patient switching: add a st.selectbox() above the ring that
#     lets the user pick a patient ID, then pass it to load_patient() and
#     load_history() instead of HOME_PATIENT_ID.
#
# LAYOUT ORDER (top to bottom):
#   1. Top bar    — greeting + current time
#   2. Ring chart — confidence donut
#   3. Status text — headline + advice
#   4. Cards row  — "good streak" count + "last checked" time
#   5. History dot strip — recent activity timeline
#   6. Emergency contact card
#   7. Live monitoring indicator
# =============================================================================

@st.fragment(run_every=2)  # Re-run this function every 2 seconds
def patient_view():
    # ── Load the patient record from JSON ──────────────────────────────────────
    patient = load_patient()

    # If no data is available yet (Pi hasn't started writing), show a message
    if not patient:
        st.markdown(
            f"<p style='color:{SUBTEXT};text-align:center;padding:40px;'>"
            "Waiting for monitoring data...</p>",
            unsafe_allow_html=True,
        )
        return

    # ── Unpack patient fields with safe defaults ────────────────────────────────
    # Each .get() call provides a fallback so the UI never crashes on missing keys
    pid      = patient.get("patient_id", "")
    name     = patient.get("patient_name", "Patient")
    state    = patient.get("state", "Normal")
    conf     = float(patient.get("confidence", 0.75))  # Confidence in [0, 1]
    ts       = patient.get("timestamp", "")            # ISO timestamp string
    emer     = patient.get("emergency_contact", "---") # Emergency contact name
    emer_no  = str(patient.get("emergency_phone", "---"))  # Their phone number
    sms_sent = bool(patient.get("sms_sent", False))    # Did the Pi send an SMS?
    color    = STATE_COLORS.get(state, GREEN)           # Colour for current state

    # ── Load history and derive computed values ─────────────────────────────────
    # Extract just the "state" string from each history record dict
    history     = [r.get("state", "Normal") for r in load_history(pid)]
    first_name  = name.split()[0]                       # "John Smith" → "John"
    now_str     = datetime.now().strftime("%H:%M")      # Wall-clock time for top bar

    # Count consecutive "Normal" readings from the end of history backwards.
    # This is the "good streak" shown in the card below the ring.
    # Example: [..., "Normal", "Normal", "Seizure", "Normal", "Normal"] → streak = 2
    streak = 0
    for s in reversed(history):
        if s == "Normal":
            streak += 1
        else:
            break  # Stop as soon as we hit a non-Normal reading

    # ── 1. TOP BAR — personalised greeting + clock ─────────────────────────────
    # Uses flexbox to push the name to the left and time to the right.
    # HOW TO CHANGE: add a battery icon or signal indicator on the right by
    # adding another <span> before the time span.
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

    # ── 2. RING CHART — confidence donut ───────────────────────────────────────
    # config={"displayModeBar": False} hides the Plotly toolbar (zoom, pan, etc.)
    # because patients don't need to interact with the chart.
    st.plotly_chart(make_ring(state, conf),
                    use_container_width=True,
                    config={"displayModeBar": False})

    # ── 3. STATUS TEXT — headline + advice ─────────────────────────────────────
    # Override the default label and advice text when the Pi has confirmed that
    # an SMS was sent (changes "Your caregiver has been notified" to
    # "Doctor has been contacted" — more specific and reassuring).
    status_text = STATE_LABELS.get(state, state)
    if state == "Seizure" and sms_sent:
        status_text = "🔴  Alert — Doctor contacted"

    advice_text = STATE_ADVICE.get(state, "")
    if state == "Seizure":
        # Two versions of the seizure advice depending on whether SMS was sent
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

    # ── 4. CARDS ROW — good streak + last checked ──────────────────────────────
    # Two side-by-side info cards using CSS flexbox (flex:1 = equal width).
    # streak_color turns GREEN when the streak is >= 5 readings (about 10 s of
    # Normal) to give positive reinforcement; stays TEXT otherwise.
    streak_color = GREEN if streak >= 5 else TEXT
    st.markdown(f"""
<div style='display:flex;gap:10px;padding:14px 20px 0 20px;'>
  <!-- Left card: consecutive Normal readings streak -->
  <div style='flex:1;background:{CARD};border:1px solid {BORDER};border-radius:12px;
              padding:12px 14px;'>
    <div style='color:{SUBTEXT};font-size:10px;font-weight:600;
                letter-spacing:0.05em;'>✅ GOOD STREAK</div>
    <div style='color:{streak_color};font-size:18px;font-weight:700;margin-top:4px;'>
      {streak} reading{'s' if streak != 1 else ''}
    </div>
  </div>
  <!-- Right card: timestamp of the most recent detection cycle -->
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

    # ── 5. HISTORY DOT STRIP ───────────────────────────────────────────────────
    # Wraps the HTML returned by history_html() in a padding div to align it
    # with the rest of the content (same 20 px side padding as the cards above).
    st.markdown(
        f"<div style='padding:14px 20px 0 20px;'>{history_html(history)}</div>",
        unsafe_allow_html=True,
    )

    # ── 6. EMERGENCY CONTACT CARD ──────────────────────────────────────────────
    # Displays the name and phone number stored in the patient record.
    # The Pi populates these from the patients JSON at startup.
    # HOW TO CHANGE: add a tel: hyperlink so patients can tap to call:
    #   <a href='tel:{emer_no}' style='color:{GREEN};'>{emer_no}</a>
    st.markdown(f"""
<div style='margin:14px 20px 0 20px;background:{CARD};border:1px solid {BORDER};
            border-radius:12px;padding:14px;'>
  <div style='color:{SUBTEXT};font-size:10px;font-weight:600;
              letter-spacing:0.05em;'>🚨 EMERGENCY CONTACT</div>
  <div style='color:{TEXT};font-size:15px;font-weight:700;margin-top:6px;'>{emer}</div>
  <div style='color:{SUBTEXT};font-size:13px;margin-top:2px;'>{emer_no}</div>
</div>
""", unsafe_allow_html=True)

    # ── 7. LIVE INDICATOR ──────────────────────────────────────────────────────
    # A small green glowing dot with "Live monitoring active" text.
    # This reassures the patient that the system is running and not frozen.
    # HOW TO CHANGE: to show the actual connection status (e.g., if the Pi is
    # unreachable), add logic to check whether the timestamp is stale (more
    # than ~10 s old) and change the dot colour to AMBER or RED accordingly.
    st.markdown(f"""
<div style='display:flex;align-items:center;padding:14px 20px 20px 20px;gap:8px;'>
  <span style='display:inline-block;width:8px;height:8px;border-radius:50%;
               background:{GREEN};box-shadow:0 0 6px {GREEN};'></span>
  <span style='color:{MUTED};font-size:11px;'>Live monitoring active</span>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 11 — ENTRY POINT
#
# WHY: Calling patient_view() at module level is how Streamlit executes the UI.
# Streamlit re-runs the entire script from top to bottom on every interaction,
# but because patient_view() is decorated with @st.fragment, its internal
# re-runs happen independently at the 2-second interval.
#
# HOW TO CHANGE: To add a login gate (e.g., patient enters their ID before
# seeing their data), wrap this call in an if/else that checks
# st.session_state for an authenticated flag.
# =============================================================================
patient_view()
