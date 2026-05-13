# =============================================================================
# sms_notifier.py  —  NeuroWatch Twilio SMS Notification Module
#
# Used by BOTH dashboard.py and main_pi_RPI_GPIO_dashboard.py
#
# Setup:
#   1. pip install twilio
#   2. Sign up at https://www.twilio.com → get Account SID, Auth Token, phone number
#   3. Fill in YOUR_TWILIO_* values below  (or set as environment variables)
# =============================================================================

# === SECTION 1 — MODULE PURPOSE AND ARCHITECTURE ==============================
#
# This module is the single place in NeuroWatch responsible for all outbound
# SMS communication. No other file should directly call Twilio — they all go
# through send_sms_alert() or send_sms_alert_async() defined here.
#
# Callers:
#   dashboard.py                     — calls send_sms_alert_async() from the UI thread
#   main_pi_RPI_GPIO_dashboard.py    — calls send_sms_alert_async() from the Pi loop
#
# Data flow for a single alert:
#   1. Caller calls send_sms_alert_async(...)
#   2. A new daemon thread is spawned — the calling thread returns immediately
#   3. In the background thread, send_sms_alert() runs:
#       a. Checks credentials aren't placeholders
#       b. Checks the rate-limiter (cooldown) for this patient+alert type
#       c. Builds the message text via _build_message()
#       d. Loops over RECIPIENT_NUMBERS, calls Twilio API for each
#       e. On success, records timestamp in _last_sent for future cooldown checks
#
# WHY ASYNC: The Twilio API call can take 1-3 seconds. On the Pi, the inference
# loop runs every ~1 second. Blocking it with a synchronous SMS send would
# cause missed EEG windows and delayed seizure detection. Running SMS in a
# daemon thread means the loop continues uninterrupted.
#
# === SECTION 2 — IMPORTS ======================================================

import os
# os.getenv() lets us read credentials from environment variables instead of
# hardcoding them. This is the safe practice for production deployments:
# set the variables in a .env file or the shell, never commit real credentials.

import time
# time.time() returns the current Unix timestamp (seconds since 1970-01-01).
# We use it to record WHEN the last SMS was sent and calculate how long ago it was.

import threading
# threading.Lock() — a mutex that prevents two threads from reading/writing
#   _last_sent at the same time. Without this, two simultaneous seizure alerts
#   could both pass the cooldown check and both send — doubling the SMS cost.
# threading.Thread() — used by the async wrapper to fire send_sms_alert() in
#   a background thread so the caller never blocks.

from typing import Dict, Tuple
# Dict and Tuple are type hints — they make the code easier to read and let
# IDEs (like VS Code) catch type errors. They have no runtime effect.
# Dict[Tuple, float] means: a dictionary where keys are tuples and values are floats.

from datetime import datetime
# datetime.now().strftime() formats the current time as "HH:MM:SS" for SMS messages.
# This shows the doctor the EXACT moment the alert was triggered.


# === SECTION 3 — TWILIO CREDENTIALS ==========================================
#
# SECURITY NOTE: These credentials are currently hardcoded for development.
# For a real hospital deployment you should REMOVE the hardcoded fallback
# values and use environment variables ONLY:
#
#   Linux/Mac:   export TWILIO_ACCOUNT_SID="ACxxx..."
#   Windows:     $env:TWILIO_ACCOUNT_SID = "ACxxx..."
#   .env file:   use python-dotenv (pip install python-dotenv) to load them
#
# The os.getenv("VAR", "fallback") pattern means:
#   - If the environment variable exists → use it  (production path)
#   - If not → use the hardcoded string (development fallback)
#
# WHERE TO FIND THESE VALUES:
#   Log in to https://www.twilio.com/console
#   Account SID  — top of the console dashboard, starts with "AC"
#   Auth Token   — revealed by clicking the eye icon on the dashboard
#   Phone Number — under Phone Numbers → Manage → Active Numbers
#
# HOW TO CHANGE:
#   Replace the hardcoded strings with your own Twilio credentials.
#   Or set environment variables and leave the second argument to os.getenv as "".

TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID",  "AC1698dea611ff3dfdf5229ab0c5ced6d8")
# Your Twilio Account SID — identifies your account. Safe to expose in logs
# but should not be committed to a public repository.

TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN",   "3c8684b22cce16c0af83a77eddf4be6c")
# Your Twilio Auth Token — the SECRET password for your account. NEVER share
# this or commit it to a public repository. Rotate it immediately at
# https://www.twilio.com/console if it leaks.

TWILIO_FROM_NUMBER  = os.getenv("TWILIO_FROM_NUMBER",  "+14786075237")
# The Twilio phone number that SMS messages are sent FROM.
# Must be a number you own in your Twilio account (international format).
# This is what the doctor sees as the sender number on their phone.


# === SECTION 4 — RECIPIENT PHONE NUMBERS =====================================
#
# RECIPIENT_NUMBERS is a list of phone numbers that will EACH receive a copy
# of every SMS alert. You can have as many recipients as you like — a neurology
# nurse, a senior registrar, an on-call doctor, etc.
#
# HOW TO CHANGE:
#   Option A — Edit the environment variable (recommended for deployment):
#     Set NEUROWATCH_ALERT_NUMBERS="+26771234567, +26779876543" in your shell
#     or .env file. Separate multiple numbers with commas.
#
#   Option B — Edit the fallback string below directly:
#     Change "+26774390351, +26771847749" to your recipients' numbers.
#     All numbers must be in international format: +<country code><number>
#     Example: Botswana = +267, UK = +44, US = +1
#
# HOW THE PARSING WORKS:
#   os.getenv(...) returns the string, e.g. "+26774390351, +26771847749"
#   .split(",")    splits on commas → ["+26774390351", " +26771847749"]
#   n.strip()      removes leading/trailing spaces from each number
#   if n.strip()   skips empty strings (handles trailing commas gracefully)

RECIPIENT_NUMBERS = [
    n.strip()
    for n in os.getenv("NEUROWATCH_ALERT_NUMBERS", "+26774390351, +26771847749").split(",")
    if n.strip()
]


# === SECTION 5 — RATE LIMITING (COOLDOWN) =====================================
#
# The Problem: In a seizure event, the inference loop might classify the same
# patient as "seizure" on 30 consecutive EEG windows (30 seconds at 1 Hz).
# Without a rate-limiter, that would send 30 SMS messages and cost you 30x
# the SMS fee — and overwhelm the doctor's phone.
#
# The Solution: After sending an SMS, record the timestamp. Block any further
# SMS of the same type for the same patient until the cooldown period expires.
#
# COOLDOWN_SECONDS maps alert_type → seconds to wait before sending again.
# Key: one of "seizure", "preseizure", or "normal"
# Value: minimum seconds between consecutive SMS of that type for ONE patient.
#
# HOW TO CHANGE:
#   Increase a value to reduce SMS frequency (cheaper, but doctor notified later).
#   Decrease a value to alert faster (more expensive, more notifications).
#   Add new keys if you add new alert types (e.g. "battery_low": 3600).

COOLDOWN_SECONDS = {
    "seizure":    120,   # 2 minutes — seizures are urgent but 30 texts in 30s is excessive
    "preseizure":  300,   # 5 minutes — pre-seizure is a warning; less time-critical
    "normal":      600,   # 10 minutes — "all clear" is informational, very low urgency
}

# _last_sent stores the last send timestamp for each (patient_id, alert_type) pair.
# Example contents after a seizure alert for P001:
#   { ("P001", "seizure"): 1715260800.0 }
# The value is a Unix timestamp (float) from time.time().
# Starts empty; grows as alerts are sent; never persisted to disk (resets on restart).
_last_sent: Dict[Tuple, float] = {}

# _lock is a mutual-exclusion lock (mutex). It ensures that when two alerts arrive
# at almost the same time (possible in a multi-threaded dashboard), they don't both
# read _last_sent simultaneously and both decide "cooldown not active" and both send.
# Any thread that wants to read or write _last_sent must first acquire this lock.
_lock = threading.Lock()


# --- Helper: check if cooldown is still active --------------------------------
def _is_on_cooldown(patient_id: str, alert_type: str) -> bool:
    """
    Returns True if an SMS of this type was recently sent for this patient
    and the cooldown period has not yet elapsed.

    Parameters
    ----------
    patient_id  : str — e.g. "P001". Used as part of the dict key.
    alert_type  : str — "seizure", "preseizure", or "normal".

    HOW IT WORKS:
        key = (patient_id, alert_type)  e.g. ("P001", "seizure")
        Look up the last-sent timestamp for this key (default 0 = never sent).
        If time.time() - last_sent < cooldown, we're still in the cooldown window.

    Returns False if this is the first alert or if enough time has passed.
    """
    key     = (patient_id, alert_type)
    # Fall back to 300 seconds if the alert_type isn't in COOLDOWN_SECONDS.
    # This handles any future alert types you add without breaking the rate-limiter.
    cooldown = COOLDOWN_SECONDS.get(alert_type, 300)
    with _lock:
        # _last_sent.get(key, 0) returns 0 if this key has never been sent.
        # time.time() - 0 will always be > any cooldown, so "never sent" → not on cooldown.
        last = _last_sent.get(key, 0)
        return (time.time() - last) < cooldown


# --- Helper: record that an SMS was just sent ---------------------------------
def _mark_sent(patient_id: str, alert_type: str):
    """
    Record the current time as the last-sent timestamp for this patient+alert pair.
    Call this AFTER a successful send so the cooldown clock starts from now.

    WHY: If we recorded it BEFORE sending and the send failed, the cooldown
    would prevent the next retry even though nothing was actually delivered.
    """
    with _lock:
        # Store the current Unix timestamp. Future calls to _is_on_cooldown()
        # will compare against this value.
        _last_sent[(patient_id, alert_type)] = time.time()


# === SECTION 6 — MESSAGE TEMPLATES ============================================
#
# _build_message() constructs the SMS body text for each alert type.
# All strings use Python f-string formatting (the f"..." syntax) so that
# patient-specific values are inserted at the right place.
#
# HOW TO CHANGE THE MESSAGE TEXT:
#   Modify the return strings inside each if/elif block.
#   You can add or remove lines freely — just keep the f-string syntax correct.
#   {variable_name} inserts the value of a local variable.
#   {variable_name!r} would insert it with quotes — avoid that in SMS text.
#
# SMS LENGTH NOTE: Standard SMS is 160 characters. Longer messages are split
# into multi-part SMS (MMS) and cost more. The current templates exceed 160
# characters intentionally because the clinical detail is worth the extra cost.

def _build_message(alert_type: str, patient_name: str, ward: str,
                   confidence: float, patient_id: str,
                   location: str = "", maps_link: str = "",
                   emergency_contact: str = "") -> str:
    """
    Build the SMS body text for the given alert type and patient details.

    Parameters
    ----------
    alert_type        : "seizure", "preseizure", or "normal"
    patient_name      : Full name from PATIENT_REGISTRY
    ward              : Ward string from PATIENT_REGISTRY
    confidence        : Float 0.0–1.0 from the ML model's predict_proba() output
    patient_id        : e.g. "P001" — for doctor reference
    location          : Optional override for the bed/room location string
    maps_link         : Optional Google Maps URL generated from GPS coordinates
    emergency_contact : Name of next-of-kin from PATIENT_REGISTRY

    Returns
    -------
    str : The complete SMS message body ready to send.
    """
    # Format current time as HH:MM:SS for the SMS timestamp.
    # This uses the local system clock — make sure the Pi's clock is synced via NTP.
    ts   = datetime.now().strftime("%H:%M:%S")

    # Convert confidence from a decimal (e.g. 0.94) to a percentage integer (94).
    # int() truncates — so 0.947 becomes 94, not 95.
    conf = int(confidence * 100)

    # --- SEIZURE alert — highest urgency, immediate action required -----------
    if alert_type == "seizure":
        return (
            f"🚨 NEUROWATCH SEIZURE ALERT\n"
            f"Patient : {patient_name} ({patient_id})\n"   # Name + ID for quick identification
            f"Ward    : {ward}\n"                           # Where to go
            f"Time    : {ts}\n"                             # When it started
            f"Status  : EXPERIENCING A SEIZURE\n"
            f"Confidence: {conf}%\n"                        # How certain the ML model is (0-100%)
            f"Time    : {ts}\n"                             # Repeated — intentional for visibility
            f"⚡ Immediate attention required!\n"
            f"📍 Location: {location or ward}\n"            # Use detailed location if provided,
                                                            # fall back to ward name otherwise
            f"👤 Emergency: {emergency_contact}\n"          # Next-of-kin name for the doctor
            + (f"🗺 Map: {maps_link}" if maps_link else "") # Only append map line if link exists
        )

    # --- PRE-SEIZURE alert — warning, patient needs close monitoring ----------
    elif alert_type == "preseizure":
        return (
            f"⚠️ NEUROWATCH PRE-SEIZURE WARNING\n"
            f"Patient : {patient_name} ({patient_id})\n"
            f"Ward    : {ward}\n"
            f"Confidence: {conf}%\n"
            f"Time    : {ts}\n"
            f"Monitor patient closely.\n"                   # Less urgent than seizure
            f"📍 Location: {location or ward}\n"
            f"👤 Emergency: {emergency_contact}\n"
            + (f"🗺 Map: {maps_link}" if maps_link else "")
        )

    # --- NORMAL / CLEARED alert — episode has ended, informational only -------
    elif alert_type == "normal":
        return (
            f"✅ NEUROWATCH — Seizure Cleared\n"
            f"Patient : {patient_name} ({patient_id})\n"
            f"Ward    : {ward}\n"
            f"Time    : {ts}\n"
            f"Status  : Patient has returned to Normal.\n"
            f"Seizure episode has ended."
            # No location or map link needed — no action required from the doctor
        )

    # --- Fallback for any unrecognised alert_type ----------------------------
    else:
        # This branch handles future alert types you might add (e.g. "battery_low").
        # If you add a new alert type, add a matching elif block above rather than
        # relying on this generic fallback — it has no patient context details.
        return f"NeuroWatch alert ({alert_type}) for {patient_name} at {ts}."


# === SECTION 7 — CORE SEND FUNCTION ==========================================
#
# This is the main function you call (or that send_sms_alert_async calls for you).
# It orchestrates credential checking → rate-limit checking → message building
# → Twilio API call → cooldown recording.

def send_sms_alert(
    alert_type:        str,
    patient_name:      str,
    patient_id:        str,
    ward:              str,
    confidence:        float,
    location:          str  = "",
    maps_link:         str  = "",
    emergency_contact: str  = "",
    force:             bool = False,
) -> dict:
    """
    Send an SMS alert to all RECIPIENT_NUMBERS.

    Parameters
    ----------
    alert_type        : "seizure", "preseizure", or "normal"
    patient_name      : Full name for display in the SMS
    patient_id        : Unique patient ID used for rate-limiting key
    ward              : Ward/location string
    confidence        : ML model confidence, 0.0–1.0
    location          : Optional detailed location string (e.g. "Bed 1, Room 4A")
    maps_link         : Optional Google Maps URL with GPS coordinates
    emergency_contact : Name of next-of-kin shown in the SMS body
    force             : If True, bypass the cooldown check and send regardless.
                        Use this for testing or critical override situations.

    Returns
    -------
    dict with keys:
        "sent"           : bool — True if at least one SMS was delivered successfully
        "skipped_reason" : str or None — why the SMS was not sent (if applicable)
        "errors"         : list of str — any per-recipient error messages

    HOW TO CALL:
        result = send_sms_alert(
            alert_type="seizure",
            patient_name="John Molebatsi",
            patient_id="P001",
            ward="Neuro A",
            confidence=0.94,
        )
        if result["sent"]:
            print("SMS delivered!")
    """

    # --- 7a. Validate credentials aren't still placeholder text ---------------
    # If a developer copies this file and forgets to fill in their Twilio
    # credentials, the hardcoded values may say "YOUR_TWILIO_ACCOUNT_SID" etc.
    # This check catches that and prints a clear warning instead of crashing.
    # HOW TO FIX: Replace "YOUR_TWILIO_*" strings in SECTION 3 with real values.
    if "YOUR_TWILIO" in TWILIO_ACCOUNT_SID or "YOUR_TWILIO" in TWILIO_AUTH_TOKEN:
        print("⚠️  [SMS] Twilio credentials not configured — skipping SMS send.")
        return {"sent": False, "skipped_reason": "credentials_not_configured", "errors": []}

    # --- 7b. Rate-limit check — skip if too soon after last send --------------
    # force=True overrides this check — useful for the __main__ test block below
    # or for a manual "send now" button in the dashboard.
    if not force and _is_on_cooldown(patient_id, alert_type):
        print(f"⏳ [SMS] Cooldown active for {patient_name} / {alert_type} — skipped.")
        return {"sent": False, "skipped_reason": "cooldown", "errors": []}

    # --- 7c. Build the message body -------------------------------------------
    # Delegates to _build_message() in Section 6 which applies the right template.
    body = _build_message(alert_type, patient_name, ward, confidence,
                      patient_id, location, maps_link, emergency_contact)

    # --- 7d. Send via Twilio REST API -----------------------------------------
    # We import twilio.rest.Client here (inside the function) rather than at the
    # top of the file. This means the module can still be imported even if the
    # twilio package is not installed — only calling send_sms_alert() will fail.
    errors = []
    any_sent = False

    try:
        from twilio.rest import Client
        # Authenticate with your Twilio account using SID + Token.
        # The Client object manages the HTTPS connection to Twilio's servers.
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        # Loop over every recipient in RECIPIENT_NUMBERS (defined in Section 4).
        # Each recipient gets their own API call and their own try/except so that
        # a failure to reach one doctor does NOT prevent the others from being notified.
        for number in RECIPIENT_NUMBERS:
            try:
                # client.messages.create() makes the actual HTTPS POST to Twilio.
                # It returns a Message object; we only use msg.sid for logging.
                # msg.sid is Twilio's unique ID for this message — useful for
                # looking up delivery status in the Twilio console later.
                msg = client.messages.create(
                    body=body,                   # The SMS text built in step 7c
                    from_=TWILIO_FROM_NUMBER,    # Your Twilio number (sender)
                    to=number,                   # This recipient's number
                )
                print(f"✅ [SMS] Sent {alert_type} alert for {patient_name} → {number} (SID: {msg.sid})")
                any_sent = True  # At least one message got through
            except Exception as e:
                # A per-recipient failure (e.g. invalid number, network timeout).
                # Log it, add to errors list, and continue to the next recipient.
                err = f"Failed to send to {number}: {e}"
                print(f"❌ [SMS] {err}")
                errors.append(err)

    except ImportError:
        # The twilio Python package is not installed at all.
        # Tell the developer exactly how to fix it and return early.
        err = "twilio package not installed. Run: pip install twilio"
        print(f"❌ [SMS] {err}")
        errors.append(err)
        return {"sent": False, "skipped_reason": "twilio_not_installed", "errors": errors}

    # --- 7e. Record the send timestamp for future cooldown checks -------------
    # Only do this if at least one message was successfully delivered.
    # If ALL sends failed, we don't start the cooldown — the next alert attempt
    # should try again immediately rather than waiting for a cooldown to expire.
    if any_sent:
        _mark_sent(patient_id, alert_type)

    # Return a structured result so the caller can decide what to log or display.
    return {"sent": any_sent, "skipped_reason": None, "errors": errors}


# === SECTION 8 — ASYNC (NON-BLOCKING) WRAPPER =================================
#
# This is the function that dashboard.py and the Pi engine should call in normal
# operation.  It fires send_sms_alert() in a background thread and returns
# immediately, so the UI or inference loop is never blocked waiting for a network call.
#
# WHY DAEMON THREAD:
#   daemon=True means this thread is automatically killed when the main program exits.
#   Without this, the program would hang for 30 seconds waiting for an in-flight
#   Twilio call to finish every time you press Ctrl+C or the Pi reboots.

def send_sms_alert_async(alert_type: str, patient_name: str, patient_id: str,
                          ward: str, confidence: float,
                          location: str = "", maps_link: str = "",
                          emergency_contact: str = "",
                          force: bool = False):
    """
    Non-blocking version of send_sms_alert(). Safe to call from dashboard or Pi loop.

    Starts a daemon background thread and returns immediately.
    The caller cannot know the result — check logs or use send_sms_alert() directly
    if you need the return value (e.g. for a confirmation popup in the dashboard).

    Parameters: same as send_sms_alert() — see that function's docstring.

    HOW TO CALL (typical usage in the Pi engine or dashboard):
        send_sms_alert_async(
            alert_type="seizure",
            patient_name=patient["name"],
            patient_id=patient["id"],
            ward=patient["ward"],
            confidence=pred_confidence,
            location=patient["gps_label"],
            maps_link=f"https://maps.google.com/?q={patient['gps_lat']},{patient['gps_lng']}",
            emergency_contact=patient["emergency_contact"],
        )
        # Returns immediately; SMS is sent in the background.
    """
    # Create a Thread that targets send_sms_alert() and passes all arguments via args tuple.
    # All positional arguments must match the order in send_sms_alert()'s signature exactly.
    t = threading.Thread(
        target=send_sms_alert,
        args=(alert_type, patient_name, patient_id, ward,
              confidence, location, maps_link, emergency_contact, force),
        daemon=True,   # auto-killed on program exit — prevents hanging shutdown
    )
    # Start the thread. From this point, send_sms_alert() runs in parallel.
    t.start()
    # We don't call t.join() — that would block. We deliberately let it run freely.


# === SECTION 9 — QUICK TEST BLOCK =============================================
#
# Running this file directly (python sms_notifier.py) triggers this block.
# It sends one test SMS for each alert type with force=True to bypass cooldown.
# Use this to verify your Twilio credentials and recipient numbers work before
# deploying to the Pi.
#
# HOW TO RUN:
#   cd <project root>
#   python src/sms_notifier.py
#
# EXPECTED OUTPUT (if credentials are correct):
#   ✅ [SMS] Sent seizure alert for Test Patient → +2677XXXXXXX (SID: SMxxx...)
#   Result: {'sent': True, 'skipped_reason': None, 'errors': []}
#
# If you see "Twilio credentials not configured", fill in Section 3.
# If you see "twilio package not installed", run: pip install twilio

if __name__ == "__main__":
    print("🧪 NeuroWatch SMS Notifier — Test Mode\n")

    # Loop over all three alert types and fire a test message for each.
    for alert in ["seizure", "preseizure", "normal"]:
        print(f"Sending test '{alert}' alert...")
        result = send_sms_alert(
            alert_type=alert,
            patient_name="Test Patient",
            patient_id="P000",            # P000 is a dummy ID that won't affect real cooldowns
            ward="Test Ward",
            confidence=0.92,
            force=True,                   # bypass cooldown so all three send even if run quickly
        )
        print(f"Result: {result}\n")
        time.sleep(2)   # small pause between sends to avoid Twilio rate limits
