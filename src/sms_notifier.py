# ─────────────────────────────────────────────────────────────────────────────
# sms_notifier.py  —  NeuroWatch Twilio SMS Notification Module
#
# Used by BOTH dashboard.py and main_pi_RPI_GPIO_dashboard.py
#
# Setup:
#   1. pip install twilio
#   2. Sign up at https://www.twilio.com → get Account SID, Auth Token, phone number
#   3. Fill in YOUR_TWILIO_* values below  (or set as environment variables)
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import threading
from typing import Dict, Tuple
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# !! CONFIGURE YOUR TWILIO CREDENTIALS HERE !!
# Or set environment variables: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
#                               TWILIO_FROM_NUMBER, NEUROWATCH_ALERT_NUMBERS
# ─────────────────────────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID",  "YOUR_TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN",   "YOUR_TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER  = os.getenv("TWILIO_FROM_NUMBER",  "YOUR_TWILIO_FROM_NUMBER")

# Add every recipient number here (international format, e.g. "+26771234567")
RECIPIENT_NUMBERS = [
    n.strip()
    for n in os.getenv("NEUROWATCH_ALERT_NUMBERS", "+1234567890,+1987654321").split(",")
    if n.strip()
]

# ─────────────────────────────────────────────────────────────────────────────
# RATE-LIMITING  — prevents SMS flood for the same patient/event type
# Key: (patient_id, alert_type)  →  last sent timestamp
# ─────────────────────────────────────────────────────────────────────────────
COOLDOWN_SECONDS = {
    "seizure":    120,   # 2 min between seizure alerts per patient
    "preseizure":  300,   # 5 min between pre-seizure alerts per patient
    "normal":      600,   # 10 min between "back to normal" alerts per patient
}

_last_sent: Dict[Tuple, float] = {}
_lock = threading.Lock()


def _is_on_cooldown(patient_id: str, alert_type: str) -> bool:
    key     = (patient_id, alert_type)
    cooldown = COOLDOWN_SECONDS.get(alert_type, 300)
    with _lock:
        last = _last_sent.get(key, 0)
        return (time.time() - last) < cooldown


def _mark_sent(patient_id: str, alert_type: str):
    with _lock:
        _last_sent[(patient_id, alert_type)] = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# MESSAGE TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

def _build_message(alert_type: str, patient_name: str, ward: str,
                   confidence: float, patient_id: str,
                   location: str = "", maps_link: str = "",
                   emergency_contact: str = "") -> str:
    ts   = datetime.now().strftime("%H:%M:%S")
    conf = int(confidence * 100)

    if alert_type == "seizure":
        return (
            f"🚨 NEUROWATCH SEIZURE ALERT\n"
            f"Patient : {patient_name} ({patient_id})\n"
            f"Ward    : {ward}\n"
            f"Confidence: {conf}%\n"
            f"Time    : {ts}\n"
            f"⚡ Immediate attention required!\n"
            f"📍 Location: {location or ward}\n"
            f"👤 Emergency: {emergency_contact}\n"
            + (f"🗺 Map: {maps_link}" if maps_link else "")
        )
    elif alert_type == "preseizure":
        return (
            f"⚠️ NEUROWATCH PRE-SEIZURE WARNING\n"
            f"Patient : {patient_name} ({patient_id})\n"
            f"Ward    : {ward}\n"
            f"Confidence: {conf}%\n"
            f"Time    : {ts}\n"
            f"Monitor patient closely.\n"
            f"📍 Location: {location or ward}\n"
            f"👤 Emergency: {emergency_contact}\n"
            + (f"🗺 Map: {maps_link}" if maps_link else "")
        )
    elif alert_type == "normal":
        return (
            f"✅ NEUROWATCH — Patient Stable\n"
            f"Patient : {patient_name} ({patient_id})\n"
            f"Ward    : {ward}\n"
            f"Time    : {ts}\n"
            f"Status returned to Normal."
        )
    else:
        return f"NeuroWatch alert ({alert_type}) for {patient_name} at {ts}."


# ─────────────────────────────────────────────────────────────────────────────
# CORE SEND FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

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

    Returns a result dict:
        { "sent": bool, "skipped_reason": str|None, "errors": list }
    """

    # ── Validate credentials aren't placeholders ──────────────────────────
    if "YOUR_TWILIO" in TWILIO_ACCOUNT_SID or "YOUR_TWILIO" in TWILIO_AUTH_TOKEN:
        print("⚠️  [SMS] Twilio credentials not configured — skipping SMS send.")
        return {"sent": False, "skipped_reason": "credentials_not_configured", "errors": []}

    # ── Rate-limit check ──────────────────────────────────────────────────
    if not force and _is_on_cooldown(patient_id, alert_type):
        print(f"⏳ [SMS] Cooldown active for {patient_name} / {alert_type} — skipped.")
        return {"sent": False, "skipped_reason": "cooldown", "errors": []}

    # ── Build message ─────────────────────────────────────────────────────
    body = _build_message(alert_type, patient_name, ward, confidence,
                      patient_id, location, maps_link, emergency_contact)

    # ── Send via Twilio ───────────────────────────────────────────────────
    errors = []
    any_sent = False

    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        for number in RECIPIENT_NUMBERS:
            try:
                msg = client.messages.create(
                    body=body,
                    from_=TWILIO_FROM_NUMBER,
                    to=number,
                )
                print(f"✅ [SMS] Sent {alert_type} alert for {patient_name} → {number} (SID: {msg.sid})")
                any_sent = True
            except Exception as e:
                err = f"Failed to send to {number}: {e}"
                print(f"❌ [SMS] {err}")
                errors.append(err)

    except ImportError:
        err = "twilio package not installed. Run: pip install twilio"
        print(f"❌ [SMS] {err}")
        errors.append(err)
        return {"sent": False, "skipped_reason": "twilio_not_installed", "errors": errors}

    if any_sent:
        _mark_sent(patient_id, alert_type)

    return {"sent": any_sent, "skipped_reason": None, "errors": errors}


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC WRAPPER — fire-and-forget so it never blocks the UI or Pi loop
# ─────────────────────────────────────────────────────────────────────────────

def send_sms_alert_async(alert_type: str, patient_name: str, patient_id: str,
                          ward: str, confidence: float,
                          location: str = "", maps_link: str = "",
                          emergency_contact: str = "",
                          force: bool = False):
    """Non-blocking version. Safe to call from dashboard or Pi loop."""
    t = threading.Thread(
        target=send_sms_alert,
        args=(alert_type, patient_name, patient_id, ward,
              confidence, location, maps_link, emergency_contact, force),
        daemon=True,
    )
    t.start()


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — run this file directly to verify your setup
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🧪 NeuroWatch SMS Notifier — Test Mode\n")

    for alert in ["seizure", "preseizure", "normal"]:
        print(f"Sending test '{alert}' alert...")
        result = send_sms_alert(
            alert_type=alert,
            patient_name="Test Patient",
            patient_id="P000",
            ward="Test Ward",
            confidence=0.92,
            force=True,          # bypass cooldown for test
        )
        print(f"Result: {result}\n")
        time.sleep(2)
