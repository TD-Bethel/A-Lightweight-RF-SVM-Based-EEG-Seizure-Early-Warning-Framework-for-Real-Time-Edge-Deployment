# =============================================================================
# patients.py  —  NeuroWatch Patient Registry
#
# LIVE vs DEMO:
#   "live": True  → This patient is driven by real ML predictions from the
#                   EEG dataset. GPIO, LCD, buzzer and SMS all react to this
#                   patient's state.  Only ONE patient should be live=True.
#
#   "live": False → Demo patient. State cycles through a realistic random
#                   simulation so the dashboard ward looks populated, but
#                   no hardware alerts or SMS fire for demo patients.
#
# GPS: Replace with real coordinates from a GPS module (e.g. Neo-6M).
# =============================================================================

# === SECTION 1 — PURPOSE AND HOW THIS FILE IS USED ===========================
#
# PATIENT_REGISTRY is a plain Python list of dicts.  Every script in
# NeuroWatch that needs to know about patients imports this list:
#
#   from patients import PATIENT_REGISTRY      (from src/ or root depending on script)
#
# Then loops over it:
#   for patient in PATIENT_REGISTRY:
#       print(patient["name"], patient["ward"])
#
# The dashboard (dashboard.py) and the Pi engine (main_pi_*.py) both read this
# list at startup to populate the ward panel and decide which patient gets real
# hardware alerts.
#
# === HOW TO ADD A NEW PATIENT ==================================================
#
# Copy any existing dict block (between the outer { } braces), paste it below
# the last entry (inside the outer list [ ]), and fill in the fields.
# Key rules:
#   - "id" must be unique (P008, P009, etc.)
#   - "live" should be False for the new patient unless you want it to drive
#     real GPIO/SMS — only ONE patient can be live at a time.
#   - "eeg_slice" (start, end): which rows of the loaded EEG dataset this
#     patient's demo simulation draws from. Keep ranges non-overlapping and
#     within the total row count of your dataset.
#   - GPS coordinates: use Google Maps → right-click a location → copy lat/lng.
#
# === HOW TO CHANGE THE LIVE PATIENT ============================================
#
# 1. Find the entry with "live": True and change it to "live": False.
# 2. Find the patient you want to make live and change "live": False → True.
# 3. Only ONE patient should ever have "live": True.
#
# === SECTION 2 — THE REGISTRY LIST ============================================

PATIENT_REGISTRY = [

    # --------------------------------------------------------------------------
    # LIVE PATIENT — real ML predictions, GPIO buzzer, LCD display, and SMS
    # --------------------------------------------------------------------------
    # This is the patient whose EEG the Pi is actually monitoring in real time.
    # The main_pi_*.py engine checks the "live" flag on every patient and feeds
    # ONLY this patient's data through the ML pipeline.  All hardware outputs
    # (GPIO seizure pin, LCD text, buzzer, Twilio SMS) are triggered by this
    # patient's predictions.
    {
        "id":               "P001",          # Unique patient ID used as the key in logs,
                                              # SMS rate-limiter, and JSON metrics file.
                                              # Change this if you need a different ID scheme.

        "name":             "John Molebatsi", # Full display name — appears on the dashboard
                                              # and in SMS messages sent to doctors.

        "age":              34,               # Age in years — shown on the dashboard card.
                                              # No functional effect on the ML pipeline.

        "diagnosis":        "Temporal Lobe Epilepsy",
                                              # Clinical diagnosis — displayed on the
                                              # dashboard for context. Free-text string.

        "ward":             "Neuro A",        # Ward name — displayed on dashboard and
                                              # included in SMS alerts so the doctor knows
                                              # where to go. Must match the ward label used
                                              # in your hospital floor plan if you have one.

        "bed":              "Bed 1",          # Bed number within the ward.
                                              # Used alongside "ward" in SMS messages.

        "emergency_contact":"Kelebogile Molebatsi",
                                              # Name of the patient's next-of-kin or
                                              # designated emergency contact. Displayed in
                                              # the SMS alert body so the responding doctor
                                              # knows who to call if needed.

        "emergency_phone":  "+26771000001",   # Phone number of the emergency contact.
                                              # NOT the same as RECIPIENT_NUMBERS in
                                              # sms_notifier.py — this is shown in the
                                              # SMS text, not the destination of the SMS.
                                              # Use international format: +<country><number>

        "gps_lat":          -23.9571,         # Latitude of this patient's bed.
        "gps_lng":          26.8368,          # Longitude of this patient's bed.
                                              # Together these form a Google Maps link in
                                              # the SMS alert so the doctor can navigate.
                                              # Replace with real coordinates from a GPS
                                              # module (e.g. u-blox Neo-6M) if available,
                                              # or look up the exact room on Google Maps.

        "gps_label":        "Neuro Ward A, Bed 1, Princess Marina Hospital",
                                              # Human-readable location label shown in the
                                              # SMS alongside the map link. Keep it short
                                              # and clear so a doctor can act on it quickly.

        "eeg_slice":        (0, 20),          # Row range (start, end) within the loaded
                                              # EEG dataset that the DEMO simulation for
                                              # this patient draws from.
                                              # For the live patient this field is only used
                                              # if the Pi falls back to demo mode.
                                              # Make sure ranges don't overlap between patients
                                              # and stay within the dataset row count.

        "live":             True,             # ← THE KEY FLAG.
                                              # True  = this patient drives real hardware.
                                              # False = demo/simulation only.
                                              # ONLY ONE patient should have live=True.
                                              # The Pi engine searches the list for the first
                                              # patient where live==True.
    },

    # --------------------------------------------------------------------------
    # DEMO PATIENTS — simulated states only, no hardware alerts fired
    # --------------------------------------------------------------------------
    # These patients exist so the dashboard ward panel looks realistically
    # populated during demos or development.  Their "state" (normal / pre-seizure
    # / seizure) cycles through a random simulation driven by demo_tick() in the
    # main engine — they never touch GPIO, the buzzer, or Twilio SMS.
    #
    # To turn any demo patient into a real monitored patient:
    #   1. Set their "live" to True
    #   2. Set ALL other patients' "live" to False
    # --------------------------------------------------------------------------

    {
        "id":               "P002",
        "name":             "Mpho Setlhare",
        "age":              27,
        "diagnosis":        "Juvenile Myoclonic Epilepsy",
        "ward":             "Neuro A",
        "bed":              "Bed 2",
        "emergency_contact":"Tshepo Setlhare",
        "emergency_phone":  "+26771000002",
        "gps_lat":          -23.9572,
        "gps_lng":          26.8370,
        "gps_label":        "Neuro Ward A, Bed 2, Princess Marina Hospital",
        "eeg_slice":        (20, 40),  # rows 20-39 of the EEG dataset used for demo simulation
        "live":             False,
    },
    {
        "id":               "P003",
        "name":             "Boitumelo Kgosi",
        "age":              45,
        "diagnosis":        "Focal Cortical Dysplasia",
        "ward":             "Neuro B",
        "bed":              "Bed 1",
        "emergency_contact":"Lebo Kgosi",
        "emergency_phone":  "+26771000003",
        "gps_lat":          -23.9575,
        "gps_lng":          26.8372,
        "gps_label":        "Neuro Ward B, Bed 1, Princess Marina Hospital",
        "eeg_slice":        (40, 60),  # rows 40-59 — non-overlapping with P002
        "live":             False,
    },
    {
        "id":               "P004",
        "name":             "Oratile Tau",
        "age":              19,
        "diagnosis":        "Absence Epilepsy",
        "ward":             "Neuro B",
        "bed":              "Bed 2",
        "emergency_contact":"Neo Tau",
        "emergency_phone":  "+26771000004",
        "gps_lat":          -23.9576,
        "gps_lng":          26.8374,
        "gps_label":        "Neuro Ward B, Bed 2, Princess Marina Hospital",
        "eeg_slice":        (60, 80),  # rows 60-79
        "live":             False,
    },
    {
        "id":               "P005",
        "name":             "Kefilwe Nthebe",
        "age":              52,
        "diagnosis":        "Post-Traumatic Epilepsy",
        "ward":             "ICU",     # Different ward — shows the system can span multiple wards
        "bed":              "Bed 3",
        "emergency_contact":"Dintle Nthebe",
        "emergency_phone":  "+26771000005",
        "gps_lat":          -23.9580,
        "gps_lng":          26.8376,
        "gps_label":        "ICU, Bed 3, Princess Marina Hospital",
        "eeg_slice":        (80, 100), # rows 80-99
        "live":             False,
    },
    {
        "id":               "P006",
        "name":             "Tshepiso Motlhabane",
        "age":              38,
        "diagnosis":        "Generalised Epilepsy",
        "ward":             "Neuro C",
        "bed":              "Bed 1",
        "emergency_contact":"Refilwe Motlhabane",
        "emergency_phone":  "+26771000006",
        "gps_lat":          -23.9582,
        "gps_lng":          26.8378,
        "gps_label":        "Neuro Ward C, Bed 1, Princess Marina Hospital",
        "eeg_slice":        (0, 20),   # Reuses rows 0-19 — fine because this is demo only
                                       # and runs independently of the live patient's real data
        "live":             False,
    },
    {
        "id":               "P007",
        "name":             "Gaone Segwabe",
        "age":              61,
        "diagnosis":        "Lennox-Gastaut Syndrome",
        "ward":             "Neuro C",
        "bed":              "Bed 2",
        "emergency_contact":"Mothusi Segwabe",
        "emergency_phone":  "+26771000007",
        "gps_lat":          -23.9583,
        "gps_lng":          26.8380,
        "gps_label":        "Neuro Ward C, Bed 2, Princess Marina Hospital",
        "eeg_slice":        (20, 40),  # Reuses rows 20-39 for demo simulation
        "live":             False,
    },
]

# === SECTION 3 — QUICK REFERENCE FOR SCRIPTS THAT IMPORT THIS FILE ============
#
# Get the live patient (the one the Pi is actually monitoring):
#   live_patient = next(p for p in PATIENT_REGISTRY if p["live"])
#
# Get all demo patients:
#   demo_patients = [p for p in PATIENT_REGISTRY if not p["live"]]
#
# Look up a patient by ID:
#   patient = next((p for p in PATIENT_REGISTRY if p["id"] == "P003"), None)
#
# Get all unique wards:
#   wards = list({p["ward"] for p in PATIENT_REGISTRY})
#
# Build a Google Maps link for a patient:
#   maps_link = f"https://maps.google.com/?q={p['gps_lat']},{p['gps_lng']}"
