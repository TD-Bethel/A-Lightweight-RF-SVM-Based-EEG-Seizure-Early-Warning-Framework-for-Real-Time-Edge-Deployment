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

PATIENT_REGISTRY = [
    # ── LIVE PATIENT — real ML predictions, GPIO, SMS ──────────────────────
    {
        "id":               "P001",
        "name":             "John Molebatsi",
        "age":              34,
        "diagnosis":        "Temporal Lobe Epilepsy",
        "ward":             "Neuro A",
        "bed":              "Bed 1",
        "emergency_contact":"Kelebogile Molebatsi",
        "emergency_phone":  "+26771000001",
        "gps_lat":          -23.9571,
        "gps_lng":          26.8368,
        "gps_label":        "Neuro Ward A, Bed 1, Princess Marina Hospital",
        "eeg_slice":        (0, 20),
        "live":             True,   # ← ONLY this patient drives real hardware
    },
    # ── DEMO PATIENTS — simulated states, no hardware alerts ───────────────
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
        "eeg_slice":        (20, 40),
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
        "eeg_slice":        (40, 60),
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
        "eeg_slice":        (60, 80),
        "live":             False,
    },
    {
        "id":               "P005",
        "name":             "Kefilwe Nthebe",
        "age":              52,
        "diagnosis":        "Post-Traumatic Epilepsy",
        "ward":             "ICU",
        "bed":              "Bed 3",
        "emergency_contact":"Dintle Nthebe",
        "emergency_phone":  "+26771000005",
        "gps_lat":          -23.9580,
        "gps_lng":          26.8376,
        "gps_label":        "ICU, Bed 3, Princess Marina Hospital",
        "eeg_slice":        (80, 100),
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
        "eeg_slice":        (0, 20),
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
        "eeg_slice":        (20, 40),
        "live":             False,
    },
]
