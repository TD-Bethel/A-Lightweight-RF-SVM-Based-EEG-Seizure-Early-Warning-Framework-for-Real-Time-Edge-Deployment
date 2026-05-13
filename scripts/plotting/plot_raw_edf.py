"""
plot_raw_edf.py  —  Visualise raw Mendeley EDF recordings before preprocessing.

Shows all 19 EEG channels stacked for a chosen patient / record, with coloured
bands marking the labelled regions:
    green  = Normal  (≥60 min from any seizure)
    orange = Pre-ictal (5–25 min before onset)
    red    = Ictal   (during seizure)

USAGE:
    python scripts/plotting/plot_raw_edf.py              # interactive picker
    python scripts/plotting/plot_raw_edf.py --patient 10 --record 1
    python scripts/plotting/plot_raw_edf.py --list       # just list all files
"""

import os, sys, argparse, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, date, time as dtime
warnings.filterwarnings("ignore")

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    print("ERROR: mne not installed  —  pip install mne")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EDF_DIR  = os.path.join(BASE_DIR, "data", "Mendelay dataset", "Raw_EDF_Files")

FS = 500          # Hz
PREICTAL_START = 25 * 60   # seconds before onset  (–25 min)
PREICTAL_END   =  5 * 60   # seconds before onset  (–5  min)
NORMAL_DIST    = 60 * 60   # seconds (≥60 min from seizure = Normal)

# ── Same database used by preprocess_preictal.py ──────────────────────────────
SEIZURE_DB = {
    10: {1: [(7,36,38,445)],
         2: [(6,29,14,305)]},
    11: {1: [(15,8,55,64)],
         2: [(18,11,38,51)],
         3: [(19,18,38,91),(20,6,38,83),(20,53,22,76),(21,27,24,73)],
         4: [(23,55,35,1358)]},
    12: {1: [(1,56,14,76),(2,20,12,104)],
         2: [(5,50,55,118)],
         3: [(6,40,30,82),(7,21,36,164),(8,37,48,113)]},
    13: {1: [(2,31,9,52)],
         2: [(3,33,4,25),(4,38,59,16)],
         3: [(6,45,51,18)],
         4: [(10,51,41,30),(12,18,22,24)]},
    14: {1: [(14,32,2,28),(15,34,32,134)],
         2: [(16,20,58,32),(17,50,56,10)],
         3: [(20,20,46,31),(21,2,4,26),(21,27,49,40),(21,50,24,40)]},
    15: {1: [(17,18,8,50)],
         2: [(22,49,24,47)],
         3: [(2,57,4,13)],
         4: [(5,3,26,56),(6,23,29,20)]},
}

PATIENT_FILES = {
    10: ["Record1.edf", "Record2.edf"],
    11: ["Record1.edf", "Record2.edf", "Record3.edf", "Record4.edf"],
    12: ["Record1.edf", "Record2.edf", "Record3.edf"],
    13: ["Record1.edf", "Record2.edf", "Record3.edf", "Record4.edf"],
    14: ["Record1.edf", "Record2.edf", "Record3.edf"],
    15: ["Record1.edf", "Record2.edf", "Record3.edf", "Record4.edf"],
}

P10_CHANNELS = [
    'EEG Fp2-Ref','EEG Fp1-Ref','EEG F8-Ref','EEG F4-Ref','EEG Fz-Ref',
    'EEG F3-Ref','EEG F7-Ref','EEG A2-Ref','EEG T4-Ref','EEG C4-Ref',
    'EEG C3-Ref','EEG T3-Ref','EEG A1-Ref','EEG T6-Ref','EEG P4-Ref',
    'EEG P3-Ref','EEG T5-Ref','EEG O2-Ref','EEG O1-Ref',
]
DROP_CHANNELS = ['EEG Cz-Ref','EEG Pz-Ref','ECG EKG','Manual']


# ── Helpers ───────────────────────────────────────────────────────────────────
def list_files():
    print(f"\n{'='*60}")
    print(f"  Mendeley EDF files in: {EDF_DIR}")
    print(f"{'='*60}")
    total = 0
    for patient in sorted(PATIENT_FILES.keys()):
        for ridx, fname in enumerate(PATIENT_FILES[patient], start=1):
            path = os.path.join(EDF_DIR, f"p{patient}_{fname}")
            exists = os.path.exists(path)
            sz_count = len(SEIZURE_DB[patient].get(ridx, []))
            status = "OK" if exists else "MISSING"
            print(f"  [{status}]  p{patient}_{fname:<14}  seizures: {sz_count}")
            if exists:
                total += 1
    print(f"\n  {total} file(s) found on disk.")
    print(f"  Patients: {sorted(PATIENT_FILES.keys())}")
    print()


def build_regions(patient, record_idx, n_samples, rec_start):
    """Return lists of (start_s, end_s, label) regions."""
    seizure_list = SEIZURE_DB[patient].get(record_idx, [])
    seizure_intervals = []

    ictal_regions    = []
    preictal_regions = []

    for h, m, s, dur in seizure_list:
        sz_time   = datetime.combine(date.today(), dtime(h, m, s))
        onset_s   = int((sz_time - rec_start).total_seconds() * FS)
        offset_s  = onset_s + int(dur * FS)
        onset_s   = max(0, min(onset_s,  n_samples))
        offset_s  = max(0, min(offset_s, n_samples))
        seizure_intervals.append((onset_s, offset_s))

        ictal_regions.append((onset_s, offset_s))

        pi_start = max(0, onset_s - PREICTAL_START * FS)
        pi_end   = max(0, onset_s - PREICTAL_END   * FS)
        # Clip against previous seizure end
        for prev_on, prev_off in seizure_intervals[:-1]:
            if pi_start < prev_off:
                pi_start = prev_off
        if pi_end > pi_start:
            preictal_regions.append((int(pi_start), int(pi_end)))

    # Normal regions: ≥ NORMAL_DIST from any seizure
    normal_regions = []
    win = FS  # 1-second blocks
    for i in range(0, n_samples - win + 1, win):
        ok = True
        for on_s, off_s in seizure_intervals:
            buf_s = max(0, on_s  - NORMAL_DIST * FS)
            buf_e = min(n_samples, off_s + NORMAL_DIST * FS)
            if buf_s <= i < buf_e:
                ok = False
                break
        if ok:
            # Merge contiguous blocks
            if normal_regions and normal_regions[-1][1] == i:
                normal_regions[-1] = (normal_regions[-1][0], i + win)
            else:
                normal_regions.append((i, i + win))

    return ictal_regions, preictal_regions, normal_regions


def plot_edf(patient, record_idx):
    fname = PATIENT_FILES[patient][record_idx - 1]
    path  = os.path.join(EDF_DIR, f"p{patient}_{fname}")

    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    print(f"\nLoading: p{patient}_{fname}  ...")
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

    if patient == 10:
        raw.reorder_channels(P10_CHANNELS)
    else:
        existing_drops = [c for c in DROP_CHANNELS if c in raw.ch_names]
        if existing_drops:
            raw.drop_channels(existing_drops)

    data       = raw.get_data()          # (19, N)
    ch_names   = raw.ch_names
    n_ch, n_s  = data.shape
    rec_time   = raw.info['meas_date']
    rec_start  = datetime.combine(date.today(), rec_time.time())
    duration_m = n_s / FS / 60

    seizure_list = SEIZURE_DB[patient].get(record_idx, [])

    print(f"  Channels  : {n_ch}")
    print(f"  Samples   : {n_s:,}  ({duration_m:.1f} min)")
    print(f"  Seizures  : {len(seizure_list)}")
    for i, (h, m, s, dur) in enumerate(seizure_list, 1):
        print(f"    #{i}  onset {h:02d}:{m:02d}:{s:02d}  duration {dur}s")

    ictal_r, preictal_r, normal_r = build_regions(
        patient, record_idx, n_s, rec_start)

    # Time axis in minutes
    t = np.arange(n_s) / FS / 60

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_ch, 1, figsize=(18, n_ch * 0.7),
                             sharex=True, facecolor="#111111")
    fig.suptitle(
        f"Raw EEG — Patient {patient}, Record {record_idx}  "
        f"({fname})   [{duration_m:.1f} min  |  {n_ch} channels  |  "
        f"{len(seizure_list)} seizure(s)]",
        color="white", fontsize=11, y=1.0
    )

    def shade(ax, regions, color, alpha):
        for s, e in regions:
            ax.axvspan(s / FS / 60, e / FS / 60,
                       color=color, alpha=alpha, linewidth=0)

    for i, ax in enumerate(axes):
        sig = data[i]
        # Normalise per channel for display only
        rng = np.ptp(sig) or 1
        sig_n = (sig - sig.mean()) / rng

        shade(ax, normal_r,   "#22aa44", 0.10)
        shade(ax, preictal_r, "#ff8800", 0.25)
        shade(ax, ictal_r,    "#ee2222", 0.40)

        ax.plot(t, sig_n, color="#7ec8e3", linewidth=0.4, rasterized=True)
        ax.set_yticks([])
        ax.set_facecolor("#111111")
        lbl = ch_names[i].replace("EEG ", "").replace("-Ref", "")
        ax.set_ylabel(lbl, color="white", fontsize=6,
                      rotation=0, labelpad=28, va="center")
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[-1].set_xlabel("Time (minutes)", color="white", fontsize=9)
    axes[-1].tick_params(colors="white")

    legend_patches = [
        mpatches.Patch(color="#22aa44", alpha=0.5, label="Normal (≥60 min from seizure)"),
        mpatches.Patch(color="#ff8800", alpha=0.6, label="Pre-ictal (5–25 min before onset)"),
        mpatches.Patch(color="#ee2222", alpha=0.7, label="Ictal (during seizure)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               facecolor="#222222", labelcolor="white", fontsize=8,
               framealpha=0.8, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.subplots_adjust(hspace=0)
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Plot raw Mendeley EDF before preprocessing")
    ap.add_argument("--patient",  type=int, default=None,
                    help="Patient number (10–15)")
    ap.add_argument("--record",   type=int, default=None,
                    help="Record number (1-based)")
    ap.add_argument("--list",     action="store_true",
                    help="List all EDF files and exit")
    args = ap.parse_args()

    if args.list:
        list_files()
        return

    list_files()

    if args.patient and args.record:
        patient = args.patient
        record  = args.record
    else:
        # Interactive picker
        print("Available patients:", sorted(PATIENT_FILES.keys()))
        try:
            patient = int(input("Enter patient number: ").strip())
            max_rec = len(PATIENT_FILES[patient])
            print(f"Records available: 1–{max_rec}")
            record  = int(input("Enter record number: ").strip())
        except (ValueError, KeyError):
            print("Invalid input.")
            sys.exit(1)

    if patient not in PATIENT_FILES:
        print(f"Unknown patient {patient}. Choose from {sorted(PATIENT_FILES.keys())}")
        sys.exit(1)
    if not (1 <= record <= len(PATIENT_FILES[patient])):
        print(f"Record {record} out of range for patient {patient}.")
        sys.exit(1)

    plot_edf(patient, record)


if __name__ == "__main__":
    main()
