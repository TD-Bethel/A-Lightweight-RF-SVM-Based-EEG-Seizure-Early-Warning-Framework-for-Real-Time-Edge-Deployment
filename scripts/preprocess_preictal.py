# =============================================================================
# preprocess_preictal.py  —  Extract genuine pre-ictal windows from Mendeley EDF
#
# What this does that the original Preprocess.py did NOT:
#   Original: all "Pre-Seizure" windows were DURING seizures (CPS/video-detected)
#   This:     extracts windows from 5-30 min BEFORE each seizure onset → truly
#             pre-ictal, allowing the model to warn before a seizure starts.
#
# Label scheme:
#   0 = Normal      (≥60 min from any seizure)
#   1 = Pre-ictal   (5–30 min before seizure onset)
#   2 = Ictal       (during seizure)
#
# Output: data/Mendelay dataset/Npy_files_preictal/
#         x_train.npy  y_train.npy  x_val.npy  y_val.npy  x_test.npy  y_test.npy
#
# USAGE:
#   python scripts/preprocess_preictal.py
#   python scripts/preprocess_preictal.py --preictal-minutes 20 --gap-minutes 3
# =============================================================================

import os, sys, argparse, warnings
import numpy as np
from datetime import datetime, date, time as dtime, timedelta
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    print("❌  mne not installed — pip install mne")
    sys.exit(1)

# ── CLI ────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--preictal-minutes", type=int, default=25,
                help="How many minutes before seizure onset to label as pre-ictal (default 25)")
ap.add_argument("--gap-minutes", type=int, default=5,
                help="Minutes before onset to STOP pre-ictal window (buffer, default 5)")
ap.add_argument("--normal-dist-minutes", type=int, default=60,
                help="Minimum minutes from any seizure to be classed Normal (default 60)")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

# ── Parameters ────────────────────────────────────────────────────────────────
FS              = 500                          # Hz — confirmed from original script
WIN_SIZE        = 500                          # samples = 1 second at 500 Hz
STEP            = WIN_SIZE                     # non-overlapping windows
PREICTAL_START  = args.preictal_minutes * 60  # seconds before onset: window start
PREICTAL_END    = args.gap_minutes * 60       # seconds before onset: window end
NORMAL_DIST     = args.normal_dist_minutes * 60
TEST_SIZE       = 0.10
VAL_SIZE        = 0.10
SEED            = args.seed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDF_DIR  = os.path.join(BASE_DIR, "data", "Mendelay dataset", "Raw_EDF_Files")
OUT_DIR  = os.path.join(BASE_DIR, "data", "Mendelay dataset", "Npy_files_preictal")
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = {0: "Normal", 1: "Pre-ictal", 2: "Ictal"}

print(f"\n{'═'*65}")
print(f"  NeuroWatch — Pre-ictal Preprocessing")
print(f"{'═'*65}")
print(f"  Pre-ictal window : -{PREICTAL_START//60} min  to  -{PREICTAL_END//60} min before onset")
print(f"  Normal min dist  : >{NORMAL_DIST//60} min from any seizure")
print(f"  Window size      : {WIN_SIZE} samples = 1 second at {FS} Hz")
print(f"  Output dir       : {OUT_DIR}")
print(f"{'═'*65}\n")

# ── Seizure timestamps (hour, minute, second, duration_seconds) ───────────────
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
    10: ["Record1.edf","Record2.edf"],
    11: ["Record1.edf","Record2.edf","Record3.edf","Record4.edf"],
    12: ["Record1.edf","Record2.edf","Record3.edf"],
    13: ["Record1.edf","Record2.edf","Record3.edf","Record4.edf"],
    14: ["Record1.edf","Record2.edf","Record3.edf"],
    15: ["Record1.edf","Record2.edf","Record3.edf","Record4.edf"],
}

P10_CHANNELS = [
    'EEG Fp2-Ref','EEG Fp1-Ref','EEG F8-Ref','EEG F4-Ref','EEG Fz-Ref',
    'EEG F3-Ref','EEG F7-Ref','EEG A2-Ref','EEG T4-Ref','EEG C4-Ref',
    'EEG C3-Ref','EEG T3-Ref','EEG A1-Ref','EEG T6-Ref','EEG P4-Ref',
    'EEG P3-Ref','EEG T5-Ref','EEG O2-Ref','EEG O1-Ref',
]
DROP_CHANNELS = ['EEG Cz-Ref','EEG Pz-Ref','ECG EKG','Manual']


# ── Helpers ───────────────────────────────────────────────────────────────────
def slice_windows(data, start_s, end_s):
    """Non-overlapping 500-sample windows from data[:, start_s:end_s]."""
    start_s = max(0, start_s)
    end_s   = min(data.shape[1], end_s)
    if end_s - start_s < WIN_SIZE:
        return []
    wins = []
    for i in range(start_s, end_s - WIN_SIZE + 1, STEP):
        wins.append(data[:, i:i + WIN_SIZE].copy())
    return wins


def normalise(windows):
    """Per-window max-abs normalisation (matches original preprocessing)."""
    out = []
    for w in windows:
        m = np.amax(np.abs(w))
        out.append(w / m if m > 0 else w)
    return out


# ── Main extraction loop ───────────────────────────────────────────────────────
all_ictals    = []
all_preictals = []
all_normals   = []

for patient in sorted(PATIENT_FILES.keys()):
    files = PATIENT_FILES[patient]
    for record_idx, fname in enumerate(files, start=1):
        edf_path = os.path.join(EDF_DIR, f"p{patient}_{fname}")
        if not os.path.exists(edf_path):
            print(f"  ⚠  Not found: {edf_path} — skipping")
            continue

        print(f"  Patient {patient}  Record {record_idx}  ({fname})")

        # ── Load EDF ────────────────────────────────────────────────────────
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        if patient == 10:
            raw.reorder_channels(P10_CHANNELS)
        else:
            existing_drops = [c for c in DROP_CHANNELS if c in raw.ch_names]
            if existing_drops:
                raw.drop_channels(existing_drops)

        data       = raw.get_data()           # (19, N_samples)
        rec_time   = raw.info['meas_date']
        rec_start  = datetime.combine(date.today(), rec_time.time())
        n_samples  = data.shape[1]

        # ── Build seizure index list for this recording ───────────────────
        seizure_list = SEIZURE_DB[patient].get(record_idx, [])
        seizure_intervals = []   # list of (onset_s, offset_s) in samples

        for h, m, s, dur in seizure_list:
            sz_time    = datetime.combine(date.today(), dtime(h, m, s))
            onset_s    = int((sz_time - rec_start).total_seconds() * FS)
            offset_s   = onset_s + int(dur * FS)
            onset_s    = max(0, onset_s)
            offset_s   = min(n_samples, offset_s)
            seizure_intervals.append((onset_s, offset_s))

            # ── Ictal windows ───────────────────────────────────────────
            wins = slice_windows(data, onset_s, offset_s)
            all_ictals.extend(normalise(wins))

            # ── Pre-ictal windows ───────────────────────────────────────
            pi_start_s = onset_s - int(PREICTAL_START * FS)
            pi_end_s   = onset_s - int(PREICTAL_END   * FS)

            # Clip to recording bounds
            pi_start_s = max(0, pi_start_s)

            # Avoid overlapping with any previous ictal period
            for prev_on, prev_off in seizure_intervals[:-1]:
                if pi_start_s < prev_off:
                    pi_start_s = prev_off   # push start past end of previous seizure

            if pi_end_s > pi_start_s + WIN_SIZE:
                wins = slice_windows(data, pi_start_s, pi_end_s)
                all_preictals.extend(normalise(wins))

        # ── Normal windows: regions ≥ NORMAL_DIST from every seizure ─────
        n_total_s  = n_samples
        used        = set()

        # Mark all ictal + safety-buffer samples as used
        for on_s, off_s in seizure_intervals:
            buf_start = max(0,          on_s  - int(NORMAL_DIST * FS))
            buf_end   = min(n_total_s,  off_s + int(NORMAL_DIST * FS))
            for i in range(buf_start // WIN_SIZE, buf_end // WIN_SIZE + 1):
                used.add(i)

        for i in range(0, n_total_s - WIN_SIZE + 1, STEP):
            if (i // WIN_SIZE) not in used:
                win = data[:, i:i + WIN_SIZE].copy()
                m   = np.amax(np.abs(win))
                all_normals.append(win / m if m > 0 else win)

        print(f"    seizures={len(seizure_list)}  "
              f"ictal+={len(all_ictals)}  "
              f"preictal+={len(all_preictals)}  "
              f"normal+={len(all_normals)}")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  Raw counts before balancing:")
print(f"    Ictal    (class 2) : {len(all_ictals):>6}")
print(f"    Pre-ictal(class 1) : {len(all_preictals):>6}")
print(f"    Normal   (class 0) : {len(all_normals):>6}")

if not all_ictals or not all_preictals or not all_normals:
    print("\n❌  One or more classes are empty — check EDF paths and seizure timestamps.")
    sys.exit(1)

# ── Balance: cap each class to min count so no class dominates ────────────────
min_count  = min(len(all_ictals), len(all_preictals), len(all_normals))
# Cap normal and pre-ictal to a maximum of 3× ictal count so classes are close
cap        = min(min_count * 3, len(all_ictals), len(all_preictals), len(all_normals))
cap        = max(cap, len(all_ictals))   # always keep all ictal windows

rng = np.random.default_rng(SEED)

def subsample(lst, n):
    arr = np.array(lst)
    if len(arr) <= n:
        return arr
    idx = rng.choice(len(arr), size=n, replace=False)
    return arr[idx]

ictals    = np.array(all_ictals)
preictals = subsample(all_preictals, cap)
normals   = subsample(all_normals,   cap)

print(f"\n  After balancing:")
print(f"    Ictal    (class 2) : {len(ictals):>6}")
print(f"    Pre-ictal(class 1) : {len(preictals):>6}")
print(f"    Normal   (class 0) : {len(normals):>6}")

X = np.vstack([normals, preictals, ictals]).astype(np.float32)
y = np.array(
    [0] * len(normals) +
    [1] * len(preictals) +
    [2] * len(ictals),
    dtype=np.int32,
)

# ── Save as x_train (80%) + x_test (20%) so train_mendeley.py gets 100% ───────
# train_mendeley.py combines these two files then applies its own 80/10/10 split.
# Saving a separate x_val would cause 10% of data to be silently dropped there.
print(f"\n  Saving 80% / 20% files for train_mendeley.py (which re-splits 80/10/10)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)

print(f"    x_train.npy : {len(X_train):>5} samples (80%)")
print(f"    x_test.npy  : {len(X_test):>5} samples (20% — train_mendeley.py splits this into val+test)")
for split_name, y_s in [("Train", y_train), ("Test", y_test)]:
    dist = {CLASSES[c]: int(np.sum(y_s == c)) for c in [0, 1, 2]}
    print(f"    {split_name:<6}: {dist}")

# ── Save ───────────────────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "x_train.npy"), X_train)
np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUT_DIR, "x_test.npy"),  X_test)
np.save(os.path.join(OUT_DIR, "y_test.npy"),  y_test)

print(f"\n✅  Saved to: {OUT_DIR}")
print(f"    x_train.npy  {X_train.shape}")
print(f"    x_test.npy   {X_test.shape}")
print(f"\n  To train (final 80/10/10 split applied inside train_mendeley.py):")
print(f"    python train_mendeley.py --data \"{OUT_DIR}\" --out models/MODELS_PREICTAL")
print(f"\n  Label meanings in this dataset:")
print(f"    0 = Normal    (EEG >{args.normal_dist_minutes} min from any seizure)")
print(f"    1 = Pre-ictal ({args.gap_minutes}–{args.preictal_minutes} min before seizure onset)")
print(f"    2 = Ictal     (during seizure)\n")
