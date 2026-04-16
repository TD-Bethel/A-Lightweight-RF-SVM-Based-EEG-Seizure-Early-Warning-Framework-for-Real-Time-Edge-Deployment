# =============================================================================
# train_and_export.py  -  NeuroWatch Incremental Retraining Pipeline
#
# HOW IT WORKS:
#   - First run: trains from scratch on your dataset, saves models + data archive
#   - Every new run: loads previous training data, merges with new dataset,
#     retrains on the combined data so the model improves over time
#
# USAGE:
#   python train_and_export.py
#   python train_and_export.py --data "C:\new\eeg\folder"
#   python train_and_export.py --data "C:\new\eeg\folder" --reset
#
# SUPPORTS:
#   - Bonn .txt format with O/N/S/F/Z subfolders
#   - Flat EDF folders
#   - Test dataset Raw_EDF_Files with seizure timing metadata
# =============================================================================

import os
import re
import json
import argparse
import importlib.util
from datetime import datetime

import joblib
import numpy as np
from scipy.signal import butter, filtfilt, welch, resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ======================== CONFIGURATION ========================
dataset_path = r"C:\Users\thebe\Documents\MATLAB\Bonn Univeristy Dataset"
model_export_path = "models"

# Bonn Mapping: O/N=Normal(0), S/F=Pre-Seizure(1), Z=Seizure(2)
CLASS_MAP = {"O": 0, "N": 0, "S": 1, "F": 1, "Z": 2}
CLASSES = ["Normal", "Pre-Seizure", "Seizure"]
FS = 128
NPERSEG = 128
PRESEIZURE_SEC = 300

ARCHIVE_PATH = os.path.join(model_export_path, "training_archive.npz")
HISTORY_LOG = os.path.join(model_export_path, "training_history.json")

# ======================== ARGUMENT PARSER ========================
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="Path to new EEG dataset folder")
parser.add_argument(
    "--reset",
    action="store_true",
    help="Ignore previous archive and train from scratch",
)
args = parser.parse_args()

if args.data:
    dataset_path = os.path.abspath(args.data)

if not os.path.exists(model_export_path):
    os.makedirs(model_export_path)

print("=" * 60)
print("  NeuroWatch Incremental Retraining Pipeline")
print("=" * 60)
print(f"  Dataset : {dataset_path}")
print(f"  Models  : {model_export_path}")
print(f"  Mode    : {'FRESH (--reset)' if args.reset else 'INCREMENTAL'}")
print("=" * 60)


# ======================== FEATURE EXTRACTION ========================
def extract_features(data, fs=FS):
    """Extract [Delta, Theta, Alpha, Beta] band powers from a 1D EEG signal."""
    try:
        if fs != FS:
            data = resample(data, int(len(data) * FS / fs))
        b, a = butter(4, [0.5 / (0.5 * FS), 45 / (0.5 * FS)], btype="band")
        filt = filtfilt(b, a, data)
        f, psd = welch(filt, fs=FS, nperseg=NPERSEG)
        return [
            float(np.mean(psd[(f >= 1) & (f <= 4)])),
            float(np.mean(psd[(f >= 4) & (f <= 8)])),
            float(np.mean(psd[(f >= 8) & (f <= 13)])),
            float(np.mean(psd[(f >= 13) & (f <= 30)])),
        ]
    except Exception as e:
        print(f"  Feature extraction failed: {e}")
        return None


# ======================== TEST DATASET METADATA ========================
def load_test_dataset_seizure_map(raw_edf_path):
    """Load seizure timing metadata for the flat Test dataset EDF files."""
    test_root = os.path.dirname(raw_edf_path)
    seizure_file = os.path.join(test_root, "Preprocessing_Scripts", "Seizure_times.py")
    if not os.path.exists(seizure_file):
        return {}

    spec = importlib.util.spec_from_file_location("neurowatch_seizure_times", seizure_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    seizure_map = {}
    for patient in (10, 11, 12, 13, 14, 15):
        attr = f"seizures_{patient}"
        if hasattr(module, attr):
            seizure_map[patient] = getattr(module, attr)
    return seizure_map


def parse_test_edf_identity(file_path):
    """Extract patient and record numbers from names like p15_Record4.edf."""
    match = re.match(r"p(\d+)_Record(\d+)\.edf$", os.path.basename(file_path), re.IGNORECASE)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def get_test_dataset_events(raw, file_path, seizure_map):
    """Return seizure events for a test EDF file as a list of (start_sec, end_sec)."""
    patient, record = parse_test_edf_identity(file_path)
    if patient is None or patient not in seizure_map:
        return []

    record_events = seizure_map[patient].get(record, [])
    if not record_events:
        return []

    meas_date = raw.info.get("meas_date")
    if meas_date is None:
        return []

    record_start = meas_date.replace(tzinfo=None)
    events = []
    for hour, minute, second, duration in record_events:
        seizure_dt = record_start.replace(hour=hour, minute=minute, second=second, microsecond=0)
        onset_sec = (seizure_dt - record_start).total_seconds()
        if onset_sec < 0:
            continue
        events.append((float(onset_sec), float(onset_sec + duration)))
    return events


def label_edf_window(start_sec, end_sec, default_label, seizure_events):
    """Assign seizure/pre-seizure labels to a window if metadata is available."""
    if not seizure_events:
        return default_label

    label = 0
    for seizure_start, seizure_end in seizure_events:
        if start_sec < seizure_end and end_sec > seizure_start:
            return 2
        if start_sec >= max(0.0, seizure_start - PRESEIZURE_SEC) and end_sec <= seizure_start:
            label = max(label, 1)
    return label


# ======================== LOAD NEW DATASET ========================
def load_dataset(path):
    """
    Loads .txt (Bonn-style) or .edf files from path.
    For Bonn: expects subfolders named O, N, S, F, Z.
    For flat EDF folders: reads first channel, segments into 1-second windows.
    For Test dataset Raw_EDF_Files: derives labels from seizure timing metadata.
    Returns X (features array), y (labels array), n_files (int).
    """
    X_new, y_new = [], []
    n_files = 0

    bonn_dirs = [
        os.path.join(path, s)
        for s in CLASS_MAP.keys()
        if os.path.isdir(os.path.join(path, s))
    ]
    search_dirs = bonn_dirs if bonn_dirs else [path]
    test_seizure_map = (
        load_test_dataset_seizure_map(path)
        if os.path.basename(path).lower() == "raw_edf_files"
        else {}
    )

    for d in search_dirs:
        if not os.path.exists(d):
            continue

        folder_name = os.path.basename(d)
        label = CLASS_MAP.get(folder_name, 0)

        txt_files = [
            os.path.join(d, f)
            for f in os.listdir(d)
            if f.lower().endswith(".txt")
        ]
        edf_files = [
            os.path.join(d, f)
            for f in os.listdir(d)
            if f.lower().endswith(".edf")
        ]

        for fpath in txt_files:
            try:
                signal = np.loadtxt(fpath)
            except Exception as e:
                print(f"  TXT skip {os.path.basename(fpath)}: {e}")
                continue

            signal = np.asarray(signal).squeeze()
            if signal.ndim != 1 or signal.size < FS:
                print(f"  TXT skip {os.path.basename(fpath)}: invalid EEG shape {signal.shape}")
                continue

            feats = extract_features(signal)
            if feats:
                X_new.append(feats)
                y_new.append(label)
                n_files += 1

        for fpath in edf_files:
            try:
                import mne

                raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
                fs = raw.info["sfreq"]
                data = raw.get_data()[0]
                seizure_events = (
                    get_test_dataset_events(raw, fpath, test_seizure_map)
                    if test_seizure_map
                    else []
                )

                win = int(fs)
                for start in range(0, len(data) - win, win):
                    start_sec = start / fs
                    end_sec = (start + win) / fs
                    sample_label = label_edf_window(start_sec, end_sec, label, seizure_events)
                    feats = extract_features(data[start:start + win], fs=fs)
                    if feats:
                        X_new.append(feats)
                        y_new.append(sample_label)
                n_files += 1
            except Exception as e:
                print(f"  EDF skip {os.path.basename(fpath)}: {e}")

    return np.array(X_new), np.array(y_new), n_files


# ======================== LOAD ARCHIVE ========================
def load_archive():
    """Load previously saved training data. Returns empty arrays if none exists."""
    if args.reset or not os.path.exists(ARCHIVE_PATH):
        return np.empty((0, 4)), np.empty(0, dtype=int)
    try:
        arc = np.load(ARCHIVE_PATH)
        print(f"  Loaded archive: {len(arc['X'])} previous samples")
        return arc["X"], arc["y"]
    except Exception as e:
        print(f"  Archive load failed ({e}) - starting fresh.")
        return np.empty((0, 4)), np.empty(0, dtype=int)


# ======================== SAVE ARCHIVE ========================
def save_archive(X, y):
    """Persist the combined dataset so next run can build on it."""
    np.savez_compressed(ARCHIVE_PATH, X=X, y=y)
    print(f"  Archive saved: {len(X)} total samples -> {ARCHIVE_PATH}")


# ======================== LOG TRAINING RUN ========================
def log_training(n_old, n_new, n_total, acc, report_str, dataset_used):
    """Append a record to training_history.json so you can track improvement."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_used,
        "samples_prev": int(n_old),
        "samples_new": int(n_new),
        "samples_total": int(n_total),
        "accuracy": round(float(acc), 4),
        "report": report_str,
    }
    history = []
    if os.path.exists(HISTORY_LOG):
        try:
            with open(HISTORY_LOG, "r") as f:
                history = json.load(f)
        except Exception:
            pass
    history.append(entry)
    with open(HISTORY_LOG, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history updated -> {HISTORY_LOG}")


# ======================== MAIN PIPELINE ========================
X_old, y_old = load_archive()
n_old = len(X_old)

print(f"\nLoading new dataset from: {dataset_path}")
X_new, y_new, n_files = load_dataset(dataset_path)

if len(X_new) == 0:
    print("No valid EEG files found. Check your dataset path.")
    print("Expected: subfolders O, N, S, F, Z with .txt files,")
    print("          OR .edf files directly in the folder.")
    raise SystemExit(1)

print(f"  New samples loaded: {len(X_new)} from {n_files} file(s)")

if n_old > 0:
    X_combined = np.vstack([X_old, X_new])
    y_combined = np.concatenate([y_old, y_new])
    print(f"  Merged: {n_old} (archive) + {len(X_new)} (new) = {len(X_combined)} total")
else:
    X_combined = X_new
    y_combined = y_new
    print("  First training run - no archive to merge.")

unique_classes = np.unique(y_combined)
if len(unique_classes) < 2:
    print(f"Only found class(es): {unique_classes}. Need at least 2 classes to train.")
    print("Make sure your dataset has Normal AND Pre-Seizure/Seizure samples.")
    raise SystemExit(1)

class_counts = {CLASSES[c]: int(np.sum(y_combined == c)) for c in unique_classes}
print(f"  Class distribution: {class_counts}")

X_train, X_test, y_train, y_test = train_test_split(
    X_combined,
    y_combined,
    test_size=0.2,
    random_state=42,
    stratify=y_combined,
)
print(f"\n  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print("\nTraining SVM...")
svm_model = SVC(kernel="rbf", C=2.0, probability=True, class_weight="balanced")
svm_model.fit(X_train_sc, y_train)

print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    class_weight="balanced",
)
rf_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict_proba(X_test_sc)
y_pred_rf = rf_model.predict_proba(X_test)
y_pred_hybrid = np.argmax((0.4 * y_pred_svm) + (0.6 * y_pred_rf), axis=1)

acc = accuracy_score(y_test, y_pred_hybrid)
report = classification_report(
    y_test,
    y_pred_hybrid,
    target_names=[CLASSES[c] for c in sorted(unique_classes)],
    zero_division=0,
)

print("\n" + "=" * 60)
print("  TEST RESULTS")
print("=" * 60)
print(f"  Hybrid Accuracy: {acc * 100:.2f}%")
print(report)

if os.path.exists(HISTORY_LOG):
    try:
        with open(HISTORY_LOG) as f:
            hist = json.load(f)
        if hist:
            prev_acc = hist[-1]["accuracy"]
            delta = (acc - prev_acc) * 100
            arrow = "UP" if delta >= 0 else "DOWN"
            print(
                f"  {arrow} vs previous run: {prev_acc * 100:.2f}% -> {acc * 100:.2f}% "
                f"({'+' if delta >= 0 else ''}{delta:.2f}%)"
            )
    except Exception:
        pass

save_archive(X_combined, y_combined)

print(f"\nExporting models to: {model_export_path}")
joblib.dump(scaler, os.path.join(model_export_path, "scaler.pkl"))
joblib.dump(svm_model, os.path.join(model_export_path, "svm_model.pkl"))
joblib.dump(rf_model, os.path.join(model_export_path, "rf_model.pkl"))

log_training(n_old, len(X_new), len(X_combined), acc, report, dataset_path)

print("\nDone! Models updated and saved.")
print("Run main_pi_bios_v15.py or run_prediction.py to use the new models.")
