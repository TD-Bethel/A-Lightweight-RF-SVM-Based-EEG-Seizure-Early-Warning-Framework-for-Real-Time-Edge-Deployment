import os, sys, time, json, argparse, glob, random, warnings, threading, shutil, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurowatch_selector import RFImportanceSelector  # needed for joblib unpickling
import numpy as np
import joblib
import mne  # Required: pip install mne
from datetime import datetime
from scipy.signal import butter, filtfilt, welch, resample

try:
    from patients import PATIENT_REGISTRY
except ImportError:
    try:
        from src.patients import PATIENT_REGISTRY
    except ImportError:
        print("⚠️  patients.py not found — using single default patient")
        PATIENT_REGISTRY = [{
            "id": "P001", "name": "John Molebatsi", "age": 34,
            "diagnosis": "Temporal Lobe Epilepsy", "ward": "Neuro A", "bed": "Bed 1",
            "emergency_contact": "Kelebogile Molebatsi", "emergency_phone": "+26771000001",
            "gps_lat": -23.9571, "gps_lng": 26.8368,
            "gps_label": "Neuro Ward A, Bed 1, Princess Marina Hospital",
            "eeg_slice": (0, 20), "live": True,
        }]

try:
    from sms_notifier import send_sms_alert
    SMS_AVAILABLE = True
except ImportError:
    try:
        from src.sms_notifier import send_sms_alert
        SMS_AVAILABLE = True
    except ImportError:
        SMS_AVAILABLE = False

# --- HARDWARE IMPORTS ---
try:
    import RPi.GPIO as GPIO
    from RPLCD.gpio import CharLCD
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

# =============================================================================
# CONFIGURATION & UNIVERSAL PATH SELECTION
# =============================================================================
parser = argparse.ArgumentParser(description="NeuroWatch Universal Engine")
parser.add_argument("--data", type=str, help="Path to folder containing .edf or .txt files")
parser.add_argument("--models", type=str, help="Path to .pkl models folder")
parser.add_argument("--bios", action="store_true", help="Open maintenance BIOS mode")
args = parser.parse_args()

# 1. BASE_DIR points to project root (one level above src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = PROJECT_ROOT

# 2. FIX: STATUS_JSON now points to your project folder (Works on Windows & Pi)
STATUS_JSON   = os.path.join(BASE_DIR, "json", "neurowatch_status.json")
PATIENTS_JSON = os.path.join(BASE_DIR, "json", "neurowatch_patients.json")

# 3. Set Data and Model Paths
DATA_PATH = os.path.abspath(args.data) if args.data else os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.abspath(args.models) if args.models else os.path.join(BASE_DIR, "models", "MODELS_FS75")

# 4. Model Constants
TARGET_FS = 128 
CLASSES = ["Normal", "Pre-Seizure", "Seizure"]
CLASS_MAP = {"O": 0, "N": 0, "S": 1, "F": 1, "Z": 2}

# GPIO Pins
PIN_LED_GREEN, PIN_LED_RED, PIN_BUZZER = 27, 22, 4
warnings.filterwarnings("ignore")

# Prolonged-state escalation thresholds (seconds)
PRESEIZURE_LONG_SECONDS = 12
SEIZURE_LONG_SECONDS = 30

# =============================================================================
# HARDWARE & ML CORES
# =============================================================================
class HardwareManager:
    def __init__(self):
        self.lcd = None
        self._alarm_mode = 0  # 0=normal, 1=pre-seizure, 2=seizure
        self._stop_buzzer = False
        self._buzzer_thread = None
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup([PIN_LED_GREEN, PIN_LED_RED, PIN_BUZZER], GPIO.OUT, initial=GPIO.LOW)
            try:
                self.lcd = CharLCD(pin_rs=26, pin_e=19, pins_data=[13, 6, 5, 20],
                                   numbering_mode=GPIO.BCM, cols=20, rows=4)
                self.lcd.clear()
            except: print("⚠️ LCD Hardware not found.")
            self._buzzer_thread = threading.Thread(target=self._buzzer_worker, daemon=True)
            self._buzzer_thread.start()

    def update_lcd(self, rows):
        if self.lcd:
            for i, text in enumerate(rows[:4]):
                self.lcd.cursor_pos = (i, 0)
                self.lcd.write_string(text[:20].ljust(20))
        else:
            # Clean terminal print for Windows testing
            print("\n" + "-"*30 + "\n" + "\n".join(rows) + "\n" + "-"*30)

    def set_alarm(self, level):
        if not GPIO_AVAILABLE: return
        self._alarm_mode = int(level)
        GPIO.output(PIN_LED_GREEN, GPIO.HIGH if level == 0 else GPIO.LOW)
        GPIO.output(PIN_LED_RED, GPIO.HIGH if level > 0 else GPIO.LOW)

    def _buzzer_worker(self):
        """Runs continuous buzzer patterns independent of the main loop speed."""
        while not self._stop_buzzer:
            mode = self._alarm_mode
            t = time.monotonic()
            if mode == 2:
                # Seizure: rapid continuous beeping (5 Hz square wave)
                buz = (int(t * 10) % 2) == 0
            elif mode == 1:
                # Pre-seizure: single beep pulse every 1 second
                buz = (t % 1.0) < 0.18
            else:
                buz = False
            GPIO.output(PIN_BUZZER, GPIO.HIGH if buz else GPIO.LOW)
            time.sleep(0.05)

        GPIO.output(PIN_BUZZER, GPIO.LOW)

    def cleanup(self):
        self._stop_buzzer = True
        if self._buzzer_thread and self._buzzer_thread.is_alive():
            self._buzzer_thread.join(timeout=0.3)
        if GPIO_AVAILABLE:
            GPIO.cleanup()

class Brain:
    def __init__(self):
        print(f"📦 Loading Models from: {MODEL_PATH}")
        try:
            self.scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
            self.svm    = joblib.load(os.path.join(MODEL_PATH, "svm_model.pkl"))
            self.rf     = joblib.load(os.path.join(MODEL_PATH, "rf_model.pkl"))
        except Exception as e:
            raise FileNotFoundError(f"Model files missing in {MODEL_PATH}: {e}")

        self.n_features  = int(self.scaler.n_features_in_)
        self.is_mendeley = (self.n_features >= 76)
        print(f"📊 Model type: {'Mendeley (%d features, 19ch)' % self.n_features if self.is_mendeley else 'Bonn (4 features, 1ch)'}")

        # Load auto-tuned ensemble weights if available, else fall back to defaults
        weights_path = os.path.join(MODEL_PATH, "ensemble_weights.pkl")
        if os.path.exists(weights_path):
            w = joblib.load(weights_path)
            self.svm_w            = float(w.get("svm_w",            0.4))
            self.rf_w             = float(w.get("rf_w",             0.6))
            self.normal_threshold = float(w.get("normal_threshold", 0.5))
            print(f"📊 Ensemble weights: SVM {self.svm_w:.0%} + RF {self.rf_w:.0%} (auto-tuned)  |  Normal threshold: {self.normal_threshold:.2f}")
        else:
            self.svm_w, self.rf_w, self.normal_threshold = 0.4, 0.6, 0.50

        selector_path = os.path.join(MODEL_PATH, "selector.pkl")
        if os.path.exists(selector_path):
            self.selector = joblib.load(selector_path)
            print(f"📊 Feature selector: {self.selector.k} features selected")
        else:
            self.selector = None

        # Track dataset fingerprint so we retrain only when data actually changes
        self._data_fingerprint = self._fingerprint(DATA_PATH)

    # ------------------------------------------------------------------
    # AUTO-RETRAIN: called once at startup and checked every N ticks
    # ------------------------------------------------------------------
    def check_and_retrain(self, hw=None):
        """
        Re-runs train_and_export.py if new EEG files have been added to
        DATA_PATH since the last time we checked. Reloads models afterwards.
        """
        new_fp = self._fingerprint(DATA_PATH)
        if new_fp == self._data_fingerprint:
            return False  # nothing changed

        print("\n🔄 New data detected in DATA_PATH — retraining models...")
        if hw:
            hw.update_lcd(["NEW DATA FOUND", "Retraining...", "Please wait", ""])

        # Locate training script in scripts/
        trainer = os.path.join(PROJECT_ROOT, "scripts", "train_and_export.py")
        if not os.path.exists(trainer):
            print(f"⚠️  train_and_export.py not found at {trainer} — skipping retrain.")
            return False

        import subprocess
        result = subprocess.run(
            [sys.executable, trainer, "--data", DATA_PATH],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(result.stdout[-800:])   # show last 800 chars of output
            # Reload updated models into this Brain instance
            self.scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
            self.svm    = joblib.load(os.path.join(MODEL_PATH, "svm_model.pkl"))
            self.rf     = joblib.load(os.path.join(MODEL_PATH, "rf_model.pkl"))
            self._data_fingerprint = new_fp
            print("✅ Models reloaded after retraining.")
            if hw:
                hw.update_lcd(["Retrain Done!", "Models updated", "Monitoring...", ""])
            return True
        else:
            print(f"❌ Retraining failed:\n{result.stderr[-400:]}")
            return False

    @staticmethod
    def _fingerprint(path):
        """
        Returns a string fingerprint of all EEG files in path.
        Changes whenever files are added, removed, or modified.
        """
        sig = []
        for root, _, files in os.walk(path):
            for fname in sorted(files):
                if fname.lower().endswith((".txt", ".edf")):
                    fpath = os.path.join(root, fname)
                    try:
                        sig.append(f"{fpath}:{os.path.getmtime(fpath):.0f}")
                    except Exception:
                        pass
        return "|".join(sig)

    # ------------------------------------------------------------------
    # FEATURE EXTRACTION & PREDICTION (unchanged)
    # ------------------------------------------------------------------
    def extract_features(self, data):
        """13-feature Bonn extractor — must stay in sync with train_and_export.py."""
        from scipy.stats import skew, kurtosis as _kurtosis
        nyq  = 0.5 * TARGET_FS
        b, a = butter(4, [0.5 / nyq, 45 / nyq], btype="band")
        filt   = filtfilt(b, a, data)
        f, psd = welch(filt, fs=TARGET_FS, nperseg=TARGET_FS)
        bp = np.array([
            np.mean(psd[(f >= 1)  & (f <= 4)]),
            np.mean(psd[(f >= 4)  & (f <= 8)]),
            np.mean(psd[(f >= 8)  & (f <= 13)]),
            np.mean(psd[(f >= 13) & (f <= 30)]),
            np.mean(psd[(f >= 30) & (f <= 45)]),
        ])
        total   = bp.sum()
        bp_norm = bp / total if total > 0 else bp

        d1, d2   = np.diff(data), np.diff(np.diff(data))
        var_x    = np.var(data) + 1e-10
        var_d1   = np.var(d1)   + 1e-10
        var_d2   = np.var(d2)   + 1e-10
        activity   = float(var_x)
        mobility   = float(np.sqrt(var_d1 / var_x))
        complexity = float(np.sqrt(var_d2 / var_d1) / (mobility + 1e-10))
        sk   = float(skew(data))
        kurt = float(_kurtosis(data))
        rms  = float(np.sqrt(np.mean(data ** 2)))
        zcr  = float(((data[:-1] * data[1:]) < 0).sum() / len(data))
        p_n  = psd / (psd.sum() + 1e-10)
        sent = float(-np.sum(p_n * np.log2(p_n + 1e-10)) / np.log2(len(p_n) + 1))

        return np.concatenate([bp_norm, [activity, mobility, complexity,
                                         sk, kurt, rms, zcr, sent]]).tolist()

    def predict(self, features):
        x_raw    = np.array([features])
        x_scaled = self.scaler.transform(x_raw)
        if self.selector is not None:
            x_svm = self.selector.transform(x_scaled)
            x_rf  = self.selector.transform(x_raw)
        else:
            x_svm = x_scaled
            x_rf  = x_raw
        p_svm    = self.svm.predict_proba(x_svm)[0]
        p_rf     = self.rf.predict_proba(x_rf)[0]
        combined = self.svm_w * p_svm + self.rf_w * p_rf
        idx = int(np.argmax(combined))
        return idx, float(combined[idx])

    _MENDELEY_FS = 500
    _WIN_SIZE    = 4097
    _N_CH        = 19
    _NPERSEG     = 128

    def extract_features_mendeley(self, sample):
        """Feature extraction from a (19, 500) Mendeley sample.
        Matches train_mendeley.py exactly.

        If model has 247 features (original Mendeley): per-channel only.
        If model has 418 features (pre-ictal Mendeley): per-channel + 171 inter-channel correlations.
        """
        from scipy.stats import skew, kurtosis as _kurtosis
        nyq  = 0.5 * TARGET_FS
        b, a = butter(4, [0.5 / nyq, 45 / nyq], btype="band")
        all_feats = []
        resampled = []

        # Max-abs normalise to [-1, 1] — matches Preprocess.py training transform
        m = np.amax(np.abs(sample))
        if m > 0:
            sample = sample / m

        for ch in range(self._N_CH):
            sig   = sample[ch]
            n_new = int(len(sig) * TARGET_FS / self._MENDELEY_FS)
            sig_r = resample(sig, n_new)
            reps  = (self._WIN_SIZE // len(sig_r)) + 1
            sig_t = np.tile(sig_r, reps)[:self._WIN_SIZE]
            resampled.append(sig_r)

            # Band powers (5) — Option A normalised
            filt   = filtfilt(b, a, sig_t)
            f, psd = welch(filt, fs=TARGET_FS, nperseg=self._NPERSEG)
            bp = np.array([
                np.mean(psd[(f >= 1)  & (f <= 4)]),
                np.mean(psd[(f >= 4)  & (f <= 8)]),
                np.mean(psd[(f >= 8)  & (f <= 13)]),
                np.mean(psd[(f >= 13) & (f <= 30)]),
                np.mean(psd[(f >= 30) & (f <= 45)]),
            ])
            total = bp.sum()
            bp_norm = bp / total if total > 0 else bp

            # Hjorth parameters
            d1, d2   = np.diff(sig_r), np.diff(np.diff(sig_r))
            var_x    = np.var(sig_r)  + 1e-10
            var_d1   = np.var(d1)     + 1e-10
            var_d2   = np.var(d2)     + 1e-10
            activity   = float(var_x)
            mobility   = float(np.sqrt(var_d1 / var_x))
            complexity = float(np.sqrt(var_d2 / var_d1) / (mobility + 1e-10))

            # Statistical features
            sk   = float(skew(sig_r))
            kurt = float(_kurtosis(sig_r))
            rms  = float(np.sqrt(np.mean(sig_r ** 2)))
            zcr  = float(((sig_r[:-1] * sig_r[1:]) < 0).sum() / len(sig_r))
            p_n  = psd / (psd.sum() + 1e-10)
            sent = float(-np.sum(p_n * np.log2(p_n + 1e-10)) / np.log2(len(p_n) + 1))

            all_feats.extend(bp_norm.tolist())
            all_feats.extend([activity, mobility, complexity, sk, kurt, rms, zcr, sent])

        return np.array(all_feats, dtype=np.float32).tolist()

    def extract_features_auto(self, sample):
        """Dispatch to the right extractor based on loaded model type."""
        if self.is_mendeley:
            return self.extract_features_mendeley(sample)
        return self.extract_features(sample)

    @staticmethod
    def band_summary(feats):
        """Return [delta, theta, alpha, beta] for display regardless of feature count."""
        n = len(feats)
        if n <= 13:   # Bonn: first 4 are band powers
            return list(feats[:4])
        if n == 247:  # Mendeley 19×13 — first 5 per channel are band powers
            return np.array(feats).reshape(19, 13)[:, :4].mean(axis=0).tolist()
        return list(feats[:4])

# =============================================================================
# METRICS
# =============================================================================
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

METRICS_JSON = os.path.join(BASE_DIR, "json", "neurowatch_metrics.json")


def _read_json_safe(path, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def _print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _bios_status_health_check():
    _print_header("[1] System status & JSON health check")
    checks = [
        ("Status JSON", STATUS_JSON),
        ("Patients JSON", PATIENTS_JSON),
        ("Metrics JSON", METRICS_JSON),
    ]
    for label, path in checks:
        ok = os.path.exists(path)
        state = "OK" if ok else "MISSING"
        print(f"- {label:<14}: {state}  -> {path}")
        if ok:
            data = _read_json_safe(path, None)
            parse_state = "valid JSON" if data is not None else "invalid JSON"
            print(f"  parse: {parse_state}")

    history_files = glob.glob(os.path.join(BASE_DIR, "json", "neurowatch_history_*.json"))
    print(f"- History files : {len(history_files)} found")


def _bios_show_registry():
    _print_header("[2] Patient registry")
    for i, p in enumerate(PATIENT_REGISTRY, start=1):
        print(
            f"{i}. {p.get('id', '--')} | {p.get('name', '--')} | "
            f"{p.get('ward', '--')} {p.get('bed', '--')} | live={p.get('live', False)}"
        )


def _bios_view_live_history():
    _print_header("[3] View live patient history")
    live_patient = next((p for p in PATIENT_REGISTRY if p.get("live", False)), None)
    pid = live_patient.get("id") if live_patient else "P001"
    path = os.path.join(BASE_DIR, "json", f"neurowatch_history_{pid}.json")
    history = _read_json_safe(path, [])
    if not history:
        print(f"No history found for {pid} at {path}")
        return

    print(f"Live patient: {pid} ({len(history)} records)")
    for row in history[-15:]:
        ts = row.get("timestamp", "--")
        st = row.get("state", "--")
        cf = row.get("confidence", 0)
        print(f"- {ts} | {st:<11} | conf={cf}")


def _bios_export_histories_csv():
    _print_header("[4] Export all histories to CSV")
    rows = []
    for path in glob.glob(os.path.join(BASE_DIR, "json", "neurowatch_history_*.json")):
        pid = os.path.basename(path).replace("neurowatch_history_", "").replace(".json", "")
        history = _read_json_safe(path, [])
        for r in history:
            rr = dict(r)
            rr["patient_id"] = pid
            rows.append(rr)

    if not rows:
        print("No history rows to export.")
        return

    out_name = f"neurowatch_history_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out_path = os.path.join(BASE_DIR, out_name)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Export complete: {out_path} ({len(rows)} rows)")


def _bios_view_model_metrics():
    _print_header("[5] View model metrics")
    m = _read_json_safe(METRICS_JSON, None)
    if not m:
        print("Metrics not found or invalid.")
        return
    print(f"Timestamp : {m.get('timestamp', '--')}")
    print(f"Accuracy  : {m.get('accuracy', '--')}")
    print(f"Precision : {m.get('precision', '--')}")
    print(f"Recall    : {m.get('recall', '--')}")
    print(f"F1        : {m.get('f1', '--')}")
    print(f"Specificity: {m.get('specificity', '--')}")
    print(f"Seizure R : {m.get('seizure_recall', '--')}")
    print(f"Samples   : {m.get('n_samples', '--')}")


def _bios_clear_histories():
    _print_header("[6] Clear all history files")
    files = glob.glob(os.path.join(BASE_DIR, "json", "neurowatch_history_*.json"))
    if not files:
        print("No history files found.")
        return
    confirm = input("Type YES to clear all history files: ").strip()
    if confirm != "YES":
        print("Cancelled.")
        return
    for fp in files:
        with open(fp, "w") as f:
            json.dump([], f)
    print(f"Cleared {len(files)} history file(s).")


def _bios_reset_status_json():
    _print_header("[7] Clear status / reset JSON files")
    confirm = input("Type RESET to continue: ").strip()
    if confirm != "RESET":
        print("Cancelled.")
        return

    with open(STATUS_JSON, "w") as f:
        json.dump({
            "phase": "WAITING",
            "state": "Normal",
            "conf": 0.0,
            "message": "Reset by BIOS maintenance",
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }, f)
    with open(PATIENTS_JSON, "w") as f:
        json.dump({}, f)
    print("Status and patient JSON files reset.")


def _bios_show_gpio_pin_map():
    _print_header("[8] Show GPIO pin map")
    print("- LED Green : GPIO 27")
    print("- LED Red   : GPIO 22")
    print("- Buzzer    : GPIO 4")
    print("- LCD RS    : GPIO 26")
    print("- LCD E     : GPIO 19")
    print("- LCD D4..D7: GPIO 13, 6, 5, 20")


def _bios_check_disk_space():
    _print_header("[9] Check disk space")
    total, used, free = shutil.disk_usage(BASE_DIR)
    gb = 1024 ** 3
    print(f"Path : {BASE_DIR}")
    print(f"Total: {total / gb:.2f} GB")
    print(f"Used : {used / gb:.2f} GB")
    print(f"Free : {free / gb:.2f} GB")


def run_bios_mode():
    if not SMS_AVAILABLE:
        print("Warning: sms_notifier.py not found or unavailable - SMS disabled")

    while True:
        _print_header("NeuroWatch BIOS v1.0  |  MAINTENANCE MODE")
        print(f"System time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Patients    : {len(PATIENT_REGISTRY)}")
        live_patient = next((p for p in PATIENT_REGISTRY if p.get('live', False)), None)
        print(f"Live patient: {live_patient.get('name', '--') if live_patient else '--'}")
        print("\n[1]  System status & JSON health check")
        print("[2]  Patient registry")
        print("[3]  View live patient history")
        print("[4]  Export all histories to CSV")
        print("[5]  View model metrics")
        print("[6]  Clear all history files")
        print("[7]  Clear status / reset JSON files")
        print("[8]  Show GPIO pin map")
        print("[9]  Check disk space")
        print("[0]  Exit BIOS")

        choice = input("\nEnter choice: ").strip()
        if choice == "1":
            _bios_status_health_check()
        elif choice == "2":
            _bios_show_registry()
        elif choice == "3":
            _bios_view_live_history()
        elif choice == "4":
            _bios_export_histories_csv()
        elif choice == "5":
            _bios_view_model_metrics()
        elif choice == "6":
            _bios_clear_histories()
        elif choice == "7":
            _bios_reset_status_json()
        elif choice == "8":
            _bios_show_gpio_pin_map()
        elif choice == "9":
            _bios_check_disk_space()
        elif choice == "0":
            print("Exiting BIOS mode.")
            break
        else:
            print("Invalid choice. Try again.")

        input("\nPress Enter to continue...")

def compute_and_write_metrics(brain, samples):
    """
    Evaluate the hybrid model against all loaded samples and write
    neurowatch_metrics.json so the dashboard Metrics tab populates.
    Skips gracefully if there are fewer than 2 classes in the data.
    """
    if not samples:
        return

    print("📊 Computing model metrics...")
    X, y_true, y_pred = [], [], []

    for raw_data, label in samples:
        try:
            feats = brain.extract_features_auto(raw_data)
            pred, _ = brain.predict(feats)
            X.append(feats)
            y_true.append(label)
            y_pred.append(pred)
        except Exception:
            continue

    if len(set(y_true)) < 2:
        print("⚠️  Not enough class variety to compute metrics — skipping.")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc      = accuracy_score(y_true, y_pred)
    prec     = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec      = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1v      = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm       = confusion_matrix(y_true, y_pred)

    # Specificity: TN / (TN + FP) using class 0 as "negative"
    tn = cm[0, 0] if cm.shape[0] > 0 else 0
    fp = cm[0, 1:].sum() if cm.shape[0] > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Seizure-specific recall (class 2)
    seiz_rec = (recall_score(y_true, y_pred, labels=[2], average="macro",
                             zero_division=0) if 2 in y_true else 0.0)

    print(f"  Acc:{acc*100:.1f}%  F1:{f1v*100:.1f}%  "
          f"Prec:{prec*100:.1f}%  SeizRec:{seiz_rec*100:.1f}%")

    try:
        with open(METRICS_JSON, "w") as f:
            json.dump({
                "timestamp":        datetime.now().isoformat(),
                "accuracy":         round(float(acc),  3),
                "precision":        round(float(prec), 3),
                "recall":           round(float(rec),  3),
                "f1":               round(float(f1v),  3),
                "specificity":      round(float(spec), 3),
                "seizure_recall":   round(float(seiz_rec), 3),
                "confusion_matrix": cm.tolist(),
                "n_samples":        len(y_true),
                "class_counts": {
                    CLASSES[c]: int(np.sum(y_true == c))
                    for c in np.unique(y_true)
                },
            }, f)
        print(f"✅ Metrics written → {METRICS_JSON}")
    except Exception as e:
        print(f"⚠️  Metrics write error: {e}")


# =============================================================================
# DATA LOADER & MAIN LOOP
# =============================================================================
def main():
    if args.bios:
        run_bios_mode()
        return

    hw = HardwareManager()
    hw.update_lcd(["NeuroWatch v1.3", "Universal Path Fix", "", "Wait..."])
    
    try:
        brain = Brain()
    except Exception as e:
        print(f"❌ Load Error: {e}")
        hw.update_lcd(["LOAD ERROR", "Check /models", str(e)[:20], ""])
        return

    live_samples = []
    MENDELEY_LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 1}

    if brain.is_mendeley:
        # ── Mendeley mode: load pre-processed .npy windows ──────────────────
        npy_dir = (args.data if args.data and os.path.isdir(args.data)
                   else os.path.join(BASE_DIR, "data", "Mendelay dataset", "Npy_files_preictal"))
        print(f"📂 Mendeley mode — loading .npy files from: {npy_dir}")
        parts_x, parts_y = [], []
        for xname, yname in [("x_train.npy", "y_train.npy"), ("x_test.npy", "y_test.npy")]:
            xp = os.path.join(npy_dir, xname)
            yp = os.path.join(npy_dir, yname)
            if os.path.exists(xp) and os.path.exists(yp):
                parts_x.append(np.load(xp))
                parts_y.append(np.load(yp))
                print(f"  ✅ {xname}")
        if parts_x:
            X_all = np.concatenate(parts_x, axis=0)
            y_all = np.concatenate(parts_y, axis=0)
            for sample, lbl in zip(X_all, y_all):
                live_samples.append((sample, MENDELEY_LABEL_MAP[int(lbl)]))
            print(f"✅ Loaded {len(live_samples)} Mendeley samples.")
        else:
            print(f"❌ No .npy files found in {npy_dir}")
    else:
        # ── Bonn mode: scan for .edf / .txt files ───────────────────────────
        print(f"📂 Scanning for EEG (.edf/.txt) in: {DATA_PATH}")
        search_dirs = [DATA_PATH]
        for root, dirs, _ in os.walk(DATA_PATH):
            for d in dirs:
                if d in CLASS_MAP:
                    search_dirs.append(os.path.join(root, d))
        search_dirs = list(dict.fromkeys(search_dirs))

        for d in search_dirs:
            if not os.path.exists(d): continue
            files = glob.glob(os.path.join(d, "*.txt")) + glob.glob(os.path.join(d, "*.edf"))
            for f in files:
                try:
                    folder_name = os.path.basename(os.path.dirname(f))
                    label = CLASS_MAP.get(folder_name, 0)
                    if f.endswith('.edf'):
                        raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
                        if raw.info['sfreq'] != TARGET_FS:
                            raw.resample(TARGET_FS)
                        data = raw.get_data()[0]
                        if len(data) >= TARGET_FS:
                            live_samples.append((data[:TARGET_FS], label))
                    else:
                        sig = np.loadtxt(f)
                        n_new = int(len(sig) * TARGET_FS / 173.61)
                        sig = resample(sig, n_new)
                        live_samples.append((sig, label))
                except Exception as e:
                    print(f"⚠️ Skip {os.path.basename(f)}: {e}")

    if not live_samples:
        print("❌ No valid files found in path.")
        hw.update_lcd(["ERROR", "No Data Found", "Check Path", ""])
        return

    print(f"✅ Loaded {len(live_samples)} files. Monitoring active.")

    # Compute metrics in the background — does not block the live loop
    threading.Thread(
        target=compute_and_write_metrics,
        args=(brain, live_samples),
        daemon=True
    ).start()

    idx = 0
    live_state_prev = None
    live_state_since = time.time()
    seizure_sms_status = None  # None=not attempted, False=not sent, True=sent
    _had_seizure       = False  # True after a seizure episode — triggers "back to normal" SMS

    live_patient = next((p for p in PATIENT_REGISTRY if p.get("live", False)), None)
    live_patient_id = live_patient.get("id", "P001") if live_patient else "P001"

    try:
        while True:
            raw_data, true_label = live_samples[idx % len(live_samples)]
            feats = brain.extract_features_auto(raw_data)
            pred, conf = brain.predict(feats)

            state_text = CLASSES[pred]
            true_text  = CLASSES[true_label]

            # Track continuous duration in current state for LCD escalation logic.
            if state_text != live_state_prev:
                prev_state      = live_state_prev
                live_state_prev = state_text
                live_state_since = time.time()
                if state_text == "Seizure":
                    _had_seizure = True
                else:
                    seizure_sms_status = None
                # Clearing SMS — only fires if the seizure SMS was actually sent (30s threshold reached)
                if state_text == "Normal" and _had_seizure and seizure_sms_status is True and SMS_AVAILABLE and live_patient:
                    _had_seizure = False
                    try:
                        send_sms_alert(
                            alert_type="normal",
                            patient_name=live_patient.get("name", "Live Patient"),
                            patient_id=live_patient.get("id", live_patient_id),
                            ward=live_patient.get("ward", "Neuro"),
                            confidence=float(conf),
                            force=False,
                        )
                    except Exception:
                        pass
            state_duration = time.time() - live_state_since

            # If seizure persists, try notifying doctor once per seizure episode.
            if state_text == "Seizure" and state_duration >= SEIZURE_LONG_SECONDS and seizure_sms_status is None:
                if SMS_AVAILABLE and live_patient:
                    maps_link = ""
                    if live_patient.get("gps_lat") and live_patient.get("gps_lng"):
                        maps_link = f"https://maps.google.com/?q={live_patient.get('gps_lat')},{live_patient.get('gps_lng')}"
                    try:
                        sms_result = send_sms_alert(
                            alert_type="seizure",
                            patient_name=live_patient.get("name", "Live Patient"),
                            patient_id=live_patient.get("id", live_patient_id),
                            ward=live_patient.get("ward", "Neuro"),
                            confidence=float(conf),
                            location=live_patient.get("gps_label", live_patient.get("ward", "")),
                            maps_link=maps_link,
                            emergency_contact=live_patient.get("emergency_contact", ""),
                            force=False,
                        )
                        seizure_sms_status = bool(sms_result.get("sent", False))
                    except Exception:
                        seizure_sms_status = False
                else:
                    seizure_sms_status = False

            if state_text == "Pre-Seizure" and state_duration >= PRESEIZURE_LONG_SECONDS:
                row4 = "TAKE MEDICATION"
            elif state_text == "Seizure" and state_duration >= SEIZURE_LONG_SECONDS:
                row4 = "DR NOTIFIED" if seizure_sms_status else "CONTACT DR"
            else:
                row4 = f"T:{datetime.now().strftime('%H:%M:%S')}"

            hw.set_alarm(pred)

            # Terminal scan line — one row per sample so you can watch the model run
            _state_tag = {
                "Normal":      "  NORMAL    ",
                "Pre-Seizure": "  PRE-SZ ⚡ ",
                "Seizure":     "  SEIZURE 🔴",
            }.get(state_text, f"  {state_text:<10}")
            bands = Brain.band_summary(feats)
            _match = "✓" if pred == true_label else f"✗ (true={true_text})"
            print(
                f"[{idx % len(live_samples):05d}/{len(live_samples)}] "
                f"{datetime.now().strftime('%H:%M:%S')}  │"
                f"{_state_tag}  │  Conf: {conf*100:5.1f}%  │  {_match}  │  "
                f"δ:{bands[0]:.3f}  θ:{bands[1]:.3f}  α:{bands[2]:.3f}  β:{bands[3]:.3f}"
                + (f"  ⏱ {state_duration:.0f}s" if state_duration > 3 else "")
            )

            # LCD/Terminal Display
            hw.update_lcd([
                "LIVE: John M.",
                f"STATE: {state_text}",
                f"CONF: {conf*100:.1f}%",
                row4
            ])
            
            # Update Shared JSON — status
            try:
                with open(STATUS_JSON, 'w') as f:
                    json.dump({
                        "phase":   "DETECTING",
                        "state":   state_text,
                        "conf":    round(conf, 4),
                        "message": f"Monitoring {len(PATIENT_REGISTRY)} patients",
                        "time":    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }, f)
            except: pass

            # Update patients JSON — one entry per patient.
            # Live patient (live=True) gets real ML prediction.
            # Demo patients get a weighted random simulation so the
            # dashboard ward looks populated with no extra hardware.
            try:
                patients_out = {}
                for p in PATIENT_REGISTRY:
                    is_live = p.get("live", False)
                    if is_live:
                        p_pred       = pred
                        p_state      = state_text
                        p_conf       = round(conf, 3)
                        bands        = brain.band_summary(feats)
                        p_delta      = round(float(bands[0]), 4)
                        p_theta      = round(float(bands[1]), 4)
                        p_alpha      = round(float(bands[2]), 4)
                        p_beta       = round(float(bands[3]), 4)
                        conf_normal  = round(p_conf if p_pred == 0 else 1.0 - p_conf, 3)
                        conf_pre     = round(p_conf if p_pred == 1 else 0.05, 3)
                        conf_seizure = round(p_conf if p_pred == 2 else 0.03, 3)
                        p_sms_sent   = bool(seizure_sms_status) if p_state == "Seizure" else False
                    else:
                        p_pred       = random.choices([0, 1, 2], weights=[0.75, 0.18, 0.07])[0]
                        p_state      = CLASSES[p_pred]
                        p_conf       = round(random.uniform(0.62, 0.96), 3)
                        p_delta      = round(random.uniform(0.01, 0.80), 4)
                        p_theta      = round(random.uniform(0.005, 0.40), 4)
                        p_alpha      = round(random.uniform(0.002, 0.30), 4)
                        p_beta       = round(random.uniform(0.001, 0.20), 4)
                        conf_normal  = round(p_conf if p_pred == 0 else random.uniform(0.02, 0.15), 3)
                        conf_pre     = round(p_conf if p_pred == 1 else random.uniform(0.02, 0.12), 3)
                        conf_seizure = round(p_conf if p_pred == 2 else random.uniform(0.01, 0.08), 3)
                        p_sms_sent   = False

                    patients_out[p["id"]] = {
                        "patient_id":        p["id"],
                        "patient_name":      p["name"],
                        "age":               p["age"],
                        "diagnosis":         p["diagnosis"],
                        "ward":              p["ward"],
                        "bed":               p["bed"],
                        "emergency_contact": p.get("emergency_contact", ""),
                        "state":             p_state,
                        "confidence":        p_conf,
                        "conf_normal":       conf_normal,
                        "conf_preseizure":   conf_pre,
                        "conf_seizure":      conf_seizure,
                        "delta":             p_delta,
                        "theta":             p_theta,
                        "alpha":             p_alpha,
                        "beta":              p_beta,
                        "gps_lat":           p.get("gps_lat", ""),
                        "gps_lng":           p.get("gps_lng", ""),
                        "gps_label":         p.get("gps_label", ""),
                        "sms_sent":          p_sms_sent,
                        "timestamp":         datetime.now().isoformat(),
                    }

                with open(PATIENTS_JSON, 'w') as f:
                    json.dump(patients_out, f)

            except Exception as e:
                print(f"⚠️  patients JSON write error: {e}")

            # Write per-patient history files (appends one record per tick)
            try:
                for p in PATIENT_REGISTRY:
                    pid        = p["id"]
                    pdata      = patients_out.get(pid, {})
                    hist_path  = os.path.join(BASE_DIR, "json", f"neurowatch_history_{pid}.json")
                    try:
                        with open(hist_path, "r") as f:
                            history = json.load(f)
                    except Exception:
                        history = []
                    history.append({
                        "timestamp":       datetime.now().isoformat(),
                        "state":           pdata.get("state", "Normal"),
                        "confidence":      pdata.get("confidence", 0),
                        "conf_normal":     pdata.get("conf_normal", 0),
                        "conf_preseizure": pdata.get("conf_preseizure", 0),
                        "conf_seizure":    pdata.get("conf_seizure", 0),
                        "delta":           pdata.get("delta", 0),
                        "theta":           pdata.get("theta", 0),
                        "alpha":           pdata.get("alpha", 0),
                        "beta":            pdata.get("beta", 0),
                        "sms_sent":        pdata.get("sms_sent", False),
                    })
                    if len(history) > 200:
                        history = history[-200:]
                    with open(hist_path, "w") as f:
                        json.dump(history, f)
            except Exception as e:
                print(f"⚠️  history write error: {e}")

            idx += 1

            # Every 50 ticks (~75 seconds) check if new data was added.
            # If so, retrain automatically and reload models.
            if idx % 50 == 0:
                retrained = brain.check_and_retrain(hw)
                if retrained:
                    compute_and_write_metrics(brain, live_samples)

            time.sleep(1.5)
            
    except KeyboardInterrupt:
        print("\n🛑 System Stopped.")
    finally:
        hw.cleanup()

if __name__ == "__main__":
    main()