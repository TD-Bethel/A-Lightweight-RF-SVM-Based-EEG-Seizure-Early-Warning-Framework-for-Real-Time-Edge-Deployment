# ══════════════════════════════════════════════════════════════════════════════
# NEUROWATCH  —  MAIN DETECTION ENGINE  (main_pi_bios_v15.py)
# ══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE
# -------
# This is the CORE file of the NeuroWatch system.  It does everything:
#   1. Loads the trained ML models (Random Forest + SVM) from disk.
#   2. Reads EEG signal files in a loop, extracts features, and predicts the
#      brain state: Normal / Pre-Seizure / Seizure.
#   3. Controls the physical Raspberry Pi hardware (LED lights, buzzer, LCD).
#   4. Sends SMS alerts to the doctor when a seizure is detected.
#   5. Writes JSON files that the web dashboard reads in real time.
#   6. Provides a maintenance "BIOS" menu you can access from the terminal.
#
# HOW TO RUN (examples)
# ---------------------
#   Normal monitoring:
#       python src/main_pi_bios_v15.py
#
#   Point to specific data / model folders:
#       python src/main_pi_bios_v15.py --data /path/to/eeg --models /path/to/models
#
#   Open the maintenance BIOS menu:
#       python src/main_pi_bios_v15.py --bios
#
# STUDENT QUICK-REFERENCE — where to look for common changes
# -----------------------------------------------------------
#   Adjusting detection thresholds  → Brain.predict()  and ensemble weights
#   Changing alarm timing           → PRESEIZURE_LONG_SECONDS / SEIZURE_LONG_SECONDS
#   Changing SMS timing             → SEIZURE_LONG_SECONDS
#   Adding / editing patients       → patients.py  (this file uses it read-only)
#   Changing LCD messages           → hw.update_lcd() calls inside main()
#   Changing model folder           → MODEL_PATH constant or --models flag
#   Changing GPIO pin numbers       → PIN_LED_GREEN, PIN_LED_RED, PIN_BUZZER
#   Changing buzzer pattern speed   → HardwareManager._buzzer_worker()
# ══════════════════════════════════════════════════════════════════════════════


# ── IMPORTS ────────────────────────────────────────────────────────────────────
# Standard library modules: file paths, system, timing, JSON, CLI args, file
# search, random numbers, warnings, threads, file copy, and CSV writing.
import os, sys, time, json, argparse, glob, random, warnings, threading, shutil, csv

# Make sure Python can find modules in the project root folder (one level up).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# RFImportanceSelector is a custom class defined in neurowatch_selector.py.
# It MUST be imported here even if not used directly, because joblib needs it
# to be in scope when it unpickles (loads) the saved selector.pkl model file.
from neurowatch_selector import RFImportanceSelector  # needed for joblib unpickling

# Numerical computing and model loading.
import numpy as np
import joblib

# MNE is a library for reading EEG files in .edf format.
import mne  # Required: pip install mne

from datetime import datetime

# Signal processing: butter designs a filter, filtfilt applies it without phase
# distortion, welch computes the power spectrum, resample changes the Hz rate.
from scipy.signal import butter, filtfilt, welch, resample

# ── PATIENT REGISTRY IMPORT ───────────────────────────────────────────────────
# Try to import the patient list from patients.py (in the project root or src/).
# If that file is missing, fall back to a single hard-coded demo patient so the
# system still starts without crashing.
#
# TO ADD OR EDIT PATIENTS: open patients.py and modify PATIENT_REGISTRY there.
# Do NOT modify the fallback list below — it is only a last-resort safety net.
#
# Key patient fields:
#   "live": True  — this patient receives the real ML prediction each tick.
#           False — patient shows simulated/random data on the dashboard.
#   "eeg_slice": (start_sec, end_sec) — portion of EEG file to use (Bonn mode).
#   "gps_lat/lng" — used to build the Google Maps link in the SMS alert.
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

# ── SMS NOTIFIER IMPORT ───────────────────────────────────────────────────────
# Try to import the SMS sending function from sms_notifier.py.
# If the file is missing, SMS_AVAILABLE = False and the system runs without
# text messages — no crash, just no alerts sent to the doctor.
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
# RPi.GPIO controls the Raspberry Pi's physical GPIO pins (LEDs, buzzer).
# CharLCD drives the 20x4 character LCD screen over those GPIO pins.
# If these libraries are not installed (e.g. running on Windows for testing),
# GPIO_AVAILABLE is set to False and hardware calls are silently skipped —
# LCD output is printed to the terminal instead so you can still test the logic.
try:
    import RPi.GPIO as GPIO
    from RPLCD.gpio import CharLCD
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

# ── SECTION: CONFIGURATION & UNIVERSAL PATH SELECTION ─────────────────────────
# =============================================================================
# CONFIGURATION & UNIVERSAL PATH SELECTION
# =============================================================================

# CLI argument parser — lets you override default paths when launching from terminal.
# --data    : folder containing .edf or .txt EEG signal files
# --models  : folder containing scaler.pkl, svm_model.pkl, rf_model.pkl
# --bios    : flag that opens the maintenance BIOS menu instead of monitoring
parser = argparse.ArgumentParser(description="NeuroWatch Universal Engine")
parser.add_argument("--data", type=str, help="Path to folder containing .edf or .txt files")
parser.add_argument("--models", type=str, help="Path to .pkl models folder")
parser.add_argument("--bios", action="store_true", help="Open maintenance BIOS mode")
args = parser.parse_args()

# 1. BASE_DIR points to project root (one level above src/)
# os.path.abspath(__file__) gets the full path of this script.
# dirname() strips the filename to get the folder; called twice to go up two levels.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = PROJECT_ROOT

# 2. FIX: STATUS_JSON now points to your project folder (Works on Windows & Pi)
# STATUS_JSON   — the dashboard reads this every second to show live detection state.
# PATIENTS_JSON — the dashboard reads this to populate the ward patient list.
STATUS_JSON   = os.path.join(BASE_DIR, "json", "neurowatch_status.json")
PATIENTS_JSON = os.path.join(BASE_DIR, "json", "neurowatch_patients.json")

# 3. Set Data and Model Paths
# If --data was provided on the command line, use that; otherwise use default "data" folder.
DATA_PATH = os.path.abspath(args.data) if args.data else os.path.join(BASE_DIR, "data")

# MODEL_PATH: folder that must contain scaler.pkl, svm_model.pkl, rf_model.pkl.
# TO SWITCH MODEL SETS: change "MODELS_FS75" to your model folder name,
# or launch with --models /path/to/your/models to override at runtime.
MODEL_PATH = os.path.abspath(args.models) if args.models else os.path.join(BASE_DIR, "models", "MODELS_FS75")

# 4. Model Constants
# TARGET_FS: the sampling rate (Hz) all signals are resampled to before feature
# extraction. The models were trained at 128 Hz — do NOT change this value
# unless you also retrain the models from scratch at the new rate.
TARGET_FS = 128

# CLASSES: the three possible prediction outputs, indexed 0 / 1 / 2.
# The integer returned by brain.predict() maps directly into this list.
CLASSES = ["Normal", "Pre-Seizure", "Seizure"]

# CLASS_MAP: maps Bonn dataset folder names to their class numbers.
# O = eyes Open (Normal), N = Normal, S = Seizure focus, F = Focus (Pre-Sz), Z = Seizure.
CLASS_MAP = {"O": 0, "N": 0, "S": 1, "F": 1, "Z": 2}

# GPIO Pins — BCM numbering (refers to chip GPIO numbers, not physical header positions).
# TO CHANGE WIRING: update these numbers AND rewire the physical connections to match.
PIN_LED_GREEN, PIN_LED_RED, PIN_BUZZER = 27, 22, 4

# Suppress irrelevant library warnings so the terminal output stays clean.
warnings.filterwarnings("ignore")

# Prolonged-state escalation thresholds (seconds).
# PRESEIZURE_LONG_SECONDS: how many continuous seconds of "Pre-Seizure" must pass
# before the LCD row 4 changes from the clock to "TAKE MEDICATION".
# Lower = message appears sooner.  Raise to require a longer warning period.
PRESEIZURE_LONG_SECONDS = 12

# SEIZURE_LONG_SECONDS: how many continuous seconds of "Seizure" must pass before
# an SMS is sent to the doctor and the LCD shows "DR NOTIFIED".
# Lower = doctor alerted sooner.  Raise = fewer false SMS alerts on brief spikes.
SEIZURE_LONG_SECONDS = 30

# ── SECTION: HARDWARE & ML CORES ──────────────────────────────────────────────
# =============================================================================
# HARDWARE & ML CORES
# =============================================================================

# ── CLASS: HardwareManager ────────────────────────────────────────────────────
class HardwareManager:
    """
    Manages all physical hardware attached to the Raspberry Pi:
      - Green LED  (GPIO 27) — lit when brain state is Normal.
      - Red LED    (GPIO 22) — lit when state is Pre-Seizure or Seizure.
      - Buzzer     (GPIO  4) — beeps in patterns that differ by alert level.
      - 20x4 LCD screen     — displays patient name, state, confidence, status.

    When GPIO libraries are not available (e.g. running on Windows for testing),
    all hardware calls are silently skipped and LCD text is printed to terminal.
    """

    def __init__(self):
        # lcd holds the CharLCD object when real hardware is connected, else None.
        self.lcd = None

        # _alarm_mode stores the current alert level used by the buzzer thread:
        #   0 = Normal     (green LED on,  buzzer silent)
        #   1 = Pre-Seizure (red LED on,  slow single beep every second)
        #   2 = Seizure    (red LED on,   rapid 5 Hz beeping)
        self._alarm_mode = 0  # 0=normal, 1=pre-seizure, 2=seizure

        # _stop_buzzer is set True on shutdown to tell the buzzer thread to exit.
        self._stop_buzzer = False
        self._buzzer_thread = None

        if GPIO_AVAILABLE:
            # BCM mode means pin numbers refer to the chip's GPIO numbers,
            # not the physical header pin positions on the board.
            GPIO.setmode(GPIO.BCM)
            GPIO.setup([PIN_LED_GREEN, PIN_LED_RED, PIN_BUZZER], GPIO.OUT, initial=GPIO.LOW)

            # Initialise the 20x4 LCD with the GPIO pins it is wired to.
            # pin_rs=26: Register Select  |  pin_e=19: Enable
            # pins_data=[13,6,5,20]: 4-bit data bus (D4, D5, D6, D7)
            # TO CHANGE LCD WIRING: update pin numbers here AND rewire to match.
            try:
                self.lcd = CharLCD(pin_rs=26, pin_e=19, pins_data=[13, 6, 5, 20],
                                   numbering_mode=GPIO.BCM, cols=20, rows=4)
                self._reset_lcd()
            except: print("⚠️ LCD Hardware not found.")

            # Start the buzzer in a background daemon thread so it runs
            # independently of the 1.5-second main detection loop cycle.
            self._buzzer_thread = threading.Thread(target=self._buzzer_worker, daemon=True)
            self._buzzer_thread.start()

    def _reset_lcd(self):
        """Full 20x4 LCD reset — clears all rows, homes cursor, cycles backlight.

        Called once at startup to remove any leftover characters from a previous
        run. Cycles the backlight off/on and overwrites all 80 character cells
        with spaces to eliminate "ghost" characters that sometimes persist on
        cheap LCD modules after a power cycle.
        """
        if not self.lcd:
            return
        try:
            self.lcd.backlight_enabled = False
            time.sleep(0.1)
            self.lcd.clear()
            time.sleep(0.05)
            self.lcd.home()
            # Overwrite all 80 cells with spaces to flush any ghost characters
            for row in range(4):
                self.lcd.cursor_pos = (row, 0)
                self.lcd.write_string(" " * 20)
            self.lcd.clear()
            self.lcd.home()
            self.lcd.backlight_enabled = True
            time.sleep(0.05)
        except Exception as e:
            print(f"⚠️ LCD reset error: {e}")

    def update_lcd(self, rows):
        """
        Write up to 4 text rows onto the LCD screen.
        Each string is truncated to 20 characters and padded with spaces to fill
        the full row width, preventing old characters from showing at the end.

        When GPIO is unavailable (Windows / no hardware), prints to terminal so
        you can see what would appear on the screen during testing.

        TO CHANGE LCD MESSAGES: call hw.update_lcd(["row1","row2","row3","row4"])
        with your desired text anywhere in the code.
        """
        if self.lcd:
            for i, text in enumerate(rows[:4]):
                self.lcd.cursor_pos = (i, 0)
                self.lcd.write_string(text[:20].ljust(20))
        else:
            # Clean terminal print for Windows testing
            print("\n" + "-"*30 + "\n" + "\n".join(rows) + "\n" + "-"*30)

    def set_alarm(self, level):
        """
        Set LEDs and buzzer to the correct state for the current prediction level.
          level 0 = Normal:      green LED on,  red LED off, buzzer silent.
          level 1 = Pre-Seizure: green LED off, red LED on,  slow beep.
          level 2 = Seizure:     green LED off, red LED on,  rapid beep.
        The actual buzzer timing is handled by _buzzer_worker running in a thread.
        """
        if not GPIO_AVAILABLE: return
        self._alarm_mode = int(level)
        GPIO.output(PIN_LED_GREEN, GPIO.HIGH if level == 0 else GPIO.LOW)
        GPIO.output(PIN_LED_RED, GPIO.HIGH if level > 0 else GPIO.LOW)

    def _buzzer_worker(self):
        """Runs continuous buzzer patterns independent of the main loop speed.

        This method runs forever in a background thread, checking _alarm_mode
        60 times per second (every 0.05 s) and driving the buzzer pin accordingly.

        Seizure (mode 2):     5 Hz square wave — rapid continuous beeping.
          int(t * 10) % 2 creates a pattern that flips every 0.1 s = 5 Hz.
        Pre-Seizure (mode 1): one 0.18-second beep every 1 second.
          (t % 1.0) < 0.18 is True for the first 180 ms of every second.
        Normal (mode 0):      silent.

        TO CHANGE BUZZER SPEED: adjust the multiplier in (int(t * 10) % 2)
        for seizure rate, or change 1.0 and 0.18 for pre-seizure timing.
        """
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
        """
        Safely shut down all hardware when the program exits (Ctrl+C or error).
        Signals the buzzer thread to stop, waits briefly for it to finish,
        then calls GPIO.cleanup() to release all pins so they are not left in
        a driven state that could damage components between runs.
        """
        self._stop_buzzer = True
        if self._buzzer_thread and self._buzzer_thread.is_alive():
            self._buzzer_thread.join(timeout=0.3)
        if GPIO_AVAILABLE:
            GPIO.cleanup()

# ── CLASS: Brain ──────────────────────────────────────────────────────────────
class Brain:
    """
    The machine learning core of NeuroWatch. Responsible for:
      1. Loading the trained model files (scaler, SVM, Random Forest) from disk.
      2. Extracting numerical features from a raw EEG signal array.
      3. Combining SVM and RF predictions into one weighted ensemble vote.
      4. Auto-retraining if new EEG data files appear in DATA_PATH.

    Two feature extractors exist because the system supports two datasets:
      - Bonn dataset     (1 channel,  .txt files)  → extract_features()
      - Mendeley dataset (19 channels, .npy files) → extract_features_mendeley()
    The correct extractor is chosen automatically by extract_features_auto().
    """

    def __init__(self):
        # Print the model folder path at startup — helps debug "wrong model" issues.
        print(f"📦 Loading Models from: {MODEL_PATH}")

        # Load the three required model files from MODEL_PATH.
        # scaler.pkl    — normalises feature values to the range the models expect.
        # svm_model.pkl — trained Support Vector Machine classifier.
        # rf_model.pkl  — trained Random Forest classifier.
        # TO SWITCH MODEL SETS: change MODEL_PATH above or pass --models on the CLI.
        try:
            self.scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
            self.svm    = joblib.load(os.path.join(MODEL_PATH, "svm_model.pkl"))
            self.rf     = joblib.load(os.path.join(MODEL_PATH, "rf_model.pkl"))
        except Exception as e:
            raise FileNotFoundError(f"Model files missing in {MODEL_PATH}: {e}")

        # Detect which dataset this model was trained on by counting its input features.
        # Bonn models use 13 features (1 channel).
        # Mendeley models use 76 or more features (19 channels × 13 features each).
        self.n_features  = int(self.scaler.n_features_in_)
        self.is_mendeley = (self.n_features >= 76)
        print(f"📊 Model type: {'Mendeley (%d features, 19ch)' % self.n_features if self.is_mendeley else 'Bonn (4 features, 1ch)'}")

        # Load auto-tuned ensemble weights if available, else fall back to defaults.
        # ensemble_weights.pkl stores how much to trust each model's probability output.
        # svm_w + rf_w do not need to sum to 1, but typically do (40% SVM + 60% RF).
        # normal_threshold: minimum combined probability required to classify as Normal.
        weights_path = os.path.join(MODEL_PATH, "ensemble_weights.pkl")
        if os.path.exists(weights_path):
            w = joblib.load(weights_path)
            self.svm_w            = float(w.get("svm_w",            0.4))
            self.rf_w             = float(w.get("rf_w",             0.6))
            self.normal_threshold = float(w.get("normal_threshold", 0.5))
            print(f"📊 Ensemble weights: SVM {self.svm_w:.0%} + RF {self.rf_w:.0%} (auto-tuned)  |  Normal threshold: {self.normal_threshold:.2f}")
        else:
            # Default weights if no ensemble_weights.pkl exists.
            # RF gets 60% trust, SVM gets 40%. Adjust experimentally if needed.
            self.svm_w, self.rf_w, self.normal_threshold = 0.4, 0.6, 0.50

        # selector.pkl is an optional feature-selection filter saved alongside models.
        # When present, it reduces the feature vector to only the most informative
        # features before prediction, which can improve accuracy on smaller datasets.
        selector_path = os.path.join(MODEL_PATH, "selector.pkl")
        if os.path.exists(selector_path):
            self.selector = joblib.load(selector_path)
            print(f"📊 Feature selector: {self.selector.k} features selected")
        else:
            self.selector = None

        # Track dataset fingerprint so we retrain only when data actually changes.
        # A "fingerprint" is a string encoding the path and modification time of
        # every EEG file. If any file is added or changed, the string changes too.
        self._data_fingerprint = self._fingerprint(DATA_PATH)

    # ------------------------------------------------------------------
    # AUTO-RETRAIN: called once at startup and checked every N ticks
    # ------------------------------------------------------------------
    def check_and_retrain(self, hw=None):
        """
        Re-runs train_and_export.py if new EEG files have been added to
        DATA_PATH since the last time we checked. Reloads models afterwards.

        Called every 50 detection ticks (~75 seconds) from the main loop.
        If the data fingerprint has not changed, it returns immediately (fast path).
        If new files are found, it triggers a full retrain via subprocess, then
        hot-reloads the updated model files so monitoring continues uninterrupted.

        Returns True if models were retrained and reloaded, False otherwise.
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

        # Run the training script as a separate process so it cannot crash this one.
        # capture_output=True collects stdout and stderr for printing below.
        import subprocess
        result = subprocess.run(
            [sys.executable, trainer, "--data", DATA_PATH],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(result.stdout[-800:])   # show last 800 chars of output
            # Reload updated models into this Brain instance (hot-swap, no restart needed).
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

        Walks the entire directory tree under 'path', collects the full path
        and last-modified timestamp of every .txt and .edf file, sorts them,
        and joins into one string. Any addition, removal, or file change
        produces a different string, triggering a retrain in check_and_retrain().
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
        """13-feature Bonn extractor — must stay in sync with train_and_export.py.

        Takes a 1-D numpy array of raw EEG samples (1 second at 128 Hz = 128 values).
        Returns a list of 13 numbers that describe the signal mathematically.
        These same 13 features must match what was used at training time exactly.

        Features produced (in order):
          [0-4]  Normalised band powers: delta(1-4Hz), theta(4-8Hz), alpha(8-13Hz),
                 beta(13-30Hz), gamma(30-45Hz).  They sum to 1.0.
          [5]    Hjorth Activity   — overall signal variance (power).
          [6]    Hjorth Mobility   — how fast the signal changes on average.
          [7]    Hjorth Complexity — irregularity of the rate of change.
          [8]    Skewness          — asymmetry of the amplitude distribution.
          [9]    Kurtosis          — how "peaked" the distribution is (high in seizures).
          [10]   RMS               — root-mean-square amplitude (overall energy level).
          [11]   Zero-Crossing Rate — how often the signal crosses zero (frequency proxy).
          [12]   Spectral Entropy  — randomness of the power spectrum (low = seizure).
        """
        from scipy.stats import skew, kurtosis as _kurtosis

        # Design a 4th-order Butterworth bandpass filter covering 0.5 to 45 Hz.
        # This removes slow DC drift (below 0.5 Hz) and high-frequency noise (above 45 Hz).
        # nyq is the Nyquist frequency — half the sampling rate.
        nyq  = 0.5 * TARGET_FS
        b, a = butter(4, [0.5 / nyq, 45 / nyq], btype="band")

        # Apply the filter both forward and backward (filtfilt) to avoid phase shift.
        filt   = filtfilt(b, a, data)

        # Welch's method: divide signal into overlapping segments, compute the FFT
        # of each, and average the power spectra. Gives a stable estimate of power
        # at each frequency. f = frequency axis (Hz), psd = power at each frequency.
        f, psd = welch(filt, fs=TARGET_FS, nperseg=TARGET_FS)

        # Average power within each EEG frequency band.
        bp = np.array([
            np.mean(psd[(f >= 1)  & (f <= 4)]),   # Delta  (1–4 Hz)  — deep sleep, seizure
            np.mean(psd[(f >= 4)  & (f <= 8)]),   # Theta  (4–8 Hz)  — drowsiness
            np.mean(psd[(f >= 8)  & (f <= 13)]),  # Alpha  (8–13 Hz) — relaxed wakefulness
            np.mean(psd[(f >= 13) & (f <= 30)]),  # Beta   (13–30 Hz)— active thinking
            np.mean(psd[(f >= 30) & (f <= 45)]),  # Gamma  (30–45 Hz)— high cognition
        ])

        # Normalise: divide each band power by the total so they sum to 1.
        # This makes features comparable across patients with different signal amplitudes.
        total   = bp.sum()
        bp_norm = bp / total if total > 0 else bp

        # Hjorth parameters use the first (d1) and second (d2) derivatives of the signal.
        # Adding 1e-10 to variances prevents division-by-zero on flat/silent signals.
        d1, d2   = np.diff(data), np.diff(np.diff(data))
        var_x    = np.var(data) + 1e-10
        var_d1   = np.var(d1)   + 1e-10
        var_d2   = np.var(d2)   + 1e-10
        activity   = float(var_x)
        mobility   = float(np.sqrt(var_d1 / var_x))
        complexity = float(np.sqrt(var_d2 / var_d1) / (mobility + 1e-10))

        # Statistical shape descriptors of the amplitude distribution.
        sk   = float(skew(data))                              # asymmetry
        kurt = float(_kurtosis(data))                         # peakedness
        rms  = float(np.sqrt(np.mean(data ** 2)))             # energy
        zcr  = float(((data[:-1] * data[1:]) < 0).sum() / len(data))  # zero-crossing rate

        # Spectral entropy: how spread out (random) the frequency content is.
        # Low entropy means energy is concentrated in narrow bands — typical of seizures.
        p_n  = psd / (psd.sum() + 1e-10)
        sent = float(-np.sum(p_n * np.log2(p_n + 1e-10)) / np.log2(len(p_n) + 1))

        # Concatenate band powers and time-domain features into one flat list.
        return np.concatenate([bp_norm, [activity, mobility, complexity,
                                         sk, kurt, rms, zcr, sent]]).tolist()

    def predict(self, features):
        """
        Run the ensemble classifier on a feature vector and return the prediction.

        Steps:
          1. Reshape the feature list into a 2-D array (sklearn expects this shape).
          2. Scale features using the saved scaler (e.g. zero-mean, unit variance).
          3. Optionally reduce dimensions with the feature selector (if one was saved).
          4. Get class probability arrays from both SVM and RF independently.
          5. Combine: combined = (svm_w * p_svm) + (rf_w * p_rf).
          6. Return the class with the highest combined probability + that probability.

        Returns:
          idx  (int)   — predicted class: 0 = Normal, 1 = Pre-Seizure, 2 = Seizure.
          conf (float) — confidence between 0.0 and 1.0 (the winning class probability).

        TO ADJUST SENSITIVITY: raise rf_w / lower svm_w to trust RF more, or edit
        ensemble_weights.pkl values in the model folder.
        """
        # Reshape to (1, n_features) — sklearn classifiers require a 2-D input array.
        x_raw    = np.array([features])

        # Apply the same scaling that was used during training so features are in range.
        x_scaled = self.scaler.transform(x_raw)

        # If a feature selector exists, apply it to reduce to the most important features.
        # SVM uses the scaled version; RF uses raw values (tree models don't need scaling).
        if self.selector is not None:
            x_svm = self.selector.transform(x_scaled)
            x_rf  = self.selector.transform(x_raw)
        else:
            x_svm = x_scaled
            x_rf  = x_raw

        # predict_proba returns an array like [0.7, 0.2, 0.1] meaning
        # 70% Normal, 20% Pre-Seizure, 10% Seizure. [0] picks the first (only) row.
        p_svm    = self.svm.predict_proba(x_svm)[0]
        p_rf     = self.rf.predict_proba(x_rf)[0]

        # Weighted average: gives each model a vote proportional to its weight.
        combined = self.svm_w * p_svm + self.rf_w * p_rf

        # argmax picks the index (class) with the highest combined probability.
        idx = int(np.argmax(combined))
        return idx, float(combined[idx])

    # Internal constants for the Mendeley 19-channel feature extractor.
    # These must match the values used in train_mendeley.py exactly.
    _MENDELEY_FS = 500   # Original Mendeley dataset sampling rate in Hz.
    _WIN_SIZE    = 4097  # Window length used during training (~8 seconds at 500 Hz).
    _N_CH        = 19    # Number of EEG channels in the Mendeley dataset.
    _NPERSEG     = 128   # FFT segment length for Welch's method (frequency resolution).

    def extract_features_mendeley(self, sample):
        """Feature extraction from a (19, 500) Mendeley sample.
        Matches train_mendeley.py exactly — do not modify unless you also retrain.

        Takes a 2-D numpy array of shape (19 channels, 500 samples).
        For each of the 19 channels it computes the same 13 features as
        extract_features(), giving 19 x 13 = 247 features total.

        If model has 247 features (original Mendeley): per-channel only.
        If model has 418 features (pre-ictal Mendeley): per-channel + 171 inter-channel correlations.

        Processing steps per channel:
          1. Max-abs normalise the whole sample to [-1, 1].
          2. Resample from 500 Hz to TARGET_FS (128 Hz).
          3. Tile the resampled signal to fill _WIN_SIZE samples.
          4. Apply bandpass filter and compute band powers (5 features).
          5. Compute Hjorth parameters (3 features).
          6. Compute statistical features: skew, kurtosis, RMS, ZCR, entropy (5 features).
        """
        from scipy.stats import skew, kurtosis as _kurtosis
        nyq  = 0.5 * TARGET_FS
        b, a = butter(4, [0.5 / nyq, 45 / nyq], btype="band")
        all_feats = []
        resampled = []

        # Max-abs normalise to [-1, 1] — matches Preprocess.py training transform.
        # Dividing by the largest absolute value scales the whole 19-channel recording
        # so differences between patients and devices do not affect features.
        m = np.amax(np.abs(sample))
        if m > 0:
            sample = sample / m

        # Process each of the 19 EEG channels independently.
        for ch in range(self._N_CH):
            sig   = sample[ch]

            # Resample the channel from 500 Hz (Mendeley) down to 128 Hz (TARGET_FS).
            n_new = int(len(sig) * TARGET_FS / self._MENDELEY_FS)
            sig_r = resample(sig, n_new)

            # Tile (repeat) the resampled signal to fill _WIN_SIZE samples.
            # The Welch PSD needs a long enough window; tiling fills the gap when
            # the resampled signal is shorter than _WIN_SIZE.
            reps  = (self._WIN_SIZE // len(sig_r)) + 1
            sig_t = np.tile(sig_r, reps)[:self._WIN_SIZE]
            resampled.append(sig_r)

            # Band powers (5) — Option A normalised
            filt   = filtfilt(b, a, sig_t)
            f, psd = welch(filt, fs=TARGET_FS, nperseg=self._NPERSEG)
            bp = np.array([
                np.mean(psd[(f >= 1)  & (f <= 4)]),   # Delta
                np.mean(psd[(f >= 4)  & (f <= 8)]),   # Theta
                np.mean(psd[(f >= 8)  & (f <= 13)]),  # Alpha
                np.mean(psd[(f >= 13) & (f <= 30)]),  # Beta
                np.mean(psd[(f >= 30) & (f <= 45)]),  # Gamma
            ])
            total = bp.sum()
            bp_norm = bp / total if total > 0 else bp

            # Hjorth parameters — computed on the resampled signal (not the tiled version).
            d1, d2   = np.diff(sig_r), np.diff(np.diff(sig_r))
            var_x    = np.var(sig_r)  + 1e-10
            var_d1   = np.var(d1)     + 1e-10
            var_d2   = np.var(d2)     + 1e-10
            activity   = float(var_x)
            mobility   = float(np.sqrt(var_d1 / var_x))
            complexity = float(np.sqrt(var_d2 / var_d1) / (mobility + 1e-10))

            # Statistical features (same as the Bonn extractor above).
            sk   = float(skew(sig_r))
            kurt = float(_kurtosis(sig_r))
            rms  = float(np.sqrt(np.mean(sig_r ** 2)))
            zcr  = float(((sig_r[:-1] * sig_r[1:]) < 0).sum() / len(sig_r))
            p_n  = psd / (psd.sum() + 1e-10)
            sent = float(-np.sum(p_n * np.log2(p_n + 1e-10)) / np.log2(len(p_n) + 1))

            # Append this channel's 13 features to the growing master list.
            all_feats.extend(bp_norm.tolist())
            all_feats.extend([activity, mobility, complexity, sk, kurt, rms, zcr, sent])

        # Return as a flat float32 list (19 channels x 13 features = 247 values minimum).
        return np.array(all_feats, dtype=np.float32).tolist()

    def extract_features_auto(self, sample):
        """Dispatch to the correct extractor based on which model is loaded.

        This is the only extract method called from the main loop — always use this one.
        It checks is_mendeley (set in __init__ based on the model's feature count)
        and calls the appropriate extractor automatically.
        """
        if self.is_mendeley:
            return self.extract_features_mendeley(sample)
        return self.extract_features(sample)

    @staticmethod
    def band_summary(feats):
        """Return [delta, theta, alpha, beta] for display regardless of feature count.

        Different model types store band powers at different positions in the feature
        vector.  This helper normalises that difference so the LCD and terminal always
        display four comparable band-power values regardless of which model is loaded.
          n <= 13   : Bonn model  — band powers are at positions 0-3.
          n == 247  : Mendeley model — average the first 4 bands across all 19 channels.
        """
        n = len(feats)
        if n <= 13:   # Bonn: first 4 are band powers
            return list(feats[:4])
        if n == 247:  # Mendeley 19×13 — first 5 per channel are band powers
            return np.array(feats).reshape(19, 13)[:, :4].mean(axis=0).tolist()
        return list(feats[:4])

# ── SECTION: METRICS ──────────────────────────────────────────────────────────
# =============================================================================
# METRICS
# =============================================================================

# Import sklearn scoring functions used to evaluate the model's performance.
# accuracy_score  — fraction of correct predictions overall.
# precision_score — of all seizure alerts raised, how many were real.
# recall_score    — of all real seizures, how many were caught (sensitivity).
# f1_score        — harmonic mean of precision and recall.
# confusion_matrix — table of true vs predicted labels for all classes.
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

# Path to the JSON file that the dashboard's "Metrics" tab reads.
METRICS_JSON = os.path.join(BASE_DIR, "json", "neurowatch_metrics.json")


def _read_json_safe(path, default):
    """
    Read and parse a JSON file safely, returning 'default' if the file is
    missing, empty, or contains invalid JSON.  Used throughout the BIOS menu
    so that missing files cause a graceful message rather than a crash.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def _print_header(title):
    """
    Print a formatted section header to the terminal.
    Used inside the BIOS menu to visually separate each option's output.
    """
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ── SECTION: BIOS MAINTENANCE MENU FUNCTIONS ──────────────────────────────────
# The following functions are ONLY called when the program is launched with the
# --bios flag.  They provide a text-based maintenance interface for checking
# system health, reviewing patient data, exporting logs, and resetting files.
# None of these functions affect live monitoring.

def _bios_status_health_check():
    """
    BIOS Option 1: Check that all required JSON files exist and are valid.
    Reports the file path, whether it is present, and whether it parses correctly.
    Also counts how many per-patient history files are saved in the json/ folder.
    Use this when the dashboard is blank or showing stale data.
    """
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
    """
    BIOS Option 2: List every patient in PATIENT_REGISTRY.
    Shows ID, name, ward, bed, and whether the patient is the live-monitored one.
    Use this to confirm that patients.py was loaded correctly and all entries
    look right before starting a monitoring session.
    """
    _print_header("[2] Patient registry")
    for i, p in enumerate(PATIENT_REGISTRY, start=1):
        print(
            f"{i}. {p.get('id', '--')} | {p.get('name', '--')} | "
            f"{p.get('ward', '--')} {p.get('bed', '--')} | live={p.get('live', False)}"
        )


def _bios_view_live_history():
    """
    BIOS Option 3: Print the last 15 detection records for the live patient.
    Reads neurowatch_history_<PID>.json and shows timestamp, predicted state,
    and confidence for each stored tick.  Useful for reviewing what the model
    was seeing before a suspected seizure event.
    """
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
    """
    BIOS Option 4: Combine all patient history JSON files into one CSV file.
    Each row in the CSV corresponds to one detection tick for one patient.
    The CSV is saved in the project root with a timestamp in its filename.
    Open it in Excel or load it in a Jupyter notebook for offline analysis.
    """
    _print_header("[4] Export all histories to CSV")
    rows = []
    for path in glob.glob(os.path.join(BASE_DIR, "json", "neurowatch_history_*.json")):
        # Extract the patient ID from the filename (e.g. "neurowatch_history_P001.json" -> "P001").
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
    # Collect every field name that appears across all rows, sorted alphabetically.
    keys = sorted({k for r in rows for k in r.keys()})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Export complete: {out_path} ({len(rows)} rows)")


def _bios_view_model_metrics():
    """
    BIOS Option 5: Display the most recently computed model performance metrics.
    Reads neurowatch_metrics.json which is written by compute_and_write_metrics()
    at startup.  If the file is missing, run the system once to generate it.

    Key metrics to check for a project submission:
      Accuracy      — overall correct predictions across all classes.
      Precision     — how many raised alerts were genuine (low false alarms).
      Recall        — how many real seizures were caught (high = safer).
      F1            — balanced score combining precision and recall.
      Specificity   — how well normal EEG is correctly identified as normal.
      Seizure R     — recall for class 2 (Seizure) specifically — the most critical.
    """
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
    """
    BIOS Option 6: Reset all patient history JSON files to empty lists.
    Use this at the start of a new monitoring session to clear old trend data
    from the dashboard charts.  You must type 'YES' (all caps) to confirm —
    this action cannot be undone and deletes all recorded detection history.
    """
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
    """
    BIOS Option 7: Reset neurowatch_status.json and neurowatch_patients.json
    to clean default values.  Use this when the dashboard is stuck showing
    a stale state (e.g. "Seizure") from a previous crash or interrupted run.
    You must type 'RESET' to confirm.  History files are NOT affected.
    """
    _print_header("[7] Clear status / reset JSON files")
    confirm = input("Type RESET to continue: ").strip()
    if confirm != "RESET":
        print("Cancelled.")
        return

    # Write a blank "waiting" state so the dashboard shows a neutral status.
    with open(STATUS_JSON, "w") as f:
        json.dump({
            "phase": "WAITING",
            "state": "Normal",
            "conf": 0.0,
            "message": "Reset by BIOS maintenance",
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }, f)
    # Write an empty patients dict so no stale patient cards appear on the dashboard.
    with open(PATIENTS_JSON, "w") as f:
        json.dump({}, f)
    print("Status and patient JSON files reset.")


def _bios_show_gpio_pin_map():
    """
    BIOS Option 8: Print the full GPIO wiring table for the NeuroWatch hardware.
    All pin numbers use BCM numbering (the GPIO chip number, not the physical
    header pin position).  Consult this when wiring the circuit or debugging
    a component that is not responding.  To change any pin, update both the
    constant at the top of this file AND the CharLCD constructor in __init__.
    """
    _print_header("[8] Show GPIO pin map")
    print("- LED Green : GPIO 27")
    print("- LED Red   : GPIO 22")
    print("- Buzzer    : GPIO 4")
    print("- LCD RS    : GPIO 26")
    print("- LCD E     : GPIO 19")
    print("- LCD D4..D7: GPIO 13, 6, 5, 20")


def _bios_check_disk_space():
    """
    BIOS Option 9: Show total, used, and free disk space on the drive where
    the project lives.  Important on a Raspberry Pi where a small SD card can
    fill up over time as history JSON files and log data accumulate.
    If free space drops below ~200 MB, use Option 6 to clear old histories
    or manually delete unneeded data files.
    """
    _print_header("[9] Check disk space")
    total, used, free = shutil.disk_usage(BASE_DIR)
    gb = 1024 ** 3
    print(f"Path : {BASE_DIR}")
    print(f"Total: {total / gb:.2f} GB")
    print(f"Used : {used / gb:.2f} GB")
    print(f"Free : {free / gb:.2f} GB")


def run_bios_mode():
    """
    Entry point for the BIOS maintenance menu (launched with --bios flag).
    Displays a numbered menu in a loop and dispatches to the matching helper
    function.  Pressing 0 exits the loop and returns to the terminal.
    No monitoring takes place while BIOS mode is active.
    """
    if not SMS_AVAILABLE:
        print("Warning: sms_notifier.py not found or unavailable - SMS disabled")

    while True:
        # Print the menu header with live system summary on each loop iteration.
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

        # Pause after each option so the user can read the output before the menu redraws.
        input("\nPress Enter to continue...")

# ── FUNCTION: compute_and_write_metrics ───────────────────────────────────────
def compute_and_write_metrics(brain, samples):
    """
    Evaluate the hybrid model against all loaded EEG samples and write the
    results to neurowatch_metrics.json so the dashboard Metrics tab is populated.

    This is called once at startup in a background thread (so it does not delay
    the start of live monitoring) and again after any auto-retrain completes.

    Parameters:
      brain   — the Brain instance containing the loaded and ready models.
      samples — list of (raw_data, true_label) tuples loaded from disk files.

    Metrics computed and written to METRICS_JSON:
      accuracy        — fraction of all samples predicted correctly.
      precision       — weighted average: of raised alerts, how many were real.
      recall          — weighted average: of real events, how many were caught.
      f1              — harmonic mean of precision and recall.
      specificity     — TN / (TN + FP): how well normal EEG is left unalarmed.
      seizure_recall  — recall for class 2 (Seizure) only — the safety-critical metric.
      confusion_matrix — full N x N table for all three classes.
      n_samples        — total number of samples evaluated.
      class_counts     — how many samples belong to each class.

    Skips gracefully (with a warning) if all samples belong to a single class,
    because several metrics are undefined without at least two different classes.
    """
    if not samples:
        return

    print("📊 Computing model metrics...")
    X, y_true, y_pred = [], [], []

    # Run every sample through the pipeline to build true-label and predicted-label arrays.
    for raw_data, label in samples:
        try:
            feats = brain.extract_features_auto(raw_data)
            pred, _ = brain.predict(feats)
            X.append(feats)
            y_true.append(label)
            y_pred.append(pred)
        except Exception:
            continue

    # Metrics like precision and recall require at least two distinct classes.
    if len(set(y_true)) < 2:
        print("⚠️  Not enough class variety to compute metrics — skipping.")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Standard sklearn classification metrics.
    acc      = accuracy_score(y_true, y_pred)
    prec     = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec      = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1v      = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm       = confusion_matrix(y_true, y_pred)

    # Specificity: TN / (TN + FP) using class 0 (Normal) as the "negative" class.
    # cm[0,0] = true negatives (Normal correctly predicted as Normal).
    # cm[0,1:] = false positives (Normal incorrectly predicted as Pre-Sz or Seizure).
    tn = cm[0, 0] if cm.shape[0] > 0 else 0
    fp = cm[0, 1:].sum() if cm.shape[0] > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Seizure-specific recall (class 2) — the most safety-critical metric.
    # Answers: "Of all actual seizures in the data, what fraction did the model catch?"
    seiz_rec = (recall_score(y_true, y_pred, labels=[2], average="macro",
                             zero_division=0) if 2 in y_true else 0.0)

    print(f"  Acc:{acc*100:.1f}%  F1:{f1v*100:.1f}%  "
          f"Prec:{prec*100:.1f}%  SeizRec:{seiz_rec*100:.1f}%")

    # Write all metrics to the JSON file the dashboard reads.
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


# ── SECTION: DATA LOADER & MAIN LOOP ──────────────────────────────────────────
# =============================================================================
# DATA LOADER & MAIN LOOP
# =============================================================================

def main():
    """
    Program entry point — runs when the script is launched normally (no --bios).

    Phase 1 — Startup:
      - Open BIOS menu instead if --bios was passed, then exit.
      - Create HardwareManager (sets up GPIO, LCD, buzzer thread).
      - Load Brain (reads model .pkl files from MODEL_PATH).
      - Load all EEG data files from DATA_PATH into live_samples list.
      - Launch a background thread to compute and write model metrics.

    Phase 2 — Live detection loop (runs until Ctrl+C):
      Each iteration (tick) does the following in order:
        1. Pick the next EEG sample from live_samples (cycles round-robin).
        2. Extract features from the raw signal.
        3. Predict brain state (Normal / Pre-Seizure / Seizure) and confidence.
        4. Track how long the system has been in the current state continuously.
        5. Send an SMS alert if Seizure persists past SEIZURE_LONG_SECONDS.
        6. Send a "back to normal" SMS when state returns to Normal after a seizure.
        7. Choose LCD row 4 content based on state and duration.
        8. Drive LEDs and buzzer via HardwareManager.set_alarm().
        9. Print a one-line status summary to the terminal.
       10. Write neurowatch_status.json (dashboard live state).
       11. Write neurowatch_patients.json (all patients, live + simulated).
       12. Append one record to each patient's history JSON file.
       13. Every 50 ticks: check for new EEG data and retrain if found.
       14. Sleep 1.5 seconds before the next tick.

    Phase 3 — Shutdown:
      - Ctrl+C (KeyboardInterrupt) breaks the loop cleanly.
      - hw.cleanup() always runs to release GPIO pins safely.
    """

    # If --bios was passed on the command line, open maintenance menu and exit.
    if args.bios:
        run_bios_mode()
        return

    # Initialise hardware manager. On Windows this skips GPIO and prints
    # LCD content to the terminal so the logic can still be tested.
    hw = HardwareManager()

    # Show a startup splash message on the LCD / terminal while models load.
    # TO CHANGE THE STARTUP MESSAGE: edit the strings in the list below.
    hw.update_lcd(["NeuroWatch v1.3", "Universal Path Fix", "", "Wait..."])
    
    # Load the ML models. If any model file is missing, show an error on the
    # LCD and exit cleanly rather than crashing with an unhandled exception.
    try:
        brain = Brain()
    except Exception as e:
        print(f"❌ Load Error: {e}")
        hw.update_lcd(["LOAD ERROR", "Check /models", str(e)[:20], ""])
        return

    # live_samples holds all EEG data as a list of (raw_data_array, true_label) tuples.
    # It is populated below in either Mendeley mode or Bonn mode depending on the model.
    live_samples = []

    # MENDELEY_LABEL_MAP: converts Mendeley integer labels to our 3-class system.
    # Mendeley uses: 0=Normal, 1=Pre-ictal, 2=Ictal(Seizure), 3=Artifact.
    # Artifact (3) is treated as Pre-Seizure (1) since it often appears before seizures.
    MENDELEY_LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 1}

    if brain.is_mendeley:
        # ── Mendeley mode: load pre-processed .npy windows ──────────────────
        # npy_dir must contain x_train.npy, y_train.npy, x_test.npy, y_test.npy.
        # TO CHANGE DATA FOLDER: pass --data /your/npy/folder when launching,
        # or update the default path string below.
        npy_dir = (args.data if args.data and os.path.isdir(args.data)
                   else os.path.join(BASE_DIR, "data", "Mendelay dataset", "Npy_files_preictal"))
        print(f"📂 Mendeley mode — loading .npy files from: {npy_dir}")
        parts_x, parts_y = [], []

        # Load both the training and test splits into the monitoring pool.
        # The system uses all available samples for the live cycling loop.
        for xname, yname in [("x_train.npy", "y_train.npy"), ("x_test.npy", "y_test.npy")]:
            xp = os.path.join(npy_dir, xname)
            yp = os.path.join(npy_dir, yname)
            if os.path.exists(xp) and os.path.exists(yp):
                parts_x.append(np.load(xp))
                parts_y.append(np.load(yp))
                print(f"  ✅ {xname}")
        if parts_x:
            # Concatenate train and test arrays into one large dataset.
            X_all = np.concatenate(parts_x, axis=0)
            y_all = np.concatenate(parts_y, axis=0)
            for sample, lbl in zip(X_all, y_all):
                live_samples.append((sample, MENDELEY_LABEL_MAP[int(lbl)]))
            print(f"✅ Loaded {len(live_samples)} Mendeley samples.")
        else:
            print(f"❌ No .npy files found in {npy_dir}")
    else:
        # ── Bonn mode: scan for .edf / .txt files ───────────────────────────
        # Walks the DATA_PATH tree. Subfolder names matching CLASS_MAP keys
        # (O, N, S, F, Z) determine the ground-truth label for files inside them.
        print(f"📂 Scanning for EEG (.edf/.txt) in: {DATA_PATH}")
        search_dirs = [DATA_PATH]
        for root, dirs, _ in os.walk(DATA_PATH):
            for d in dirs:
                if d in CLASS_MAP:
                    search_dirs.append(os.path.join(root, d))

        # dict.fromkeys preserves order while removing any duplicate paths.
        search_dirs = list(dict.fromkeys(search_dirs))

        for d in search_dirs:
            if not os.path.exists(d): continue
            files = glob.glob(os.path.join(d, "*.txt")) + glob.glob(os.path.join(d, "*.edf"))
            for f in files:
                try:
                    # Infer class label from the name of the containing folder.
                    folder_name = os.path.basename(os.path.dirname(f))
                    label = CLASS_MAP.get(folder_name, 0)  # default to Normal if unrecognised

                    if f.endswith('.edf'):
                        # Load EDF with MNE, resample to TARGET_FS if needed, take 1 second.
                        raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
                        if raw.info['sfreq'] != TARGET_FS:
                            raw.resample(TARGET_FS)
                        data = raw.get_data()[0]  # use the first channel only
                        if len(data) >= TARGET_FS:
                            live_samples.append((data[:TARGET_FS], label))
                    else:
                        # Bonn .txt files contain one sample per line at 173.61 Hz.
                        # Resample to TARGET_FS (128 Hz) before storing.
                        sig = np.loadtxt(f)
                        n_new = int(len(sig) * TARGET_FS / 173.61)
                        sig = resample(sig, n_new)
                        live_samples.append((sig, label))
                except Exception as e:
                    print(f"⚠️ Skip {os.path.basename(f)}: {e}")

    # If no data was found at all, report the error and exit gracefully.
    if not live_samples:
        print("❌ No valid files found in path.")
        hw.update_lcd(["ERROR", "No Data Found", "Check Path", ""])
        return

    print(f"✅ Loaded {len(live_samples)} files. Monitoring active.")

    # Compute metrics in the background — does not block the live loop.
    # The background thread calls compute_and_write_metrics() which evaluates
    # all loaded samples and writes neurowatch_metrics.json for the dashboard.
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