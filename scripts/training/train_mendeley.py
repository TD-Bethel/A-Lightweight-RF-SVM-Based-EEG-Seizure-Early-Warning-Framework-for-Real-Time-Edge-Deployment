## =============================================================================
# train_mendeley.py  --  NeuroWatch Model Training (Mendeley Dataset)
#
# DATASET:
#   Mendeley "Epileptic EEG Dataset" (Wassim Nasreddine, AUB)
#   Input files: x_train.npy, y_train.npy, x_test.npy, y_test.npy
#   Located in:  data/Mendelay dataset/Npy_files/
#
# SPLIT STRATEGY (stratified, from combined 7790 samples):
#   Train      : 80%  -> 6232 samples  -- model learns from this
#   Validation : 10%  -> 779  samples  -- used during training to detect overfitting
#   Test       : 10%  -> 779  samples  -- final untouched evaluation
#
#   Stratified split: every class keeps identical proportions in all 3 sets.
#   This is critical for the minority Video-detected class (only 111 samples).
#
# LABEL MAPPING (4-class Mendeley -> 3-class NeuroWatch):
#   0 = Normal              ->  0 = Normal
#   1 = CPS                 ->  1 = Pre-Seizure  (partial seizure, subtle EEG)
#   2 = Electrographic      ->  2 = Seizure      (clearest EEG signal)
#   3 = Video-detected      ->  1 = Pre-Seizure  (behavioural only, no EEG change)
#
# FEATURES (per sample):
#   All 19 channels x 13 per-channel features + 171 inter-channel correlations = 418 features
#   Option A normalisation: band powers divided by their sum per channel
#   -> relative proportions, scale-invariant across any EEG equipment
#
# MODEL:
#   Hybrid: 40% SVM (RBF) + 60% Random Forest -- same blend as main_pi_bios_v15.py
#   StandardScaler fitted on training set ONLY -- no leakage into val or test
#
# OUTPUT (saved to models/MODELS_MENDELEY/):
#   scaler.pkl
#   svm_model.pkl
#   rf_model.pkl
#   training_report.json   # full metrics for all 3 splits
#
# USAGE:
#   python train_mendeley.py
#   python train_mendeley.py --data "C:\...\Npy_files" --out "C:\...\MODELS_V1"
#   python train_mendeley.py --no-plots
# =============================================================================

# ── Standard library imports ───────────────────────────────────────────────────
# os/sys: file paths and Python path manipulation
# time  : measuring how long each step takes
# json  : saving the training report as a readable file
# argparse: reading command-line arguments (--data, --out, etc.)
# warnings: suppress minor warnings that clutter the output
import os, sys, time, json, argparse, warnings
import numpy as np    # numerical arrays and maths
import joblib         # saving/loading model objects as .pkl files

# ── Signal processing ──────────────────────────────────────────────────────────
from scipy.signal import butter, filtfilt, welch, resample
# butter   : design a Butterworth band-pass filter (smooth roll-off, no ripple)
# filtfilt : apply filter forwards + backwards so there is zero phase distortion
# welch    : estimate the power spectral density (how much power at each frequency)
# resample : change the sampling rate of a signal

from scipy.stats import skew, kurtosis as scipy_kurtosis
# skew     : measures left/right asymmetry of the amplitude distribution
# kurtosis : measures how "peaked" the distribution is (high in spiky seizure signals)

# ── Machine learning imports ───────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler            # zero-mean, unit-variance scaling
from sklearn.svm import SVC                                 # Support Vector Classifier
from sklearn.ensemble import RandomForestClassifier         # Random Forest
from sklearn.model_selection import train_test_split        # split data into train/val/test
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ── Optional: MLflow experiment tracking ──────────────────────────────────────
# MLflow logs every run's hyperparameters and metrics so you can compare experiments.
# View results: mlflow ui  (then open localhost:5000 in a browser).
# Install: pip install mlflow
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("WARN  mlflow not found -- pip install mlflow  (tracking disabled)")

# ── Optional: SMOTE class balancing ───────────────────────────────────────────
# SMOTE (Synthetic Minority Over-sampling Technique) generates artificial samples
# for minority classes (e.g. Pre-Seizure) to balance the training set.
# Install: pip install imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("WARN  imbalanced-learn not found -- install with: pip install imbalanced-learn")

# Suppress non-critical warnings from scipy and sklearn during training.
warnings.filterwarnings("ignore")

# Add the project root directory to Python's module search path.
# This allows 'from neurowatch_selector import RFImportanceSelector' to work
# regardless of which directory you run the script from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from neurowatch_selector import RFImportanceSelector


# =============================================================================
# SECTION: PATHS  --  update if your folders are in different locations
# =============================================================================
# BASE_DIR resolves to the project root folder automatically.
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DEFAULT_DATA: path to the folder containing x_train.npy, y_train.npy, etc.
# Change this if your data is stored elsewhere.
DEFAULT_DATA = os.path.join(BASE_DIR, "data", "Mendelay dataset", "Npy_files")

# DEFAULT_OUT: where trained model .pkl files and reports will be written.
# A new folder is created if it does not exist.
DEFAULT_OUT  = os.path.join(BASE_DIR, "models", "MODELS_MENDELEY")


# =============================================================================
# SECTION: CONSTANTS
# =============================================================================
# These values define the signal processing pipeline and feature dimensions.
# Changing any of them requires re-training the model from scratch.

# TARGET_FS: all EEG signals are resampled to this rate (Hz) before feature extraction.
#   Increase to 256 for more high-frequency detail (slower); decrease to 64 for speed.
TARGET_FS    = 128      # resample everything to this

# FS_MENDELEY: the native recording rate of the Mendeley dataset.
#   Do NOT change — this is fixed by the data files.
FS_MENDELEY  = 500      # native Mendeley rate

# WIN_SIZE: length of the signal window (samples at TARGET_FS) used in Welch's method.
#   Must be at least 2 × NPERSEG. Larger values give smoother PSD estimates.
WIN_SIZE     = 4097     # tiling window size for Welch consistency

# N_ORDER: Butterworth filter order — controls how steeply the filter rolls off.
#   Order 4 provides a good balance between sharpness and numerical stability.
N_ORDER      = 4

# F_LOW, F_HIGH: bandpass filter cutoff frequencies (Hz).
#   0.5 Hz removes slow DC drift; 45 Hz removes high-frequency noise and mains hum.
#   Do not raise F_HIGH above 63 Hz for 128 Hz data (Nyquist limit is 64 Hz).
F_LOW, F_HIGH = 0.5, 45.0

# NPERSEG: number of samples per Welch segment.
#   Larger = finer frequency resolution but fewer averages (noisier estimate).
#   128 / TARGET_FS = 1 second — gives ~1 Hz frequency resolution.
NPERSEG      = 128

# N_CHANNELS: number of EEG electrodes. The 10-20 system uses 19 channels.
N_CHANNELS   = 19

# N_BANDS: number of frequency bands extracted per channel.
#   delta(1-4Hz), theta(4-8Hz), alpha(8-13Hz), beta(13-30Hz), gamma(30-45Hz).
N_BANDS      = 5        # delta, theta, alpha, beta, gamma

# N_STATS: number of statistical time-domain features extracted per channel.
N_STATS      = 8        # activity, mobility, complexity, skewness, kurtosis, rms, zcr, spectral_entropy

# N_CORR: inter-channel correlation features (set to 0 — removed to reduce overfitting).
#   Correlation between all 19×18/2 = 171 channel pairs was originally included
#   but caused overfitting without sufficient generalisation benefit on this dataset size.
N_CORR       = 0  # correlation features removed to reduce overfitting

# N_FEATURES: total number of features per sample before selection.
#   19 channels × (5 bands + 8 stats) = 247 features.
N_FEATURES   = N_CHANNELS * (N_BANDS + N_STATS)  # 19 x 13 = 247

# N_SELECT: how many of the 247 features to keep after RF importance ranking.
#   Higher (e.g. 100) = more information but risk of overfitting on small datasets.
#   Lower  (e.g. 40)  = faster inference and less overfit, but may miss weak signals.
#   75 was chosen empirically to balance accuracy with generalisation on this dataset.
N_SELECT     = 75  # top-K features kept after SelectKBest(f_classif)

# TRAIN/VAL/TEST ratios — must sum to 1.0.
# To change the split (e.g. 70/15/15), update all three values consistently.
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
TEST_RATIO   = 0.10

# RANDOM_SEED: fixes all randomness for reproducible results.
#   Changing this gives a different (but equally valid) train/val/test split.
RANDOM_SEED  = 42

# ── Label mapping: Mendeley 4-class -> NeuroWatch 3-class ─────────────────────
# The Mendeley dataset has 4 classes; NeuroWatch uses 3.
# Class 3 (Video-detected) maps to Pre-Seizure (1) because the EEG signal is subtle,
# similar to CPS, rather than a full electrographic seizure.
LABEL_MAP   = {0: 0, 1: 1, 2: 2, 3: 1}
LABEL_NAMES = {
    0: "Normal",
    1: "CPS (Complex Partial)",
    2: "Electrographic",
    3: "Video-detected",
}
CLASSES = ["Normal", "Pre-Seizure", "Seizure"]
COLORS  = ["#00CC66", "#FFD633", "#FF3333"]


# =============================================================================
# SECTION: ARGUMENT PARSING
# =============================================================================
# Command-line arguments let you customise runs without editing this file.
# Example: python train_mendeley.py --data "C:\data\eeg" --out "C:\models\v2" --no-plots
ap = argparse.ArgumentParser(description="NeuroWatch -- Mendeley Model Trainer")
ap.add_argument("--data",       default=DEFAULT_DATA,
                help="Path to folder containing x_train.npy etc.")
ap.add_argument("--out",        default=DEFAULT_OUT,
                help="Where to save trained model .pkl files")
ap.add_argument("--no-plots",   action="store_true",
                help="Skip all matplotlib plots")

# --svm-weight: pin the SVM blend weight to a fixed value.
# Example: --svm-weight 0.40 uses SVM 40% + RF 60% without running the auto-search.
ap.add_argument("--svm-weight", type=float, default=None,
                help="Fix SVM blend weight (0.0–1.0). Skips auto-tuning. "
                     "e.g. --svm-weight 0.40 gives SVM 40%% + RF 60%%")
args = ap.parse_args()

# ── Plotting setup ────────────────────────────────────────────────────────────
# Plots are enabled by default; --no-plots disables them (useful for headless servers).
PLOT = not args.no_plots
if PLOT:
    try:
        # TkAgg opens an interactive window on-screen.
        # On a server without a display, use matplotlib.use("Agg") instead.
        import matplotlib; matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import Patch
        import seaborn as sns
        # Apply a dark navy colour theme to all figures.
        plt.rcParams.update({
            "figure.facecolor": "#1a1a2e", "axes.facecolor":  "#16213e",
            "axes.edgecolor":   "#444",    "axes.labelcolor": "#ccc",
            "xtick.color":      "#aaa",    "ytick.color":     "#aaa",
            "text.color":       "#eee",    "grid.color":      "#2a2a4a",
            "grid.linestyle":   "--",      "grid.alpha":      0.5,
        })
        PLOT_OK = True
    except Exception as e:
        print(f"  WARN  Plots unavailable: {e}")
        PLOT_OK = False
else:
    PLOT_OK = False


# =============================================================================
# SECTION: STEP 1 — SIGNAL PROCESSING & FEATURE EXTRACTION
# =============================================================================
# These helper functions perform the core signal processing on each EEG channel.
# They are called inside extract_features() for every channel of every sample.

# ── Bandpass filter ───────────────────────────────────────────────────────────
# Removes frequencies below F_LOW (DC drift) and above F_HIGH (noise, mains hum).
# filtfilt applies the filter forwards and backwards to eliminate phase shift.
def _bandpass(sig, fs):
    nyq  = 0.5 * fs          # Nyquist frequency = half the sampling rate
    b, a = butter(N_ORDER, [F_LOW / nyq, F_HIGH / nyq], btype="band")
    return filtfilt(b, a, sig)

# ── Frequency band powers ─────────────────────────────────────────────────────
# Returns the average power in each of the 5 clinically relevant EEG bands.
# Uses Welch's method: divides the signal into overlapping windows, computes FFT
# on each, then averages the results — reduces noise compared to a single FFT.
def _band_power(sig, fs=TARGET_FS):
    """Extract [delta, theta, alpha, beta, gamma] power from one channel."""
    filt   = _bandpass(sig, fs)
    f, psd = welch(filt, fs=fs, nperseg=NPERSEG)
    return np.array([
        np.mean(psd[(f >= 1)  & (f <= 4)]),    # Delta (1–4 Hz):  slow waves, deep sleep
        np.mean(psd[(f >= 4)  & (f <= 8)]),    # Theta (4–8 Hz):  drowsiness, temporal lobe
        np.mean(psd[(f >= 8)  & (f <= 13)]),   # Alpha (8–13 Hz): relaxed wakefulness
        np.mean(psd[(f >= 13) & (f <= 30)]),   # Beta (13–30 Hz): alertness, motor activity
        np.mean(psd[(f >= 30) & (f <= 45)]),   # Gamma (30–45 Hz):high-frequency bursts
    ])

# ── Option A normalisation ────────────────────────────────────────────────────
# Divides each band's power by the total power so all five values sum to 1.
# This makes features scale-invariant: the same patient recorded on different
# EEG equipment (with different amplifier gains) gives identical features.
def _normalise(feat_vec):
    total = feat_vec.sum()
    return feat_vec / total if total > 0 else feat_vec

# ── Hjorth parameters ─────────────────────────────────────────────────────────
# Three features that describe the complexity of an EEG signal using its derivatives.
#   Activity   = variance of the signal — measures overall power/energy.
#   Mobility   = normalised slope variance — related to the dominant frequency.
#   Complexity = ratio of mobility between signal and its derivative — measures irregularity.
# High complexity and low mobility often indicate seizure activity.
def _hjorth(sig):
    """Hjorth activity, mobility, complexity."""
    d1      = np.diff(sig)     # first derivative (change per sample)
    d2      = np.diff(d1)      # second derivative (rate of change of change)
    var_x   = np.var(sig)  + 1e-10   # small epsilon prevents division by zero
    var_d1  = np.var(d1)   + 1e-10
    var_d2  = np.var(d2)   + 1e-10
    activity   = float(var_x)
    mobility   = float(np.sqrt(var_d1 / var_x))
    mob_d1     = float(np.sqrt(var_d2 / var_d1))
    complexity = float(mob_d1 / (mobility + 1e-10))
    return activity, mobility, complexity

# ── Spectral entropy ──────────────────────────────────────────────────────────
# Measures how spread out signal energy is across the frequency spectrum.
# High entropy = energy evenly spread (noisy/complex signal, typical of seizure).
# Low entropy  = energy concentrated in a narrow band (e.g. clean alpha rhythm).
# Normalised by log2(len(psd)) so the value is always between 0 and 1.
def _spectral_entropy(psd):
    """Normalised spectral entropy."""
    p = psd / (psd.sum() + 1e-10)   # convert PSD to a probability distribution
    return float(-np.sum(p * np.log2(p + 1e-10)) / np.log2(len(p) + 1))

# ── Zero crossing rate ────────────────────────────────────────────────────────
# Counts how often the signal crosses zero per sample.
# High ZCR indicates rapid oscillations (high-frequency content), common during seizures.
def _zcr(sig):
    return float(((sig[:-1] * sig[1:]) < 0).sum() / len(sig))


# ── Main feature extraction function ─────────────────────────────────────────
# Converts one EEG sample (shape: 19 channels × N time-points) into a flat
# 247-element feature vector that the ML models can process.
#
# Feature layout per channel (13 features × 19 channels = 247 total):
#   [0–4]  : 5 normalised band powers (delta, theta, alpha, beta, gamma)
#   [5–7]  : 3 Hjorth parameters (activity, mobility, complexity)
#   [8]    : skewness of amplitude distribution
#   [9]    : kurtosis (spike sharpness)
#   [10]   : RMS (root mean square amplitude)
#   [11]   : ZCR (zero crossing rate)
#   [12]   : spectral entropy
def extract_features(sample):
    """
    Extract 247-feature vector from one Mendeley sample.
    sample: (19, 500) -- 19 channels x 1 second at 500 Hz

    Per channel (13 features x 19 channels = 247):
      Band powers (5): delta, theta, alpha, beta, gamma  -- Option A normalised
      Statistical (8): activity, mobility, complexity,
                       skewness, kurtosis, rms, zcr, spectral_entropy

    Correlation features removed (were 171 pairs) -- caused overfitting
    on this dataset size without enough generalization benefit.

    Returns (247,) float32 array.
    """
    per_ch_feats = []
    resampled    = []

    for ch in range(N_CHANNELS):
        sig   = sample[ch]

        # ── Resample from 500 Hz -> 128 Hz ───────────────────────────────────
        # Reduces data size and standardises across recording equipment.
        n_new = int(len(sig) * TARGET_FS / FS_MENDELEY)
        sig_r = resample(sig, n_new)

        # ── Tile to WIN_SIZE for consistent Welch FFT length ─────────────────
        # Welch's method needs a minimum number of samples.
        # If the resampled signal is shorter than WIN_SIZE, repeat it and trim.
        reps  = (WIN_SIZE // len(sig_r)) + 1
        sig_t = np.tile(sig_r, reps)[:WIN_SIZE]
        resampled.append(sig_r)

        # ── Band powers (5 features) — Option A normalised ────────────────────
        # Compute raw band powers then normalise so they sum to 1.
        bp      = _band_power(sig_t)
        bp_norm = _normalise(bp)

        # ── Hjorth parameters (3 features) ───────────────────────────────────
        # Computed on the resampled signal (not tiled) for accurate time-domain stats.
        act, mob, comp = _hjorth(sig_r)

        # ── Additional statistical features (5 features) ──────────────────────
        # Skewness: asymmetry of amplitude distribution (spikes make it positive).
        sk   = float(skew(sig_r))
        # Kurtosis: peakedness — seizure spikes create very high kurtosis.
        kurt = float(scipy_kurtosis(sig_r))
        # RMS: root mean square — measures average signal power (amplitude).
        rms  = float(np.sqrt(np.mean(sig_r ** 2)))
        # ZCR: fraction of adjacent sample pairs that cross zero.
        zcr  = _zcr(sig_r)

        # ── Spectral entropy (1 feature) ──────────────────────────────────────
        # Compute PSD of the filtered+tiled signal, then measure its entropy.
        filt    = _bandpass(sig_t, TARGET_FS)
        _, psd  = welch(filt, fs=TARGET_FS, nperseg=NPERSEG)
        sent    = _spectral_entropy(psd)

        # Append all 13 features for this channel to the running list.
        per_ch_feats.extend(bp_norm.tolist())
        per_ch_feats.extend([act, mob, comp, sk, kurt, rms, zcr, sent])

    # Return one flat (247,) float32 array for this sample.
    return np.array(per_ch_feats, dtype=np.float32)


# =============================================================================
# SECTION: STEP 2 — LOAD NPY FILES & BUILD FEATURE MATRIX
# =============================================================================
# load_and_extract:
#   1. Loads x_train.npy / y_train.npy / x_test.npy / y_test.npy.
#   2. Concatenates them into one dataset.
#   3. Applies a stratified 80/10/10 split.
#   4. Calls extract_features() on every sample in each split.
#   Returns six arrays: X_train, y_train, X_val, y_val, X_test, y_test.
def load_and_extract(data_dir):
    print(f"\n{'-'*65}")
    print(f"  STEP 1: Loading npy files from:")
    print(f"  {data_dir}")
    print(f"{'-'*65}")

    # ── Helper: load one pair of npy files ───────────────────────────────────
    def try_load(xname, yname):
        xp = os.path.join(data_dir, xname)
        yp = os.path.join(data_dir, yname)
        if os.path.exists(xp) and os.path.exists(yp):
            X = np.load(xp); Y = np.load(yp)
            print(f"  OK {xname:<20} {str(X.shape):<20}  "
                  f"{yname:<20} {str(Y.shape)}")
            return X, Y
        return None, None

    Xtr, Ytr = try_load("x_train.npy", "y_train.npy")
    Xte, Yte = try_load("x_test.npy",  "y_test.npy")

    if Xtr is None and Xte is None:
        print(f"  FAIL No npy files found in {data_dir}")
        print(f"     Update DEFAULT_DATA at the top of this file.")
        sys.exit(1)

    # Combine whichever files were found into one large dataset.
    parts_x = [x for x in [Xtr, Xte] if x is not None]
    parts_y = [y for y in [Ytr, Yte] if y is not None]
    X_all   = np.concatenate(parts_x, axis=0)
    y_all   = np.concatenate(parts_y, axis=0)

    print(f"\n  Combined: {len(X_all)} samples  |  shape per sample: {X_all[0].shape}")
    # Show the original 4-class distribution before label mapping.
    print(f"\n  Original 4-class distribution:")
    for lbl in sorted(np.unique(y_all)):
        cnt = int(np.sum(y_all == lbl))
        bar = "#" * int(cnt / len(y_all) * 40)
        print(f"    {lbl} = {LABEL_NAMES[lbl]:<35} {cnt:5d}  {bar}")

    # ── Stratified 80/10/10 split ─────────────────────────────────────────────
    # "Stratified" ensures every class appears in the same proportion in each split.
    # This is especially important for small minority classes.
    #
    # HOW TO CHANGE THE SPLIT RATIO:
    #   To use 70/15/15: change test_size=(0.15+0.15) = 0.30 below, keep test_size=0.5 in the second split.
    #   To use 90/5/5:   change test_size=0.10 below, keep test_size=0.5.
    #   The second split always halves the "held-out" set into equal val and test portions.
    print(f"\n  Applying stratified 80 / 10 / 10 split  (seed={RANDOM_SEED})...")

    # First split: put 80% in training, 20% in a temporary pool (val + test).
    X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
        X_all, y_all,
        test_size=(VAL_RATIO + TEST_RATIO),   # 0.20 = 10% val + 10% test
        random_state=RANDOM_SEED,
        stratify=y_all,
    )
    # Second split: divide the 20% pool equally into val and test.
    X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
        X_temp, y_temp,
        test_size=0.5,                    # half of the 20% = 10% each
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    print(f"  Train : {len(X_train_raw):5d} samples  ({len(X_train_raw)/len(X_all)*100:.0f}%)")
    print(f"  Val   : {len(X_val_raw):5d} samples  ({len(X_val_raw)/len(X_all)*100:.0f}%)")
    print(f"  Test  : {len(X_test_raw):5d} samples  ({len(X_test_raw)/len(X_all)*100:.0f}%)")

    # ── Verify stratification worked correctly ────────────────────────────────
    # Each class percentage should be nearly identical across all three splits.
    # If you see "WARN" here, the class is extremely rare and stratification is imperfect.
    print(f"\n  Stratification check -- class proportions should match across splits:")
    print(f"  {'Label':<25} {'All':>7} {'Train':>7} {'Val':>7} {'Test':>7}")
    for lbl in sorted(np.unique(y_all)):
        ap  = np.mean(y_all       == lbl) * 100
        tp  = np.mean(y_train_raw == lbl) * 100
        vp  = np.mean(y_val_raw   == lbl) * 100
        ep  = np.mean(y_test_raw  == lbl) * 100
        ok  = "OK" if max(abs(tp-vp), abs(tp-ep)) < 1.5 else "WARN"
        print(f"  {ok} {LABEL_NAMES[lbl]:<23} "
              f"{ap:>6.1f}%  {tp:>6.1f}%  {vp:>6.1f}%  {ep:>6.1f}%")

    # ── Feature extraction over all three splits ──────────────────────────────
    print(f"\n  STEP 2: Extracting {N_FEATURES} features per sample "
          f"({N_CHANNELS} channels x {N_BANDS} bands)...")
    print(f"  Option A normalisation: ON  (relative band power proportions)")
    print(f"  This may take a few minutes...\n")

    # Inner function to extract features from one split with a progress bar.
    def extract_split(X_raw, y_raw, name):
        feats, labels = [], []
        n = len(X_raw)
        t0 = time.time()
        for i, (sample, lbl) in enumerate(zip(X_raw, y_raw)):
            # Print a progress bar every 200 samples.
            if i % 200 == 0:
                pct = i / n * 100
                bar = "#" * int(pct / 5) + "#" * (20 - int(pct / 5))
                print(f"  [{bar}] {pct:5.1f}%  {name}  ({i}/{n})", end="\r")
            feats.append(extract_features(sample))
            # Convert the 4-class Mendeley label to the 3-class NeuroWatch label.
            labels.append(LABEL_MAP[int(lbl)])
        elapsed = time.time() - t0
        print(f"  [{' #'*20}] 100.0%  {name} done  ({elapsed:.0f}s)       ")

        F = np.array(feats, dtype=np.float32)
        L = np.array(labels, dtype=np.int32)
        uniq, cnts = np.unique(L, return_counts=True)
        dist = {CLASSES[u]: int(c) for u, c in zip(uniq, cnts)}
        print(f"  Shape: {F.shape}  |  3-class dist: {dist}")
        return F, L

    print()
    X_train, y_train = extract_split(X_train_raw, y_train_raw, "Train")
    print()
    X_val,   y_val   = extract_split(X_val_raw,   y_val_raw,   "Val  ")
    print()
    X_test,  y_test  = extract_split(X_test_raw,  y_test_raw,  "Test ")

    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================================================
# SECTION: STEP 3 — TRAIN
# =============================================================================
# train() takes the feature matrices for train and val splits and:
#   1. Fits a StandardScaler on the training data only.
#   2. Runs a bootstrap RF to rank features and select the top N_SELECT.
#   3. Trains an SVM on scaled+selected features.
#   4. Trains a Random Forest on unscaled+selected features.
#   5. Finds the best SVM/RF blend ratio (or uses a manually fixed ratio).
#   6. Tunes a Normal threshold to reduce false Pre-Seizure predictions.
def train(X_train, y_train, X_val, y_val, forced_svm_w=None):
    print(f"\n{'-'*65}")
    print(f"  STEP 3: Training  (StandardScaler + RF-importance selection + SVM + RF)")
    print(f"{'-'*65}")

    # ── StandardScaler: fitted on train ONLY ──────────────────────────────────
    # Transforms each feature to zero mean, unit variance.
    # CRITICAL: use fit_transform on train, but only transform on val/test.
    # Fitting on val/test would "leak" information about the test distribution
    # into training, inflating metrics and making the model appear better than it is.
    print(f"\n  Fitting StandardScaler on {len(X_train)} training samples...")
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)   # fit here, never again

    # ── Bootstrap RF for feature selection ────────────────────────────────────
    # A Random Forest with 300 trees is trained on raw (unscaled) features.
    # Each feature receives an importance score; the top N_SELECT are kept.
    # Fewer features = faster training + less overfitting on small datasets.
    # RF works on unscaled features because tree splits use rank, not magnitude.
    #
    # To keep more features: increase N_SELECT at the top of this file.
    # To use a simpler selector: replace RFImportanceSelector with SelectKBest(f_classif, k=N_SELECT).
    print(f"\n  Bootstrap RF (300 trees) for RF-importance feature selection...")
    selector = RFImportanceSelector(k=N_SELECT)
    selector.fit(X_train, y_train,          # unscaled — RF is scale-invariant
                 n_estimators=300, max_depth=10, random_state=RANDOM_SEED)

    # Print the top 10 selected features for inspection.
    feat_names = ["delta","theta","alpha","beta","gamma",
                  "activity","mobility","complexity",
                  "skewness","kurtosis","rms","zcr","spectral_entropy"]
    top10_boot = np.argsort(selector.importances_)[::-1][:10]
    print(f"  Top 10 features selected (out of {N_FEATURES}):")
    for rank, idx in enumerate(top10_boot, 1):
        ch   = idx // (N_BANDS + N_STATS)    # which channel (0–18)
        feat = feat_names[idx % (N_BANDS + N_STATS)]   # which feature within the channel
        imp  = selector.importances_[idx]
        bar  = "#" * int(imp * 300)
        print(f"    {rank:2}. ch{ch:02d}-{feat:<18}  {imp:.4f}  {bar}")

    # ── Apply feature selection to all splits ─────────────────────────────────
    # SVM receives scaled then selected features (SVM is sensitive to scale).
    # RF receives only selected features without scaling (trees are scale-invariant).
    X_tr_sel  = selector.transform(X_tr_sc)          # scaled+selected  (SVM)
    X_tr_rf   = selector.transform(X_train)           # unscaled+selected (RF)
    X_val_sc  = scaler.transform(X_val)               # scale val — DO NOT fit again
    X_val_sel = selector.transform(X_val_sc)
    X_val_rf  = selector.transform(X_val)
    print(f"\n  Feature count: {N_FEATURES} → {N_SELECT} selected")

    # ── SVM training ──────────────────────────────────────────────────────────
    # SVM with RBF (Radial Basis Function) kernel finds a curved decision boundary.
    #
    # Key hyperparameters:
    #   C (regularisation strength):
    #     Low C (e.g. 0.1) = large margin, tolerates misclassification, may underfit.
    #     High C (e.g. 10) = small margin, fits training closely, may overfit.
    #     C=2 is a balanced default. Tune with --svm-c on the command line.
    #
    #   gamma="scale": RBF width = 1 / (n_features × var(X)).
    #     Lower gamma = wider, smoother boundary.
    #     Higher gamma = tighter, more complex boundary (overfit risk on small data).
    #
    #   class_weight: increases the penalty for misclassifying certain classes.
    #     {0:1.0, 1:1.8, 2:1.0} means errors on Pre-Seizure cost 80% more.
    #     Increase the Pre-Seizure weight if too many Pre-Seizure samples are missed.
    #     Increase the Seizure weight if seizures are missed (most safety-critical).
    #
    #   probability=True: enables predict_proba() needed for ensemble blending.
    #     Makes training ~5× slower (uses Platt scaling internally); cannot be disabled.
    print(f"\n  Training SVM (RBF, C=2, class_weight={{0:1.0, 1:1.8, 2:1.0}})...")
    t0  = time.time()
    svm = SVC(
        kernel="rbf", C=2, gamma="scale",
        probability=True,
        class_weight={0: 1.0, 1: 1.8, 2: 1.0},
        random_state=RANDOM_SEED,
    )
    svm.fit(X_tr_sel, y_train)
    svm_time = time.time() - t0
    svm_val  = accuracy_score(y_val, svm.predict(X_val_sel))
    print(f"  SVM done in {svm_time:.1f}s  |  "
          f"Support vectors: {sum(svm.n_support_)}  |  "
          f"Val accuracy: {svm_val*100:.1f}%")

    # ── Random Forest training ─────────────────────────────────────────────────
    # A Random Forest averages the predictions of many independent decision trees.
    # It is less sensitive to hyperparameters than SVM and naturally handles
    # non-linear boundaries without feature scaling.
    #
    # Key hyperparameters:
    #   n_estimators=500: number of trees.
    #     More trees = stabler predictions but slower training.
    #     Reduce to 200 if training is too slow; 500 is a safe default.
    #
    #   max_depth=5: maximum number of splits per tree.
    #     Shallow trees (3–5) generalise better; deep trees (10+) memorise training data.
    #     If the overfitting gap is large, reduce to 3. If accuracy is low, try 7.
    #
    #   min_samples_leaf=10: minimum samples required at a leaf node.
    #     Higher values enforce smoother decision boundaries and reduce overfitting.
    #     Lower values allow the tree to fit rare patterns — more overfit risk.
    #
    #   max_features="sqrt": each split considers only sqrt(n_features) candidates.
    #     This is the standard RF setting that ensures trees are uncorrelated (diverse).
    #
    #   class_weight="balanced": weights each class by 1 / (n_classes × n_samples_in_class).
    #     Automatically handles class imbalance without manual weight specification.
    #
    #   n_jobs=-1: parallelise across all CPU cores (much faster on multi-core machines).
    print(f"\n  Training Random Forest (500 trees, max_depth=5, balanced)...")
    t0 = time.time()
    rf  = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_tr_rf, y_train)
    rf_time = time.time() - t0
    rf_val  = accuracy_score(y_val, rf.predict(X_val_rf))
    print(f"  RF done in {rf_time:.1f}s  |  Val accuracy: {rf_val*100:.1f}%")

    # ── Ensemble weight search ────────────────────────────────────────────────
    # The final prediction blends SVM and RF probability outputs:
    #   combined_prob = SVM_W × p_svm + RF_W × p_rf
    #
    # Why blend? SVM and RF make different types of errors.
    # The ensemble reduces variance and typically outperforms either model alone.
    #
    # AUTO-SEARCH (default): tries SVM weights from 0.05 to 0.95 in steps of 0.05,
    #   picks the weight that gives the highest validation accuracy.
    #
    # MANUAL OVERRIDE: pass --svm-weight 0.4 on the command line, or set forced_svm_w
    #   by editing the call in main(). This skips the search entirely.
    #
    # TO HARDCODE FIXED WEIGHTS in the code (no command line):
    #   Replace the entire if/else block below with:
    #     SVM_W = 0.4
    #     RF_W  = 0.6
    p_svm_val = svm.predict_proba(X_val_sel)   # shape: (n_val, 3)
    p_rf_val  = rf.predict_proba(X_val_rf)     # shape: (n_val, 3)
    if forced_svm_w is not None:
        # Clip to [0, 1] to prevent invalid weights from command-line input.
        SVM_W = round(float(np.clip(forced_svm_w, 0.0, 1.0)), 2)
        RF_W  = round(1.0 - SVM_W, 2)
        print(f"\n  Ensemble weights: FIXED -> SVM {SVM_W:.0%} + RF {RF_W:.0%}  "
              f"(auto-tune skipped)")
    else:
        print(f"\n  Searching for best SVM/RF blend on validation set...")
        best_w, best_acc = 0.4, 0.0
        for w in np.arange(0.05, 1.0, 0.05):
            acc = accuracy_score(y_val, np.argmax(w * p_svm_val + (1-w) * p_rf_val, axis=1))
            if acc > best_acc:
                best_acc, best_w = acc, w
        SVM_W, RF_W = round(best_w, 2), round(1.0 - best_w, 2)
        print(f"  Best blend -> SVM {SVM_W:.0%} + RF {RF_W:.0%}")
    # Apply the chosen weights and show the resulting validation accuracy.
    p_combined_val = SVM_W * p_svm_val + RF_W * p_rf_val
    print(f"  Val accuracy (argmax): {accuracy_score(y_val, np.argmax(p_combined_val, axis=1))*100:.1f}%")

    # ── Normal threshold tuning ───────────────────────────────────────────────
    # The model sometimes predicts Pre-Seizure when the patient is actually Normal.
    # This threshold overrides a Pre-Seizure prediction to Normal when the model
    # is highly confident (>= threshold) that the patient is Normal.
    # Seizure predictions are NEVER downgraded — patient safety takes priority.
    #
    # The search tests thresholds from 0.30 to 0.64 in steps of 0.01 and picks
    # the one that maximises validation accuracy.
    #
    # To disable this correction: set NORMAL_THRESHOLD = 0.0 (no overrides will happen).
    # To be more aggressive (fewer false Pre-Seizure): increase upper bound to 0.70.
    print(f"\n  Tuning Normal threshold on validation set "
          f"(Pre-Seizure → Normal only, Seizure protected)...")
    argmax_val = np.argmax(p_combined_val, axis=1)
    best_thresh, best_thresh_acc = 0.50, 0.0
    for t in np.arange(0.30, 0.65, 0.01):
        y_t = argmax_val.copy()
        # Only convert Pre-Seizure (1) -> Normal (0) when Normal confidence >= threshold.
        y_t[(argmax_val == 1) & (p_combined_val[:, 0] >= t)] = 0
        acc_t = accuracy_score(y_val, y_t)
        if acc_t > best_thresh_acc:
            best_thresh_acc, best_thresh = acc_t, t
    NORMAL_THRESHOLD = round(float(best_thresh), 2)
    print(f"  Best Normal threshold: {NORMAL_THRESHOLD:.2f}  "
          f"(val acc with threshold: {best_thresh_acc*100:.1f}%)")

    return scaler, selector, svm, rf, SVM_W, RF_W, NORMAL_THRESHOLD


# =============================================================================
# SECTION: STEP 4 — EVALUATE on all 3 splits
# =============================================================================
# evaluate_split runs the full ensemble prediction pipeline on a single data split
# and returns a dict containing every performance metric and the confusion matrix.
#
# Pipeline for each sample:
#   1. Scale features with the fitted scaler (no re-fitting — prevents leakage).
#   2. Select the top N_SELECT features with the RF selector.
#   3. SVM predicts class probabilities (needs scaled+selected features).
#   4. RF predicts class probabilities (needs unscaled+selected features).
#   5. Blend: combined = svm_w × p_svm + rf_w × p_rf
#   6. Override: if argmax=Pre-Seizure AND Normal confidence >= threshold → Normal.
#   7. Take argmax of combined probabilities as the final prediction.
def evaluate_split(X, y_true, scaler, selector, svm, rf, label,
                   svm_w=0.4, rf_w=0.6, normal_threshold=0.50):
    # Scale and select features for each model.
    X_sc     = scaler.transform(X)              # scale (no fit — prevents leakage)
    X_sc_sel = selector.transform(X_sc)         # scaled + selected, for SVM
    X_rf_sel = selector.transform(X)            # unscaled + selected, for RF
    p_svm    = svm.predict_proba(X_sc_sel)
    p_rf     = rf.predict_proba(X_rf_sel)
    combined = svm_w * p_svm + rf_w * p_rf

    # Apply Normal threshold: override Pre-Seizure → Normal when confident enough.
    # Seizure predictions are never downgraded — protecting patient safety.
    argmax   = np.argmax(combined, axis=1)
    y_pred   = argmax.copy()
    y_pred[(argmax == 1) & (combined[:, 0] >= normal_threshold)] = 0
    confs    = combined   # keep blended probabilities for confidence distribution plots

    # Identify which classes are actually present (may differ from expected on small splits).
    present = sorted(np.unique(np.concatenate([y_true, y_pred])))
    lnames  = [CLASSES[i] for i in present]

    # ── Overall metrics ────────────────────────────────────────────────────────
    acc    = accuracy_score(y_true, y_pred)
    # "weighted" averaging weights each class by its sample count (handles imbalance).
    prec   = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1     = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm     = confusion_matrix(y_true, y_pred)

    # Seizure Recall: fraction of actual seizures correctly identified.
    # The most clinically important metric — missing a seizure can be life-threatening.
    sz_rec = (recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)
              if 2 in y_true else 0.0)

    # Specificity: fraction of actual Normal cases correctly identified.
    # tn = true normals; fp = normals misclassified as seizure/pre-seizure.
    tn = cm[0,0] if 0 in y_true else 0
    fp = cm[0,1:].sum() if 0 in y_true else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ── Per-class breakdown ────────────────────────────────────────────────────
    # Reports precision, recall, and F1 separately for Normal, Pre-Seizure, and Seizure.
    per_class = {}
    for i in present:
        mask = y_true == i
        if mask.any():
            per_class[CLASSES[i]] = {
                "count"    : int(mask.sum()),
                "precision": float(round(precision_score(y_true==i, y_pred==i, zero_division=0), 3)),
                "recall"   : float(round(recall_score(y_true==i, y_pred==i, zero_division=0), 3)),
                "f1"       : float(round(f1_score(y_true==i, y_pred==i, zero_division=0), 3)),
            }

    m = dict(label=label, acc=acc, prec=prec, rec=rec, f1=f1,
             spec=spec, sz_rec=sz_rec, cm=cm, y_true=y_true,
             y_pred=y_pred, confs=confs, present=present,
             lnames=lnames, per_class=per_class)

    # ── Print a formatted table to the terminal ───────────────────────────────
    w = 14
    print(f"\n  {'-'*60}")
    print(f"  {label}")
    print(f"  {'-'*60}")
    print(f"  Samples        : {len(y_true)}")
    # Flag the accuracy level: OK >= 90%, WARN >= 75%, FAIL below 75%.
    print(f"  Accuracy       : {acc*100:.1f}%  "
          f"{'OK' if acc>=0.90 else 'WARN ' if acc>=0.75 else 'FAIL'}")
    print(f"  Precision (w)  : {prec*100:.1f}%")
    print(f"  Recall (w)     : {rec*100:.1f}%")
    print(f"  F1 (w)         : {f1*100:.1f}%")
    print(f"  Specificity    : {spec*100:.1f}%")
    print(f"  Seizure Recall : {sz_rec*100:.1f}%  "
          f"# most clinically important metric")
    print()
    print(f"  Confusion Matrix  (rows = True, cols = Predicted):")
    print(f"  {'':>{w}}", end="")
    for l in lnames: print(f"  {l:>{w}}", end="")
    print()
    for i, row in zip(present, cm):
        print(f"  {CLASSES[i]:>{w}}", end="")
        for v in row: print(f"  {v:>{w}d}", end="")
        print()
    print()
    print(f"  Per-class breakdown:")
    for cls_name, d in per_class.items():
        print(f"    {cls_name:<14}  n={d['count']:4d}  "
              f"P={d['precision']:.2f}  R={d['recall']:.2f}  F1={d['f1']:.2f}")

    return m


# =============================================================================
# SECTION: STEP 5 — SAVE MODELS + REPORT
# =============================================================================
# save_models writes all artifacts needed to deploy the model:
#   scaler.pkl          — must be applied to new data before SVM inference
#   selector.pkl        — reduces 247 features to N_SELECT for SVM and RF
#   svm_model.pkl       — trained SVM classifier
#   rf_model.pkl        — trained Random Forest classifier
#   ensemble_weights.pkl— svm_w, rf_w, and normal_threshold used at inference
#   training_report.json— full metrics and settings for all three splits
#   HOW_TO_USE.txt      — quick-start instructions for activating this model
#   neurowatch_metrics.json — updates the live dashboard with the new test results
def save_models(scaler, selector, svm, rf, train_m, val_m, test_m, out_dir,
                svm_w=0.4, rf_w=0.6, normal_threshold=0.50):
    os.makedirs(out_dir, exist_ok=True)

    # Save all model objects as compressed pickle files.
    joblib.dump(scaler,   os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(selector, os.path.join(out_dir, "selector.pkl"))
    joblib.dump(svm,      os.path.join(out_dir, "svm_model.pkl"))
    joblib.dump(rf,       os.path.join(out_dir, "rf_model.pkl"))
    # Save the ensemble blend weights as a dict so inference code can load them.
    joblib.dump({"svm_w": svm_w, "rf_w": rf_w, "normal_threshold": normal_threshold},
                os.path.join(out_dir, "ensemble_weights.pkl"))

    # ── JSON training report ──────────────────────────────────────────────────
    # A machine-readable record of all training settings and results.
    # Useful for comparing different model versions without re-training.
    report = {
        "trained_on"       : "Mendeley Epileptic EEG Dataset",
        "timestamp"        : __import__("datetime").datetime.now().isoformat(),
        "split"            : "stratified 80/10/10",
        "n_features"       : N_FEATURES,
        "n_selected"       : N_SELECT,
        "n_channels"       : N_CHANNELS,
        "feature_selector" : "RF importance (bootstrap 300 trees)",
        "normalisation"    : "Option A -- relative band power proportions",
        "svm_class_weight" : "{0:1.0, 1:1.8, 2:1.0}",
        "normal_threshold" : normal_threshold,
        "model"            : f"Hybrid SVM ({svm_w:.0%}) + RF ({rf_w:.0%}) -- auto-tuned",
        "training": {
            "samples"      : int(len(train_m["y_true"])),
            "accuracy"     : round(train_m["acc"], 4),
            "f1"           : round(train_m["f1"],  4),
            "seizure_recall": round(train_m["sz_rec"], 4),
            "per_class"    : train_m["per_class"],
        },
        "validation": {
            "samples"      : int(len(val_m["y_true"])),
            "accuracy"     : round(val_m["acc"], 4),
            "f1"           : round(val_m["f1"],  4),
            "seizure_recall": round(val_m["sz_rec"], 4),
            # Overfitting gap: how much training accuracy exceeds validation accuracy.
            # A gap above 0.12 (12 pp) suggests the model is memorising training data.
            "overfitting_gap": round(train_m["acc"] - val_m["acc"], 4),
            "per_class"    : val_m["per_class"],
        },
        "test": {
            "samples"      : int(len(test_m["y_true"])),
            "accuracy"     : round(test_m["acc"], 4),
            "f1"           : round(test_m["f1"],  4),
            "seizure_recall": round(test_m["sz_rec"], 4),
            "per_class"    : test_m["per_class"],
        },
        "confusion_matrices": {
            "training"  : train_m["cm"].tolist(),
            "validation": val_m["cm"].tolist(),
            "test"      : test_m["cm"].tolist(),
        }
    }
    report_path = os.path.join(out_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Update the NeuroWatch live dashboard metrics ───────────────────────────
    # Writing neurowatch_metrics.json causes the Streamlit dashboard (dashboard_v2.py)
    # to automatically display the latest model performance on its next refresh.
    metrics_path = os.path.join(BASE_DIR, "json", "neurowatch_metrics.json")
    tm = test_m; vm = val_m
    metrics = {
        "timestamp"       : report["timestamp"],
        "model"           : os.path.basename(out_dir),
        "feature_selector": report.get("feature_selector", "none"),
        "svm_class_weight": report.get("svm_class_weight", "balanced"),
        "normal_threshold": report.get("normal_threshold", 0.5),
        "ensemble"        : report["model"],
        "accuracy"        : round(tm["acc"],    3),
        "precision"       : round(tm["prec"],   3),
        "recall"          : round(tm["rec"],    3),
        "f1"              : round(tm["f1"],     3),
        "specificity"     : round(tm["spec"],   3),
        "seizure_recall"  : round(tm["sz_rec"], 3),
        "overfitting_gap" : round(train_m["acc"] - vm["acc"], 3),
        "confusion_matrix": tm["cm"].tolist(),
        "n_samples"       : int(len(tm["y_true"])),
        "class_counts"    : {
            CLASSES[c]: int(np.sum(tm["y_true"] == c))
            for c in np.unique(tm["y_true"])
        },
        "per_class"       : tm["per_class"],
    }
    try:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"     neurowatch_metrics.json  (dashboard updated)")
    except Exception as e:
        print(f"  WARN  Could not write metrics JSON: {e}")

    # ── HOW_TO_USE.txt ────────────────────────────────────────────────────────
    # A plain-text quick-start guide saved alongside the model files.
    # Tells a user exactly which commands to run to activate this model on the Pi.
    model_name = os.path.basename(out_dir)
    how_to     = os.path.join(out_dir, "HOW_TO_USE.txt")
    with open(how_to, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Ensemble: SVM {svm_w:.0%} + RF {rf_w:.0%}\n")
        f.write(f"Features: {N_FEATURES} -> {N_SELECT} selected (RF importance)\n")
        f.write(f"Normal threshold: {normal_threshold:.2f}\n")
        f.write("\n")
        f.write("-- Run the Pi engine (main system) --\n")
        f.write(f'python src/main_pi_bios_v15.py --models "models/{model_name}"\n')
        f.write("\n")
        f.write("-- Run the dashboard (doctor view) --\n")
        f.write("streamlit run dashboard_v2.py\n")
        f.write("\n")
        f.write("-- Evaluate on test set --\n")
        f.write(f'python scripts/evaluation/eval_preictal_backup.py '
                f'--model-dir "models/{model_name}" '
                f'--data "data/Mendelay dataset/Npy_files_preictal"\n')

    print(f"\n  OK Saved to: {out_dir}")
    print(f"     scaler.pkl")
    print(f"     svm_model.pkl")
    print(f"     rf_model.pkl")
    print(f"     training_report.json")
    print(f"     HOW_TO_USE.txt")

    # ── MLflow logging (optional — silently skipped if mlflow not installed) ──
    # Records this training run so multiple runs (different seeds, hyperparameters,
    # feature counts, weight ratios) can be compared in the MLflow UI.
    # View later with:   mlflow ui   →   http://localhost:5000
    try:
        import mlflow
        mlflow.set_tracking_uri(f"file:{os.path.join(BASE_DIR, 'mlruns')}")
        mlflow.set_experiment("neurowatch_training")
        with mlflow.start_run(run_name=os.path.basename(out_dir)):
            mlflow.log_params({
                "n_features"      : N_FEATURES,
                "n_selected"      : N_SELECT,
                "n_channels"      : N_CHANNELS,
                "svm_weight"      : svm_w,
                "rf_weight"       : rf_w,
                "normal_threshold": normal_threshold,
                "feature_selector": "RF importance (bootstrap 300 trees)",
                "normalisation"   : "Option A (relative band power)",
            })
            for split_name, m in [("train", train_m), ("val", val_m), ("test", test_m)]:
                mlflow.log_metric(f"{split_name}_accuracy",       m["acc"])
                mlflow.log_metric(f"{split_name}_f1",             m["f1"])
                mlflow.log_metric(f"{split_name}_seizure_recall", m["sz_rec"])
            mlflow.log_metric("overfitting_gap", train_m["acc"] - val_m["acc"])
            # Attach the training report + the model pickles as run artifacts.
            mlflow.log_artifact(report_path)
            mlflow.log_artifacts(out_dir, artifact_path="models")
        print(f"     mlflow run logged  (run `mlflow ui` from {BASE_DIR})")
    except ImportError:
        pass  # mlflow not installed — silent skip
    except Exception as e:
        print(f"  WARN  MLflow logging failed: {e}")

    return report


# =============================================================================
# SECTION: STEP 6 — PLOTS
# =============================================================================
# plot_all generates four diagnostic figures.
# Figures are only created if --no-plots was NOT passed and matplotlib initialised OK.
def plot_all(train_m, val_m, test_m, out_dir=None):
    if not PLOT_OK: return

    # Helper: save figure to disk if an output directory is provided.
    def _save(fig, name):
        if out_dir:
            path = os.path.join(out_dir, name)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")

    # ── Figure 1: Confusion matrices for all three splits ─────────────────────
    # Rows = true class, Columns = predicted class.
    # Diagonal cells = correct predictions; off-diagonal = errors.
    # The Seizure row (bottom) is the most clinically important to check.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("NeuroWatch  |  Confusion Matrices -- All Splits",
                 fontsize=13, fontweight="bold")
    cmap_colors = ["Blues", "Greens", "Oranges"]
    for ax, m, cmap in zip(axes, [train_m, val_m, test_m], cmap_colors):
        # annot=True writes the integer count inside each cell.
        sns.heatmap(m["cm"], annot=True, fmt="d", cmap=cmap,
                    xticklabels=m["lnames"], yticklabels=m["lnames"],
                    cbar=False, ax=ax, square=True, annot_kws={"size": 12})
        ax.set_title(f"{m['label']}\nAcc: {m['acc']*100:.1f}%  "
                     f"F1: {m['f1']*100:.1f}%",
                     fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "confusion_matrices.png")
    plt.show(block=False); plt.pause(0.3)

    # ── Figure 2: Metrics comparison — Train vs Val vs Test ───────────────────
    # Six metrics side by side for each split.
    # Train and Val should be close; large gap = overfitting.
    # Test is the definitive result to report in coursework or papers.
    fig2, ax = plt.subplots(figsize=(13, 6))
    fig2.suptitle("NeuroWatch  |  Metrics Comparison -- Train / Val / Test",
                  fontsize=13, fontweight="bold")
    m_labels = ["Accuracy","Precision","Recall","F1","Specificity","Seizure Recall"]
    tr_vals  = [train_m["acc"],  train_m["prec"],  train_m["rec"],
                train_m["f1"],   train_m["spec"],  train_m["sz_rec"]]
    vl_vals  = [val_m["acc"],    val_m["prec"],    val_m["rec"],
                val_m["f1"],     val_m["spec"],    val_m["sz_rec"]]
    te_vals  = [test_m["acc"],   test_m["prec"],   test_m["rec"],
                test_m["f1"],    test_m["spec"],   test_m["sz_rec"]]
    x = np.arange(len(m_labels)); w = 0.26
    b1 = ax.bar(x - w,   tr_vals, w, label="Train (80%)", color="#4fc3f7", alpha=0.9, edgecolor="#333")
    b2 = ax.bar(x,       vl_vals, w, label="Val   (10%)", color="#00CC66", alpha=0.9, edgecolor="#333")
    b3 = ax.bar(x + w,   te_vals, w, label="Test  (10%)", color="#FFD633", alpha=0.9, edgecolor="#333")
    ax.set_xticks(x); ax.set_xticklabels(m_labels, fontsize=10)
    ax.set_ylim(0, 1.18); ax.set_ylabel("Score")
    ax.set_title("All Splits Comparison")
    ax.legend(fontsize=10, framealpha=0.3); ax.grid(True, axis="y")
    for bars in [b1, b2, b3]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h+0.015,
                    f"{h*100:.0f}%", ha="center", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig2, "metrics_comparison.png")
    plt.show(block=False); plt.pause(0.3)

    # ── Figure 3: Confidence distributions on the validation set ─────────────
    # For each true class, shows a histogram of the model's confidence score
    # for that class. A well-calibrated model has high confidence peaks near 1.0.
    # Wide or low-centred histograms suggest uncertainty (check these classes carefully).
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle("NeuroWatch  |  Confidence Distributions -- Validation Set",
                  fontsize=12, fontweight="bold")
    for col, cls_idx in enumerate(range(3)):
        ax   = axes3[col]
        mask = val_m["y_true"] == cls_idx
        if mask.any():
            c = val_m["confs"][mask, cls_idx]
            ax.hist(c, bins=25, color=COLORS[cls_idx], alpha=0.85, edgecolor="#333")
            ax.axvline(c.mean(), color="#fff", ls="--", lw=1.5,
                       label=f"mean={c.mean():.2f}")
            ax.legend(fontsize=9)
        ax.set_title(f"True {CLASSES[cls_idx]}")
        ax.set_xlabel("Model confidence  ->  {CLASSES[cls_idx]}")
        ax.set_xlim(0, 1); ax.grid(True)
    plt.tight_layout()
    _save(fig3, "confidence_distributions.png")
    plt.show(block=False); plt.pause(0.3)

    # ── Figure 4: Overfitting check — Train minus Val gap ────────────────────
    # The gap shows how much better the model performs on training data vs unseen data.
    # Green bar  (< 5 pp): no overfitting — the model generalises well.
    # Yellow bar (< 12 pp): mild overfitting — acceptable for most applications.
    # Red bar    (>= 12 pp): high overfitting — add regularisation, reduce max_depth, or get more data.
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    fig4.suptitle("NeuroWatch  |  Overfitting Check -- Train vs Validation",
                  fontsize=12, fontweight="bold")
    gaps = [t - v for t, v in zip(tr_vals, vl_vals)]
    cols = ["#00CC66" if abs(g) < 0.05 else "#FFD633" if abs(g) < 0.12 else "#FF3333"
            for g in gaps]
    bars4 = ax4.bar(m_labels, gaps, color=cols, edgecolor="#333")
    ax4.axhline(0, color="#aaa", lw=1)
    ax4.axhline(0.05,  color="#00CC66", lw=1, ls="--", alpha=0.5, label="#5% (good)")
    ax4.axhline(-0.05, color="#00CC66", lw=1, ls="--", alpha=0.5)
    ax4.axhline(0.12,  color="#FFD633", lw=1, ls="--", alpha=0.5, label="#12% (mild)")
    ax4.axhline(-0.12, color="#FFD633", lw=1, ls="--", alpha=0.5)
    ax4.set_ylabel("Train # Validation gap"); ax4.set_ylim(-0.3, 0.3)
    ax4.legend(fontsize=9, framealpha=0.3); ax4.grid(True, axis="y")
    for b, g in zip(bars4, gaps):
        ax4.text(b.get_x()+b.get_width()/2,
                 g + (0.01 if g >= 0 else -0.02),
                 f"{g:+.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    _save(fig4, "overfitting_check.png")
    plt.show(block=False); plt.pause(0.3)


# =============================================================================
# SECTION: FINAL SUMMARY
# =============================================================================
# Prints a formatted table of all key metrics to the terminal at the end of training.
# The overfitting gap (train accuracy − val accuracy) is highlighted with a pass/warn/fail rating.
def print_summary(train_m, val_m, test_m, report, svm_w=0.5, rf_w=0.5):
    overfit = train_m["acc"] - val_m["acc"]
    # Rate the overfitting level: < 5 pp is good, 5–12 pp is mild, > 12 pp is high.
    if overfit < 0.05:   ov_msg = "OK  No overfitting"
    elif overfit < 0.12: ov_msg = "WARN  Mild overfitting -- acceptable"
    else:                ov_msg = "FAIL  High overfitting -- consider regularisation"

    print("\n" + "+" + "="*65 + "+")
    print("|" + "  TRAINING COMPLETE -- FINAL SUMMARY".center(65) + "|")
    print("+" + "="*65 + "+")
    print(f"|  {'Metric':<20} {'Train (80%)':>14} {'Val (10%)':>12} {'Test (10%)':>12} |")
    print("+" + "-"*65 + "+")
    # Print the four most important metrics for all three splits in one table.
    rows = [
        ("Accuracy",        train_m["acc"],    val_m["acc"],    test_m["acc"]),
        ("F1 (weighted)",   train_m["f1"],     val_m["f1"],     test_m["f1"]),
        ("Specificity",     train_m["spec"],   val_m["spec"],   test_m["spec"]),
        ("Seizure Recall",  train_m["sz_rec"], val_m["sz_rec"], test_m["sz_rec"]),
    ]
    for name, tv, vv, ev in rows:
        print(f"|  {name:<20} {tv*100:>13.1f}%  {vv*100:>11.1f}%  {ev*100:>11.1f}% |")
    print("+" + "-"*65 + "+")
    print(f"|  {'Overfitting gap':<20} {'(train - val)':>14} {overfit*100:>+10.1f}pp  "
          f"{'':>12} |")
    print("+" + "="*65 + "+")
    print(f"|  {ov_msg:<63} |")
    print("+" + "-"*65 + "+")
    feat_desc = f"{N_CHANNELS}ch x {N_BANDS+N_STATS} per-ch + {N_CORR} corr = {N_FEATURES}"
    print(f"|  {'Features':<20} {feat_desc:>43} |")
    print(f"|  {'Split':<20} {'Stratified 80 / 10 / 10':>43} |")
    print(f"|  {'Normalisation':<20} {'Option A -- relative band proportions':>43} |")
    print(f"|  {'Model':<20} {f'Hybrid SVM ({svm_w*100:.0f}%) + RF ({rf_w*100:.0f}%)':>43} |")
    print("+" + "="*65 + "+")


# =============================================================================
# SECTION: MAIN — training pipeline entry point
# =============================================================================
# Orchestrates the full six-step training pipeline in sequence.
def main():
    print("\n" + "="*65)
    print("  NeuroWatch  --  Mendeley Model Trainer")
    print("="*65)
    print(f"  Data path    : {args.data}")
    print(f"  Output path  : {args.out}")
    print(f"  Features     : {N_CHANNELS} channels x {N_BANDS} bands + {N_STATS} stats = {N_FEATURES}")
    print(f"  Split        : Stratified 80 / 10 / 10")
    print(f"  Normalisation: Option A (relative band power) + statistical features")
    print(f"  SMOTE        : {'enabled' if SMOTE_AVAILABLE else 'disabled (pip install imbalanced-learn)'}")
    print(f"  Random seed  : {RANDOM_SEED}")

    # ── Step 1+2: Load data and extract features ──────────────────────────────
    # Returns numpy arrays (X_train, y_train, X_val, y_val, X_test, y_test).
    # Each X array has shape (n_samples, N_FEATURES=247).
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_extract(args.data)

    # ── Step 3: Train all models ──────────────────────────────────────────────
    # Returns the fitted scaler, selector, SVM, RF, and the chosen blend weights.
    # Pass forced_svm_w=args.svm_weight to use a manual blend ratio.
    scaler, selector, svm, rf, svm_w, rf_w, normal_threshold = train(
        X_train, y_train, X_val, y_val, forced_svm_w=args.svm_weight)

    # ── Step 4: Evaluate all three splits ────────────────────────────────────
    # Evaluate on Train/Val/Test using the final ensemble and normal threshold.
    # Test set results are the definitive figures to report.
    print(f"\n{'-'*65}")
    print(f"  STEP 4: Evaluating all three splits...")
    print(f"{'-'*65}")
    train_m = evaluate_split(X_train, y_train, scaler, selector, svm, rf, "Train (80%)",
                             svm_w, rf_w, normal_threshold)
    val_m   = evaluate_split(X_val,   y_val,   scaler, selector, svm, rf, "Validation (10%)",
                             svm_w, rf_w, normal_threshold)
    test_m  = evaluate_split(X_test,  y_test,  scaler, selector, svm, rf, "Test (10%)",
                             svm_w, rf_w, normal_threshold)

    # ── Step 5: Save models, report, and dashboard metrics ───────────────────
    print(f"\n{'-'*65}")
    print(f"  STEP 5: Saving models...")
    print(f"{'-'*65}")
    report = save_models(scaler, selector, svm, rf, train_m, val_m, test_m, args.out,
                         svm_w, rf_w, normal_threshold)

    # ── Step 5b (optional): MLflow experiment tracking ───────────────────────
    # Logs all hyperparameters and metrics to a local SQLite database (mlflow.db).
    # View results: mlflow ui
    # All parameters are logged so you can compare multiple training runs in a table.
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("NeuroWatch-EEG")
        with mlflow.start_run(run_name=os.path.basename(args.out)):
            # Log all hyperparameters so every run is fully reproducible.
            mlflow.log_param("svm_C",             svm.C)
            mlflow.log_param("svm_kernel",         svm.kernel)
            mlflow.log_param("rf_n_estimators",    rf.n_estimators)
            mlflow.log_param("rf_max_depth",       rf.max_depth)
            mlflow.log_param("rf_min_samples_leaf",rf.min_samples_leaf)
            mlflow.log_param("svm_weight",         svm_w)
            mlflow.log_param("rf_weight",          rf_w)
            mlflow.log_param("n_features",         N_FEATURES)
            mlflow.log_param("n_selected",         N_SELECT)
            mlflow.log_param("normal_threshold",   normal_threshold)
            mlflow.log_param("svm_class_weight",   "{0:1.0, 1:1.8, 2:1.0}")
            mlflow.log_param("feature_selector",   "rf_importance")
            mlflow.log_param("n_channels",         N_CHANNELS)
            mlflow.log_param("smote",              SMOTE_AVAILABLE)
            mlflow.log_param("data_path",          os.path.basename(args.data))
            mlflow.log_param("random_seed",        RANDOM_SEED)

            # Log training metrics.
            mlflow.log_metric("train_accuracy",      round(train_m["acc"],    4))
            mlflow.log_metric("train_f1",            round(train_m["f1"],     4))
            mlflow.log_metric("train_seizure_recall", round(train_m["sz_rec"], 4))

            # Log validation metrics.
            mlflow.log_metric("val_accuracy",        round(val_m["acc"],    4))
            mlflow.log_metric("val_f1",              round(val_m["f1"],     4))
            mlflow.log_metric("val_seizure_recall",  round(val_m["sz_rec"], 4))

            # Log test metrics — these are the definitive performance figures.
            mlflow.log_metric("test_accuracy",       round(test_m["acc"],    4))
            mlflow.log_metric("test_f1",             round(test_m["f1"],     4))
            mlflow.log_metric("test_seizure_recall", round(test_m["sz_rec"], 4))

            # Overfitting gap: the single most useful diagnostic number.
            # A gap above 0.12 means the model performs much better on training than test data.
            mlflow.log_metric("overfitting_gap",
                              round(train_m["acc"] - val_m["acc"], 4))

            # Log the model objects themselves so they can be loaded later via MLflow.
            mlflow.sklearn.log_model(svm, "svm_model")
            mlflow.sklearn.log_model(rf,  "rf_model")
            mlflow.sklearn.log_model(scaler, "scaler")

            # Attach the training report JSON as an artifact for easy download.
            report_path = os.path.join(args.out, "training_report.json")
            mlflow.log_artifact(report_path)

            print(f"\n  MLflow run logged  ->  mlflow ui  (localhost:5000)")

    # ── Step 6: Print summary table ───────────────────────────────────────────
    print_summary(train_m, val_m, test_m, report)

    # ── Step 7: Show plots (if enabled) ──────────────────────────────────────
    plot_all(train_m, val_m, test_m, args.out)

    if PLOT_OK:
        print("\n  All plots open -- press Enter to close and exit...")
        input()
        plt.close("all")


# ── Entry point ────────────────────────────────────────────────────────────────
# Only run main() when this script is executed directly (not imported as a module).
if __name__ == "__main__":
    main()
