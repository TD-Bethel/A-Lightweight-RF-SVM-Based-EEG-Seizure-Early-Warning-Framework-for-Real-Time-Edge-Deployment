# =============================================================================
# neurowatch_metrics.py  —  NeuroWatch Dataset Metrics Pipeline
#
# HOW THIS FILE FITS INTO THE PROJECT:
#   This is an OFFLINE evaluation script — it does NOT run on the Raspberry Pi
#   in real time. You run it once on your laptop/PC with the Mendeley dataset
#   to measure how accurately the hybrid SVM + Random Forest model performs.
#   The numbers it produces (accuracy, F1, AUC, seizure recall, etc.) are saved
#   to a JSON file that the dashboard reads to display the "Model Performance"
#   panel.
#
# PIPELINE (10 steps, each clearly labelled below):
#   1. Load pre-split .npy arrays from the Mendeley dataset folder
#   2. Extract frequency-band power features (delta, theta, alpha, beta)
#   3. Train the same SVM + Random Forest hybrid used on the Pi
#   4. Run predictions on the test set
#   5. Compute 4-class metrics (original dataset labels)
#   6. Map to 3-class metrics (NeuroWatch: Normal / Pre-Seizure / Seizure)
#   7. Compute seizure-specific metrics (sensitivity, false alarm rate, etc.)
#   8. Run 5-fold cross-validation on the training set
#   9. Save all metrics to JSON (dashboard-compatible format)
#   10. Print a final summary to the terminal
#
# Run:
#   python neurowatch_metrics.py
#   python neurowatch_metrics.py /custom/path/to/Npy_files   # override path
#
# Requirements:
#   pip install numpy scikit-learn scipy matplotlib seaborn
# =============================================================================


# =============================================================================
# SECTION 1 — IMPORTS
#
# WHY each library is needed:
#   os, sys     — file path manipulation and command-line argument access.
#   json        — writing the output metrics to a .json file.
#   numpy       — array operations on EEG data (shape: epochs × channels × time).
#   warnings    — suppress sklearn convergence warnings that clutter the output.
#   datetime    — timestamp the output JSON so you know when it was generated.
#   scipy.signal.welch — Power Spectral Density estimation (converts raw EEG
#                  waveform into frequency-band powers).
#   sklearn.*   — machine learning: scaling, SVM, Random Forest, cross-validation,
#                  and all the metrics (accuracy, precision, recall, F1, AUC, etc.).
#
# HOW TO CHANGE:
#   - If you want to try a different classifier (e.g., XGBoost), add its import
#     here and update train_hybrid() below.
#   - If you want to plot ROC curves, add: import matplotlib.pyplot as plt
# =============================================================================
import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")   # Suppress noisy sklearn convergence warnings

from datetime import datetime
from scipy.signal import welch       # Welch's method for PSD — more stable than FFT
from sklearn.preprocessing import StandardScaler       # Z-score normalisation for SVM
from sklearn.svm import SVC                            # Support Vector Classifier
from sklearn.ensemble import RandomForestClassifier    # Ensemble of decision trees
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)


# =============================================================================
# SECTION 2 — FILE PATHS
#
# WHY: Using os.path.abspath + dirname to build paths means the script works
# regardless of which directory you run it from (as long as the project
# folder structure is intact).
#
# HOW TO CHANGE:
#   - NPY_PATH: update this if you move the Mendeley .npy files. You can also
#     override it at runtime: python neurowatch_metrics.py /new/path
#   - METRICS_JSON_OUT: secondary output file in the project json/ folder.
#   - METRICS_JSON_DASHBOARD: primary output read by dashboard.py. On the Pi
#     this should be /tmp/neurowatch_metrics.json or wherever dashboard.py
#     expects it.
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPY_PATH = os.path.join(PROJECT_ROOT, "data", "Test dataset", "Npy_files")

# Two output paths for the metrics JSON:
#   METRICS_JSON_OUT       — saved alongside the project files (permanent record)
#   METRICS_JSON_DASHBOARD — read live by dashboard.py (may be on the Pi or /tmp)
METRICS_JSON_OUT      = os.path.join(PROJECT_ROOT, "json", "neurowatch_metrics_output.json")
METRICS_JSON_DASHBOARD = os.path.join(PROJECT_ROOT, "json", "neurowatch_metrics.json")


# =============================================================================
# SECTION 3 — CLASS DEFINITIONS (4-class original dataset labels)
#
# WHY: The Mendeley dataset labels EEG epochs with integer class IDs 0–3.
# CLASS_NAMES maps those integers to human-readable strings for printing
# and for including in the output JSON.
#
# The four classes:
#   0 = Normal            — no seizure activity
#   1 = Complex Partial   — focal seizure that impairs consciousness
#   2 = Electrographic    — seizure visible only on EEG, no physical symptoms
#   3 = Video-detected    — behavioural seizure with no clear EEG change
#
# HOW TO CHANGE: If you use a different dataset with different class IDs,
# update this dict. The rest of the code uses CLASS_NAMES.get(int(cls)) so
# it will adapt automatically.
# =============================================================================
CLASS_NAMES = {
    0: "Normal",
    1: "Complex Partial",
    2: "Electrographic",
    3: "Video-detected"
}


# =============================================================================
# SECTION 4 — 3-CLASS MAPPING (NeuroWatch operational classes)
#
# WHY: NeuroWatch uses three states for clinical simplicity:
#   Normal      — no intervention needed
#   Pre-Seizure — early warning, patient should find a safe place
#   Seizure     — alert, contact caregiver/doctor
#
# MERGE_MAP translates the 4-class dataset labels to these 3 classes:
#   0 (Normal)      → 0 (Normal)       — unchanged
#   1 (CPS)         → 2 (Seizure)      — clinically clear seizure
#   2 (Electro)     → 2 (Seizure)      — EEG-confirmed seizure
#   3 (Video)       → 1 (Pre-Seizure)  — subtle/uncertain signal → early warning
#
# HOW TO CHANGE:
#   - If you want to merge video-detected into Seizure instead of Pre-Seizure,
#     change MERGE_MAP[3] to 2.
#   - If you add a new dataset class, add a mapping entry here.
# =============================================================================
MERGE_MAP = {
    0: 0,   # Normal      → Normal       (class index 0 in 3-class system)
    1: 2,   # CPS         → Seizure      (class index 2 in 3-class system)
    2: 2,   # Electro     → Seizure      (class index 2 in 3-class system)
    3: 1,   # Video       → Pre-Seizure  (class index 1 in 3-class system)
}
MERGED_NAMES = ["Normal", "Pre-Seizure", "Seizure"]  # Index matches MERGE_MAP output values


# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def load_data(npy_path):
    """
    Load pre-split train/test arrays from the Mendeley Epileptic EEG dataset.

    WHY .npy format: NumPy binary files load much faster than CSV and preserve
    exact float32/float64 values without rounding errors.

    EXPECTED FILES in npy_path/:
      x_train.npy — shape (n_train_epochs, n_channels, n_timepoints)
                    e.g., (7011, 19, 2500) for 500 Hz × 5 s epochs
      y_train.npy — shape (n_train_epochs,) integer labels 0–3
      x_test.npy  — shape (n_test_epochs, n_channels, n_timepoints)
      y_test.npy  — shape (n_test_epochs,) integer labels 0–3

    ARGS:
      npy_path — absolute path to the folder containing the four .npy files.

    RETURNS: tuple (x_train, y_train, x_test, y_test) as NumPy arrays.

    HOW TO CHANGE:
      - If your dataset uses a different train/test split filename convention,
        update the `required` list and the np.load() calls below.
      - If your data is stored as a single .npy (not split), load it here and
        perform the split with sklearn.model_selection.train_test_split.
    """
    print("\n" + "="*60)
    print("📂 LOADING NPY DATA")
    print("="*60)

    # Verify all four required files exist before attempting to load.
    # sys.exit(1) aborts cleanly if any file is missing.
    required = ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy"]
    for f in required:
        full = os.path.join(npy_path, f)
        if not os.path.exists(full):
            print(f"❌ Missing: {full}")
            print("   Please check your NPY_PATH setting.")
            sys.exit(1)

    # Load all four arrays into memory. For large datasets this may take a few
    # seconds and several hundred MB of RAM.
    x_train = np.load(os.path.join(npy_path, "x_train.npy"))
    y_train = np.load(os.path.join(npy_path, "y_train.npy"))
    x_test  = np.load(os.path.join(npy_path, "x_test.npy"))
    y_test  = np.load(os.path.join(npy_path, "y_test.npy"))

    # Print shapes so you can confirm the data loaded correctly
    print(f"  x_train shape : {x_train.shape}  → {x_train.shape[0]} epochs")
    print(f"  y_train shape : {y_train.shape}")
    print(f"  x_test  shape : {x_test.shape}   → {x_test.shape[0]} epochs")
    print(f"  y_test  shape : {y_test.shape}")

    # Print per-class sample counts for both splits so you can detect imbalance
    print("\n  📊 Class distribution:")
    for split_name, y in [("Train", y_train), ("Test", y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        print(f"  {split_name}:")
        for cls, cnt in zip(unique, counts):
            pct = cnt / len(y) * 100
            print(f"    {CLASS_NAMES.get(int(cls), str(cls)):20s} "
                  f"label={int(cls)}  n={cnt:5d}  ({pct:.1f}%)")

    return x_train, y_train, x_test, y_test


# =============================================================================
# STEP 2 — FEATURE EXTRACTION
#
# WHY band-power features:
#   Raw EEG time-series are high-dimensional (19 channels × 2500 timepoints =
#   47,500 values per epoch). SVM and Random Forest work poorly on raw signals.
#   Instead, we compute the average power in four clinically meaningful
#   frequency bands per channel, reducing each epoch to 76 numbers
#   (19 channels × 4 bands).
#
#   The four EEG frequency bands and their clinical significance:
#     Delta  (1–4  Hz) — deep sleep / severe brain dysfunction
#     Theta  (4–8  Hz) — drowsiness / some seizure activity
#     Alpha  (8–13 Hz) — relaxed wakefulness / eyes closed
#     Beta   (13–30 Hz) — active thinking / alert state
#
# WHY Welch's method (not FFT):
#   Welch averages multiple overlapping FFT windows, which reduces noise in
#   the PSD estimate. This gives more stable features, especially for short
#   or noisy EEG segments.
#
# NOTE: main_pi_bios_v15.py uses only a single EEG channel at runtime (due
# to the single-channel EEG hat on the Pi). This offline script uses all 19
# channels from the research dataset for higher accuracy benchmarking.
#
# HOW TO CHANGE:
#   - To add gamma band (30–100 Hz): add a line for gamma = np.mean(psd[...])
#     and extend the features.extend() list.
#   - To use a different sampling rate: change fs=500 to match your hardware.
#   - To use raw time-domain features instead of band powers: replace the
#     Welch block with statistical features (mean, std, skewness, kurtosis).
# =============================================================================

def extract_band_powers(epoch, fs=500):
    """
    Extract delta/theta/alpha/beta band power from one multi-channel EEG epoch.

    ARGS:
      epoch — NumPy array of shape (n_channels, n_timepoints).
              Each row is one EEG channel's voltage time-series.
      fs    — sampling frequency in Hz. Default 500 Hz matches the Mendeley
              dataset. Change this if your EEG device uses a different rate.

    RETURNS:
      1-D NumPy array of length (n_channels × 4). For 19 channels: 76 values.
      Feature ordering: [ch0_delta, ch0_theta, ch0_alpha, ch0_beta,
                         ch1_delta, ..., ch18_beta]

    HOW TO CHANGE:
      - To add gamma (30–100 Hz): add gamma = np.mean(psd[(f>=30)&(f<=100)])
        and include it in features.extend([delta, theta, alpha, beta, gamma]).
      - nperseg=min(256, len(sig)) balances frequency resolution vs. variance.
        Increase to 512 for better low-frequency resolution (needs longer epochs).
    """
    features = []
    for ch in range(epoch.shape[0]):          # Loop over each EEG channel
        sig = epoch[ch].astype(float)          # One channel's waveform as float64
        # Welch PSD: f = frequency axis (Hz), psd = power at each frequency
        f, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)))

        # Average power within each clinical frequency band
        delta = np.mean(psd[(f >= 1)  & (f <= 4)])    # 1–4 Hz
        theta = np.mean(psd[(f >= 4)  & (f <= 8)])    # 4–8 Hz
        alpha = np.mean(psd[(f >= 8)  & (f <= 13)])   # 8–13 Hz
        beta  = np.mean(psd[(f >= 13) & (f <= 30)])   # 13–30 Hz

        features.extend([delta, theta, alpha, beta])   # Append 4 values per channel
    return np.array(features)


def extract_all_features(X, label=""):
    """
    Run extract_band_powers() on every epoch in the dataset.

    WHY: Iterates in a plain Python loop (not vectorised) because Welch's
    method doesn't have a batch API. The progress print every 500 epochs
    gives feedback during long runs (the full Mendeley train set can take
    several minutes).

    ARGS:
      X     — NumPy array of shape (n_epochs, n_channels, n_timepoints).
      label — string printed in the progress log (e.g., "train" or "test").

    RETURNS:
      2-D NumPy array of shape (n_epochs, n_features).

    HOW TO CHANGE:
      - To parallelise (much faster on multi-core machines), replace the loop
        with joblib.Parallel:
            from joblib import Parallel, delayed
            feats = Parallel(n_jobs=-1)(
                delayed(extract_band_powers)(epoch) for epoch in X)
      - To cache features to disk so you don't recompute each run:
            np.save("F_train_cache.npy", np.array(feats))
        and load with np.load() at the top of main.
    """
    import time
    print(f"\n  ⚙️  Extracting features from {label} ({X.shape[0]} epochs × "
          f"{X.shape[1]} channels)...")
    feats = []
    n  = X.shape[0]
    t0 = time.time()
    for i, epoch in enumerate(X):
        feats.append(extract_band_powers(epoch))
        # Print progress every 500 epochs with estimated time remaining
        if (i + 1) % 500 == 0:
            elapsed   = time.time() - t0
            rate      = (i + 1) / elapsed      # epochs per second
            remaining = (n - i - 1) / rate     # seconds left
            print(f"    {i+1}/{n} ({(i+1)/n*100:.0f}%)  ~{remaining:.0f}s remaining")
    total = time.time() - t0
    print(f"  ✅ Done in {total:.1f}s  →  feature matrix: {len(feats)} × {len(feats[0])}")
    return np.array(feats)


# =============================================================================
# STEP 3 — TRAIN HYBRID MODEL
#
# WHY a hybrid SVM + Random Forest:
#   - SVM with an RBF kernel is excellent at finding non-linear decision
#     boundaries in normalised feature spaces. It tends to be precise.
#   - Random Forest is an ensemble of decision trees. It handles class
#     imbalance well (via class_weight="balanced") and is robust to outliers.
#   - Blending the two (40% SVM + 60% RF probability) gives better
#     generalisation than either model alone, at the cost of training both.
#
# IMPORTANT: SVM requires StandardScaler (zero mean, unit variance) because
# it is distance-based — large-scale features would dominate the kernel.
# Random Forest is scale-invariant, so it receives raw (unscaled) features.
#
# HOW TO CHANGE:
#   - SVM hyperparameters: C controls regularisation (higher C = less
#     regularisation, tighter fit). gamma="scale" = 1/(n_features × X.var()).
#     Try C=1, C=5, C=10 and compare metrics.
#   - RF hyperparameters: n_estimators=100 trees is a good default. Increase
#     for slightly better accuracy at the cost of longer training. max_depth=15
#     prevents over-deep trees that memorise the training set.
#   - Blend ratio: change 0.4/0.6 in hybrid_predict_proba() to 0.5/0.5 for
#     equal weighting, or 0.0/1.0 to use only the Random Forest.
# =============================================================================

def train_hybrid(X_tr, y_tr):
    """
    Fit both the SVM and Random Forest on the training features.

    ARGS:
      X_tr — feature matrix, shape (n_train, n_features). Raw (unscaled).
      y_tr — integer label array, shape (n_train,).

    RETURNS: tuple (scaler, svm, rf)
      scaler — fitted StandardScaler (use scaler.transform() on new data)
      svm    — fitted SVC with probability=True (needed for predict_proba)
      rf     — fitted RandomForestClassifier

    HOW TO CHANGE: To add a third model (e.g., LogisticRegression), train it
    here and return it, then include it in hybrid_predict_proba().
    """
    print("\n" + "="*60)
    print("🎓 TRAINING HYBRID SVM + RANDOM FOREST")
    print("="*60)

    # Scale features to zero mean / unit variance — required for SVM
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)   # fit on train, then transform train

    # SVM with RBF kernel —  C=2 is a moderate regularisation strength
    # class_weight="balanced" up-weights minority classes (seizure types that
    # have fewer samples) so the SVM doesn't just predict "Normal" for everything
    print("  Training SVM (RBF kernel)...")
    svm = SVC(kernel="rbf", C=2, gamma="scale",
              probability=True,           # Needed for predict_proba()
              class_weight="balanced")    # Compensate for class imbalance
    svm.fit(X_tr_sc, y_tr)
    print("  ✅ SVM trained")

    # Random Forest — 100 trees, max depth 15 to avoid overfitting
    # n_jobs=-1 uses all available CPU cores in parallel
    print("  Training Random Forest (100 trees)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15,
                                class_weight="balanced", random_state=42,
                                n_jobs=-1)
    rf.fit(X_tr, y_tr)   # RF gets RAW (unscaled) features
    print("  ✅ Random Forest trained")

    return scaler, svm, rf


def hybrid_predict_proba(X, scaler, svm, rf):
    """
    Compute blended class probability matrix using 40% SVM + 60% RF.

    WHY blend: each model has complementary strengths. The 60/40 split
    gives slightly more weight to the RF which handles class imbalance better.

    ARGS:
      X      — raw feature matrix, shape (n_samples, n_features).
      scaler — fitted StandardScaler from train_hybrid().
      svm    — fitted SVC from train_hybrid().
      rf     — fitted RandomForestClassifier from train_hybrid().

    RETURNS: probability matrix, shape (n_samples, n_classes), where each
    row sums to 1.0.

    HOW TO CHANGE the blend ratio: edit the 0.4 and 0.6 constants.
    Example for RF-only: return rf.predict_proba(X)
    """
    X_sc = scaler.transform(X)    # Scale X the same way as training data
    # Weighted average of the two probability matrices
    return 0.4 * svm.predict_proba(X_sc) + 0.6 * rf.predict_proba(X)


def hybrid_predict(X, scaler, svm, rf):
    """
    Predict class labels and return probabilities for a feature matrix.

    ARGS:
      X      — raw feature matrix, shape (n_samples, n_features).
      scaler — fitted StandardScaler.
      svm    — fitted SVC.
      rf     — fitted RandomForestClassifier.

    RETURNS: tuple (y_pred, proba)
      y_pred — integer label array, shape (n_samples,)
      proba  — probability matrix, shape (n_samples, n_classes)

    HOW TO CHANGE: To apply a confidence threshold (e.g., only predict
    "Seizure" when confidence > 0.7), replace np.argmax with custom logic:
        max_prob = proba.max(axis=1)
        y_pred = np.where(max_prob >= 0.7, np.argmax(proba, axis=1), -1)
    where -1 means "uncertain / abstain".
    """
    proba  = hybrid_predict_proba(X, scaler, svm, rf)
    # Pick the class with the highest blended probability
    return np.argmax(proba, axis=1), proba


# =============================================================================
# STEP 4 — FULL METRICS COMPUTATION
#
# WHY each metric matters for a medical seizure detector:
#
#   Accuracy   — overall % correct. Can be misleading if classes are imbalanced
#                (e.g., 90% Normal data → 90% "accuracy" by predicting Normal
#                for everything). Always look at per-class metrics too.
#
#   Precision  — of all PREDICTED seizures, how many were real?
#                High precision → fewer false alarms (fewer unnecessary doctor calls).
#
#   Recall     — of all ACTUAL seizures, how many did we catch?
#                High recall → fewer missed seizures (critical for patient safety).
#                In medical devices, this is the most important metric.
#
#   F1 Score   — harmonic mean of precision and recall. Balanced summary.
#
#   Specificity — of all ACTUAL normal epochs, how many did we correctly label
#                as normal? (= True Negative Rate)
#
#   AUC         — Area Under the ROC Curve. 1.0 = perfect, 0.5 = random guess.
#                 Measures how well the model discriminates between classes.
#
# HOW TO CHANGE:
#   - To use macro-average instead of weighted: change average="weighted" to
#     average="macro" in precision_score / recall_score / f1_score.
#   - To add Cohen's Kappa: from sklearn.metrics import cohen_kappa_score
#     then kappa = cohen_kappa_score(y_true, y_pred).
# =============================================================================

def compute_metrics(y_true, y_pred, proba, class_names_list, label=""):
    """
    Compute and print a comprehensive set of classification metrics.

    ARGS:
      y_true          — true integer labels, shape (n_samples,).
      y_pred          — predicted integer labels, shape (n_samples,).
      proba           — class probability matrix, shape (n_samples, n_classes).
      class_names_list — list of class name strings (index = class integer).
      label           — string shown in the print header (e.g., "4-class").

    RETURNS: dict with all computed metrics, suitable for JSON serialisation.

    HOW TO CHANGE: add any new metric to the returned dict and it will
    automatically appear in the saved JSON and dashboard display.
    """
    print(f"\n  📊 {label} Metrics:")

    # Weighted-average scalar metrics (weight by class sample count)
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec   = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1v   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm    = confusion_matrix(y_true, y_pred)   # Square matrix (n_classes × n_classes)

    # Per-class versions of precision, recall, and F1 (one value per class)
    prec_per  = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per   = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per    = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Specificity per class: TN / (TN + FP)
    # The confusion matrix lets us derive TN, FP, FN for each class using
    # one-vs-rest arithmetic.
    spec_per = []
    for i in range(len(class_names_list)):
        tp = cm[i, i]                   # Diagonal element = correctly classified as class i
        fn = cm[i, :].sum() - tp        # Rest of row i = actual class i predicted as other
        fp = cm[:, i].sum() - tp        # Rest of column i = other classes predicted as i
        tn = cm.sum() - tp - fn - fp    # Everything else = correctly rejected
        spec_per.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    spec = float(np.mean(spec_per))     # Mean specificity across all classes

    # AUC — one-vs-rest strategy for multi-class problems
    # roc_auc_score requires probability scores (not hard predictions)
    try:
        n_cls = len(class_names_list)
        if n_cls == 2:
            # Binary case: use positive class probability directly
            auc = roc_auc_score(y_true, proba[:, 1])
        else:
            # Multi-class OvR: each class is treated as positive in turn
            auc = roc_auc_score(y_true, proba,
                                multi_class="ovr", average="weighted")
    except Exception:
        auc = 0.0   # Fallback if AUC computation fails (e.g., missing classes in test)

    # Print scalar summary
    print(f"    Accuracy   : {acc*100:.2f}%")
    print(f"    Precision  : {prec*100:.2f}%")
    print(f"    Recall     : {rec*100:.2f}%")
    print(f"    F1 Score   : {f1v*100:.2f}%")
    print(f"    Specificity: {spec*100:.2f}%")
    print(f"    AUC (OvR)  : {auc:.4f}")

    # Print per-class breakdown table
    print(f"\n    Per-class breakdown:")
    print(f"    {'Class':<22} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Spec':>7} {'Support':>8}")
    print(f"    {'-'*60}")
    for i, cname in enumerate(class_names_list):
        support = int(np.sum(y_true == i))   # How many true samples of this class
        print(f"    {cname:<22} {prec_per[i]*100:>6.1f}% "
              f"{rec_per[i]*100:>6.1f}% "
              f"{f1_per[i]*100:>6.1f}% "
              f"{spec_per[i]*100:>6.1f}% "
              f"{support:>8d}")

    # Print confusion matrix (rows = true class, columns = predicted class)
    # Off-diagonal values show what the model confused each class with
    print(f"\n    Confusion Matrix:")
    header = "    " + "".join(f"{n[:6]:>8}" for n in class_names_list)
    print(header)
    for i, row in enumerate(cm):
        row_lbl = f"    {class_names_list[i][:6]:<8}"
        print(row_lbl + "".join(f"{v:>8d}" for v in row))

    # Return all metrics as a dict for JSON serialisation
    return {
        "accuracy":        round(float(acc), 4),
        "precision":       round(float(prec), 4),
        "recall":          round(float(rec), 4),
        "f1":              round(float(f1v), 4),
        "specificity":     round(float(spec), 4),
        "auc":             round(float(auc), 4),
        "precision_per_class":   [round(float(v), 4) for v in prec_per],
        "recall_per_class":      [round(float(v), 4) for v in rec_per],
        "f1_per_class":          [round(float(v), 4) for v in f1_per],
        "specificity_per_class": [round(float(v), 4) for v in spec_per],
        "confusion_matrix":      cm.tolist(),   # tolist() makes it JSON-serialisable
        "class_names":           class_names_list,
    }


# =============================================================================
# STEP 5 — CROSS-VALIDATION
#
# WHY cross-validation:
#   The train/test split in the dataset is fixed. Cross-validation gives a
#   better estimate of generalisation by repeatedly re-splitting the training
#   set into 5 folds, training on 4 and evaluating on 1, then averaging.
#   A low standard deviation between folds means the model is stable and not
#   just lucky on one particular split.
#
# WHY StratifiedKFold (implicit in cross_val_score with balanced classes):
#   Each fold maintains the same class proportion as the full training set,
#   preventing situations where one fold has almost no seizure samples.
#
# WHY only on the Random Forest:
#   SVM with cross-validation would be very slow on large datasets. RF
#   is fast enough with n_jobs=-1 parallelism.
#
# HOW TO CHANGE:
#   - Change cv=5 to cv=10 for 10-fold CV (more folds = more stable estimate
#     but 2× longer runtime).
#   - Change scoring="f1_weighted" to "accuracy" or "roc_auc_ovr_weighted"
#     to optimise for a different metric.
# =============================================================================

def cross_validate(X_tr, y_tr, rf):
    """
    Run 5-fold stratified cross-validation on the Random Forest.

    ARGS:
      X_tr — training feature matrix, shape (n_train, n_features).
      y_tr — training label array, shape (n_train,).
      rf   — already-fitted RandomForestClassifier (re-fitted internally by
              cross_val_score on each fold — the fitted model is not modified).

    RETURNS: dict with per-fold F1 scores, mean, and standard deviation.

    HOW TO CHANGE: to also cross-validate the SVM, call cross_val_score a
    second time with the SVC (using X_tr scaled by the StandardScaler).
    """
    print("\n" + "="*60)
    print("🔁 5-FOLD CROSS-VALIDATION (Random Forest on train set)")
    print("="*60)
    # cross_val_score refits the model on each fold internally
    # n_jobs=-1 evaluates folds in parallel across CPU cores
    cv_scores = cross_val_score(rf, X_tr, y_tr, cv=5,
                                scoring="f1_weighted", n_jobs=-1)
    print(f"  F1 per fold : {[f'{s*100:.1f}%' for s in cv_scores]}")
    print(f"  Mean F1     : {cv_scores.mean()*100:.2f}% "
          f"(± {cv_scores.std()*100:.2f}%)")
    return {
        "cv_f1_scores": [round(float(s), 4) for s in cv_scores],
        "cv_f1_mean":   round(float(cv_scores.mean()), 4),
        "cv_f1_std":    round(float(cv_scores.std()), 4),
    }


# =============================================================================
# STEP 6 — SEIZURE-SPECIFIC METRICS
#
# WHY separate seizure-specific metrics:
#   The overall accuracy and F1 score average across all classes. But for a
#   medical device, missing a seizure (False Negative) is far more dangerous
#   than a false alarm (False Positive). This function collapses everything
#   to a binary problem: "Is this epoch a seizure or not?" and reports the
#   clinically critical numbers directly.
#
#   KEY METRICS:
#     Sensitivity (seizure recall) — % of actual seizures caught.
#       Target: as close to 100% as possible. Even 95% means 1 in 20 seizures
#       is missed.
#     False Alarm Rate — % of normal epochs incorrectly flagged as seizures.
#       A high rate leads to "alarm fatigue" where caregivers ignore alerts.
#     PPV (Positive Predictive Value) — if the system alerts, how often is it
#       right? Low PPV = too many unnecessary emergency calls.
#     NPV (Negative Predictive Value) — if the system says "Normal", how often
#       is it right? High NPV provides confidence in quiet periods.
#
# HOW TO CHANGE:
#   - To change which class labels count as "seizure": update the
#     seizure_labels argument when calling seizure_specific() in main.
#   - To add a cost-weighted metric (missing a seizure costs 10× more than
#     a false alarm): compute weighted_cost = 10*fn + fp and print it here.
# =============================================================================

def seizure_specific(y_true, y_pred, seizure_labels, label=""):
    """
    Compute binary seizure-detection metrics (seizure vs. non-seizure).

    WHY binary: regardless of which seizure sub-type the model predicted,
    the caregiver only needs to know "is this a seizure event?". So we
    collapse the multi-class predictions to a binary: 1 = any seizure type,
    0 = normal/pre-seizure.

    ARGS:
      y_true         — true integer labels, shape (n_samples,).
      y_pred         — predicted integer labels, shape (n_samples,).
      seizure_labels — list of integer class IDs that count as "seizure".
                       For 4-class: [1, 2, 3]. For 3-class: [2].
      label          — string for the print header.

    RETURNS: dict with tp, tn, fp, fn and the five derived rate metrics.
    """
    print(f"\n  🚨 Seizure-Specific Metrics ({label}):")

    # Convert multi-class labels to binary: 1 if the label is in seizure_labels
    y_true_bin = np.isin(y_true, seizure_labels).astype(int)
    y_pred_bin = np.isin(y_pred, seizure_labels).astype(int)

    # Compute confusion matrix counts for the binary case
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))  # Correctly detected seizures
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))  # Correctly cleared normals
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))  # False alarms
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))  # Missed seizures ← CRITICAL

    # Derived rate metrics — guard against division by zero with conditional expressions
    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0   # = seizure recall
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0   # = normal recall
    ppv          = tp / (tp + fp) if (tp + fp) > 0 else 0   # precision for seizure class
    npv          = tn / (tn + fn) if (tn + fn) > 0 else 0   # precision for normal class
    false_alarm  = fp / (tn + fp) if (tn + fp) > 0 else 0   # FPR = 1 - specificity

    print(f"    True Positives  (caught seizures) : {tp}")
    print(f"    False Negatives (missed seizures) : {fn}")
    print(f"    False Positives (false alarms)    : {fp}")
    print(f"    True Negatives  (correct normals) : {tn}")
    print(f"    Sensitivity (seizure recall)      : {sensitivity*100:.2f}%")
    print(f"    Specificity                       : {specificity*100:.2f}%")
    print(f"    Positive Predictive Value         : {ppv*100:.2f}%")
    print(f"    Negative Predictive Value         : {npv*100:.2f}%")
    print(f"    False Alarm Rate                  : {false_alarm*100:.2f}%")

    # Highlight missed seizures — these are the most dangerous failures
    if fn > 0:
        print(f"\n    ⚠️  {fn} seizure epoch(s) were MISSED by the model.")
    else:
        print(f"\n    ✅ All seizure epochs were detected!")

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity":      round(float(sensitivity), 4),
        "specificity":      round(float(specificity), 4),
        "ppv":              round(float(ppv), 4),
        "npv":              round(float(npv), 4),
        "false_alarm_rate": round(float(false_alarm), 4),
    }


# =============================================================================
# STEP 7 — SAVE METRICS TO JSON
#
# WHY two output files:
#   METRICS_JSON_OUT       — permanent record stored in the project /json folder.
#                            Use this to compare model versions over time.
#   METRICS_JSON_DASHBOARD — the file dashboard.py reads to populate the
#                            "Model Performance" panel. On the Pi this should
#                            be in /tmp/ or wherever dashboard.py expects it.
#
# WHY include both 3-class and 4-class results:
#   The dashboard shows the NeuroWatch 3-class metrics (the operational view).
#   The 4-class results are stored as extended metadata for research purposes.
#
# HOW TO CHANGE:
#   - To add a new metric to the JSON: add a key to the `output` dict.
#     dashboard.py will ignore keys it doesn't know, so this is safe.
#   - To change where dashboard.py reads from: update METRICS_JSON_DASHBOARD
#     at the top of this file to match the path dashboard.py expects.
# =============================================================================

def save_metrics(metrics_4cls, metrics_3cls, sz_metrics, cv_results):
    """
    Build and save the dashboard-compatible metrics JSON.

    ARGS:
      metrics_4cls — dict returned by compute_metrics() for the 4-class problem.
      metrics_3cls — dict returned by compute_metrics() for the 3-class problem.
      sz_metrics   — dict with keys "4class" and "3class" from seizure_specific().
      cv_results   — dict returned by cross_validate().

    RETURNS: the full output dict (also saved to disk as JSON).
    """
    # Use 3-class metrics as the top-level "headline" numbers (what the
    # dashboard displays prominently). 4-class goes under "metrics_4class".
    m3 = metrics_3cls
    output = {
        # Metadata
        "timestamp": datetime.now().isoformat(),
        "dataset":   "Mendeley Epileptic EEG (Nasreddine 2021)",
        "n_train":   7011,
        "n_test":    779,

        # -- 3-class top-level fields (required by dashboard.py) --
        # These key names MUST match what dashboard.py looks up in the JSON.
        # Do not rename them without updating dashboard.py too.
        "accuracy":         m3["accuracy"],
        "precision":        m3["precision"],
        "recall":           m3["recall"],
        "f1":               m3["f1"],
        "specificity":      m3["specificity"],
        # seizure_recall = recall for the "Seizure" class (index 2 in 3-class)
        "seizure_recall":   m3["recall_per_class"][2] if len(m3["recall_per_class"]) > 2 else 0,
        "confusion_matrix": m3["confusion_matrix"],

        # -- Extended metrics (not required by dashboard, but useful for analysis) --
        "auc":               m3["auc"],
        "metrics_3class":    m3,            # Full 3-class breakdown
        "metrics_4class":    metrics_4cls,  # Full 4-class breakdown
        "seizure_detection": sz_metrics,    # Binary seizure-detection stats
        "cross_validation":  cv_results,    # CV fold scores + mean/std
    }

    # Save to the project json/ folder (permanent record)
    with open(METRICS_JSON_OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  💾 Saved → {METRICS_JSON_OUT}")

    # Also save to the dashboard path (may fail on Windows if path is a Linux
    # /tmp/ path — that's fine, the Pi will get the file when you copy it over)
    try:
        with open(METRICS_JSON_DASHBOARD, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  💾 Saved → {METRICS_JSON_DASHBOARD}  (dashboard.py will auto-read this)")
    except Exception:
        print(f"  ℹ️  /tmp write skipped (Windows) — copy the output JSON to your Pi's /tmp/ when ready")

    return output


# =============================================================================
# SECTION 10 — MAIN ENTRY POINT
#
# WHY the `if __name__ == "__main__":` guard:
#   This block only runs when you execute the script directly
#   (`python neurowatch_metrics.py`). It does NOT run when the file is
#   imported by another script, preventing accidental long training runs.
#
# EXECUTION ORDER:
#   Each numbered step corresponds to a function defined above. Steps are
#   labelled with their step number so you can easily navigate here vs. there.
#
# HOW TO CHANGE:
#   - To skip cross-validation (saves ~2 minutes): comment out step 8 and
#     pass cv_results={} to save_metrics().
#   - To run on a custom dataset path without editing the file:
#       python neurowatch_metrics.py /path/to/your/Npy_files
#   - To save intermediate results (e.g., after feature extraction):
#       np.save("F_train.npy", F_train) before step 3, and add a load branch
#       that skips steps 2–3 if the cache file exists.
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠  NEUROWATCH METRICS PIPELINE")
    print(f"    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Allow an optional command-line override of the NPY folder path.
    # Example: python neurowatch_metrics.py /data/my_dataset/Npy_files
    if len(sys.argv) > 1:
        NPY_PATH = sys.argv[1]

    # ── STEP 1: Load .npy arrays ───────────────────────────────────────────────
    x_train, y_train, x_test, y_test = load_data(NPY_PATH)

    # ── STEP 2: Extract frequency-band power features ──────────────────────────
    # This is the slowest step (~5–15 minutes depending on dataset size and CPU).
    # If you want to speed this up, consider parallelising extract_all_features()
    # using joblib (see the docstring for that function above).
    print("\n" + "="*60)
    print("⚙️  FEATURE EXTRACTION")
    print("="*60)
    F_train = extract_all_features(x_train, "train")
    F_test  = extract_all_features(x_test,  "test")
    print(f"\n  Feature vector size: {F_train.shape[1]} "
          f"(19 channels × 4 bands)")

    # ── STEP 3: Train the SVM + Random Forest hybrid ───────────────────────────
    scaler, svm, rf = train_hybrid(F_train, y_train)

    # ── STEP 4: Run predictions on the held-out test set ──────────────────────
    # The test set is NEVER seen during training or cross-validation.
    print("\n" + "="*60)
    print("🔍 RUNNING PREDICTIONS")
    print("="*60)
    y_pred, proba = hybrid_predict(F_test, scaler, svm, rf)
    print(f"  Predictions done for {len(y_pred)} test epochs")

    # ── STEP 5: 4-class metrics (original dataset class labels) ───────────────
    # Use the original 4-class labels for a research-quality baseline.
    # This shows how well the model distinguishes all four seizure sub-types.
    print("\n" + "="*60)
    print("📊 4-CLASS METRICS (original dataset labels)")
    print("="*60)
    cls4_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
    metrics_4cls = compute_metrics(
        y_test, y_pred, proba, cls4_names, "4-class")

    # ── STEP 6: 3-class metrics (NeuroWatch operational mapping) ──────────────
    # Map both true and predicted labels to the 3-class system using MERGE_MAP.
    # Also rebuild the probability matrix: combine columns 1 and 2 (both
    # seizure types) into a single "Seizure" column (index 2 in 3-class).
    print("\n" + "="*60)
    print("📊 3-CLASS METRICS (NeuroWatch: Normal / Pre-Seizure / Seizure)")
    print("="*60)
    y_test_3  = np.array([MERGE_MAP[int(v)] for v in y_test])   # Map true labels
    y_pred_3  = np.array([MERGE_MAP[int(v)] for v in y_pred])   # Map predicted labels

    # Rebuild probability matrix for 3 classes by merging seizure columns:
    #   Column 0 → Normal (unchanged from 4-class column 0)
    #   Column 1 → Pre-Seizure (4-class column 3, Video-detected)
    #   Column 2 → Seizure (sum of 4-class columns 1 and 2: CPS + Electrographic)
    proba_3 = np.zeros((len(proba), 3))
    proba_3[:, 0] = proba[:, 0]               # Normal probability
    proba_3[:, 1] = proba[:, 3]               # Pre-Seizure probability (Video class)
    proba_3[:, 2] = proba[:, 1] + proba[:, 2] # Seizure probability (CPS + Electro)

    metrics_3cls = compute_metrics(
        y_test_3, y_pred_3, proba_3, MERGED_NAMES, "3-class NeuroWatch")

    # ── STEP 7: Seizure-specific binary detection metrics ─────────────────────
    print("\n" + "="*60)
    print("🚨 SEIZURE DETECTION PERFORMANCE")
    print("="*60)
    # 4-class: any of labels 1, 2, 3 counts as a seizure
    sz_4cls = seizure_specific(y_test, y_pred, [1, 2, 3], "4-class")
    # 3-class: label 2 is "Seizure" (label 1 = Pre-Seizure, not counted here)
    sz_3cls = seizure_specific(y_test_3, y_pred_3, [2], "3-class NeuroWatch")

    # ── STEP 8: Cross-validation on the training set ──────────────────────────
    # Uses the already-fitted RF object. cross_val_score re-fits internally
    # on each fold so the original rf model is not modified.
    cv_results = cross_validate(F_train, y_train, rf)

    # ── STEP 9: Save all metrics to JSON ──────────────────────────────────────
    print("\n" + "="*60)
    print("💾 SAVING RESULTS")
    print("="*60)
    final = save_metrics(
        metrics_4cls, metrics_3cls,
        {"4class": sz_4cls, "3class": sz_3cls},
        cv_results
    )

    # ── STEP 10: Print final summary to terminal ───────────────────────────────
    # One-screen overview of the most important numbers for a quick sanity check.
    # Seizure Recall and False Alarm Rate are the two numbers to watch most closely.
    print("\n" + "="*60)
    print("✅  FINAL SUMMARY (3-class NeuroWatch)")
    print("="*60)
    m  = metrics_3cls
    sz = sz_3cls
    print(f"  Accuracy        : {m['accuracy']*100:.2f}%")
    print(f"  F1 Score        : {m['f1']*100:.2f}%")
    print(f"  AUC             : {m['auc']:.4f}")
    print(f"  Seizure Recall  : {m['recall_per_class'][2]*100:.2f}%")
    print(f"  False Alarm Rate: {sz['false_alarm_rate']*100:.2f}%")
    print(f"  Missed Seizures : {sz['fn']}")
    print(f"  CV F1 Mean      : {cv_results['cv_f1_mean']*100:.2f}% "
          f"(± {cv_results['cv_f1_std']*100:.2f}%)")
    print("\n  Output file     :", METRICS_JSON_OUT)
    print("="*60)
