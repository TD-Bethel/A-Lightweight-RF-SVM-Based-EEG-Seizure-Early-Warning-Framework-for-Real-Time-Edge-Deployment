# =============================================================================
# neurowatch_metrics.py  —  NeuroWatch Dataset Metrics Pipeline
#
# Loads the pre-labeled Npy_files from the Mendeley Epileptic EEG Dataset,
# trains the same SVM + Random Forest hybrid used in main_pi_v1.py,
# and computes a full metrics report ready to merge into dashboard.py.
#
# Run:
#   python neurowatch_metrics.py
#
# Requirements:
#   pip install numpy scikit-learn scipy matplotlib seaborn
# =============================================================================

import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

# =============================================================================
# PATHS  —  confirmed from your folder screenshots
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPY_PATH = os.path.join(PROJECT_ROOT, "data", "Test dataset", "Npy_files")

# Output JSON — saved next to this script AND copied to /tmp for dashboard.py
METRICS_JSON_OUT     = os.path.join(PROJECT_ROOT, "json", "neurowatch_metrics_output.json")
METRICS_JSON_DASHBOARD = os.path.join(PROJECT_ROOT, "json", "neurowatch_metrics.json")

# =============================================================================
# CLASS DEFINITIONS
# 0 = Normal
# 1 = Complex Partial Seizure
# 2 = Electrographic Seizure
# 3 = Video-detected Seizure (no EEG change)
# =============================================================================
CLASS_NAMES = {
    0: "Normal",
    1: "Complex Partial",
    2: "Electrographic",
    3: "Video-detected"
}

# For NeuroWatch 3-class mapping (Normal / Pre-Seizure / Seizure)
# We'll compute both 4-class and merged 3-class metrics
MERGE_MAP = {
    0: 0,   # Normal      → Normal
    1: 2,   # CPS         → Seizure
    2: 2,   # Electro     → Seizure
    3: 1,   # Video       → Pre-Seizure (subtle, hard to detect)
}
MERGED_NAMES = ["Normal", "Pre-Seizure", "Seizure"]

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def load_data(npy_path):
    print("\n" + "="*60)
    print("📂 LOADING NPY DATA")
    print("="*60)

    required = ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy"]
    for f in required:
        full = os.path.join(npy_path, f)
        if not os.path.exists(full):
            print(f"❌ Missing: {full}")
            print("   Please check your NPY_PATH setting.")
            sys.exit(1)

    x_train = np.load(os.path.join(npy_path, "x_train.npy"))
    y_train = np.load(os.path.join(npy_path, "y_train.npy"))
    x_test  = np.load(os.path.join(npy_path, "x_test.npy"))
    y_test  = np.load(os.path.join(npy_path, "y_test.npy"))

    print(f"  x_train shape : {x_train.shape}  → {x_train.shape[0]} epochs")
    print(f"  y_train shape : {y_train.shape}")
    print(f"  x_test  shape : {x_test.shape}   → {x_test.shape[0]} epochs")
    print(f"  y_test  shape : {y_test.shape}")

    # Class distribution
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
# Same approach as main_pi_v1.py: bandpass + Welch PSD → delta/theta/alpha/beta
# Extended here to use ALL 19 channels (main_pi uses single channel)
# =============================================================================

def extract_band_powers(epoch, fs=500):
    """
    epoch: shape (n_channels, n_timepoints)
    Returns feature vector: 4 bands × n_channels = 76 features for 19ch
    """
    features = []
    for ch in range(epoch.shape[0]):
        sig = epoch[ch].astype(float)
        f, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)))
        delta = np.mean(psd[(f >= 1)  & (f <= 4)])
        theta = np.mean(psd[(f >= 4)  & (f <= 8)])
        alpha = np.mean(psd[(f >= 8)  & (f <= 13)])
        beta  = np.mean(psd[(f >= 13) & (f <= 30)])
        features.extend([delta, theta, alpha, beta])
    return np.array(features)


def extract_all_features(X, label=""):
    """Extract features from all epochs. X shape: (n_epochs, n_ch, n_tp)"""
    import time
    print(f"\n  ⚙️  Extracting features from {label} ({X.shape[0]} epochs × "
          f"{X.shape[1]} channels)...")
    feats = []
    n = X.shape[0]
    t0 = time.time()
    for i, epoch in enumerate(X):
        feats.append(extract_band_powers(epoch))
        if (i + 1) % 500 == 0:
            elapsed   = time.time() - t0
            rate      = (i + 1) / elapsed
            remaining = (n - i - 1) / rate
            print(f"    {i+1}/{n} ({(i+1)/n*100:.0f}%)  ~{remaining:.0f}s remaining")
    total = time.time() - t0
    print(f"  ✅ Done in {total:.1f}s  →  feature matrix: {len(feats)} × {len(feats[0])}")
    return np.array(feats)


# =============================================================================
# STEP 3 — TRAIN HYBRID MODEL (same as main_pi_v1.py)
# =============================================================================

def train_hybrid(X_tr, y_tr):
    print("\n" + "="*60)
    print("🎓 TRAINING HYBRID SVM + RANDOM FOREST")
    print("="*60)

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)

    print("  Training SVM (RBF kernel)...")
    svm = SVC(kernel="rbf", C=2, gamma="scale",
              probability=True, class_weight="balanced")
    svm.fit(X_tr_sc, y_tr)
    print("  ✅ SVM trained")

    print("  Training Random Forest (100 trees)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15,
                                class_weight="balanced", random_state=42,
                                n_jobs=-1)
    rf.fit(X_tr, y_tr)
    print("  ✅ Random Forest trained")

    return scaler, svm, rf


def hybrid_predict_proba(X, scaler, svm, rf):
    """Returns blended probability matrix (n_samples, n_classes)."""
    X_sc = scaler.transform(X)
    return 0.4 * svm.predict_proba(X_sc) + 0.6 * rf.predict_proba(X)


def hybrid_predict(X, scaler, svm, rf):
    proba = hybrid_predict_proba(X, scaler, svm, rf)
    return np.argmax(proba, axis=1), proba


# =============================================================================
# STEP 4 — FULL METRICS COMPUTATION
# =============================================================================

def compute_metrics(y_true, y_pred, proba, class_names_list, label=""):
    print(f"\n  📊 {label} Metrics:")

    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec   = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1v   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm    = confusion_matrix(y_true, y_pred)

    # Per-class metrics
    prec_per  = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per   = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per    = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Specificity per class (TN / (TN + FP))
    spec_per = []
    for i in range(len(class_names_list)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec_per.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    spec = float(np.mean(spec_per))

    # AUC (one-vs-rest)
    try:
        n_cls = len(class_names_list)
        if n_cls == 2:
            auc = roc_auc_score(y_true, proba[:, 1])
        else:
            auc = roc_auc_score(y_true, proba,
                                multi_class="ovr", average="weighted")
    except Exception:
        auc = 0.0

    print(f"    Accuracy   : {acc*100:.2f}%")
    print(f"    Precision  : {prec*100:.2f}%")
    print(f"    Recall     : {rec*100:.2f}%")
    print(f"    F1 Score   : {f1v*100:.2f}%")
    print(f"    Specificity: {spec*100:.2f}%")
    print(f"    AUC (OvR)  : {auc:.4f}")

    print(f"\n    Per-class breakdown:")
    print(f"    {'Class':<22} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Spec':>7} {'Support':>8}")
    print(f"    {'-'*60}")
    for i, cname in enumerate(class_names_list):
        support = int(np.sum(y_true == i))
        print(f"    {cname:<22} {prec_per[i]*100:>6.1f}% "
              f"{rec_per[i]*100:>6.1f}% "
              f"{f1_per[i]*100:>6.1f}% "
              f"{spec_per[i]*100:>6.1f}% "
              f"{support:>8d}")

    print(f"\n    Confusion Matrix:")
    header = "    " + "".join(f"{n[:6]:>8}" for n in class_names_list)
    print(header)
    for i, row in enumerate(cm):
        row_lbl = f"    {class_names_list[i][:6]:<8}"
        print(row_lbl + "".join(f"{v:>8d}" for v in row))

    return {
        "accuracy":        round(float(acc), 4),
        "precision":       round(float(prec), 4),
        "recall":          round(float(rec), 4),
        "f1":              round(float(f1v), 4),
        "specificity":     round(float(spec), 4),
        "auc":             round(float(auc), 4),
        "precision_per_class": [round(float(v), 4) for v in prec_per],
        "recall_per_class":    [round(float(v), 4) for v in rec_per],
        "f1_per_class":        [round(float(v), 4) for v in f1_per],
        "specificity_per_class":[round(float(v), 4) for v in spec_per],
        "confusion_matrix":    cm.tolist(),
        "class_names":         class_names_list,
    }


# =============================================================================
# STEP 5 — CROSS-VALIDATION ON TRAINING SET
# =============================================================================

def cross_validate(X_tr, y_tr, rf):
    print("\n" + "="*60)
    print("🔁 5-FOLD CROSS-VALIDATION (Random Forest on train set)")
    print("="*60)
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
# Key for a medical device: how well do we catch seizures?
# =============================================================================

def seizure_specific(y_true, y_pred, seizure_labels, label=""):
    """
    Compute seizure detection metrics:
    - Sensitivity (seizure recall): how many seizures did we catch?
    - False Alarm Rate: how often did we wrongly flag normal as seizure?
    """
    print(f"\n  🚨 Seizure-Specific Metrics ({label}):")

    # Binary: is it ANY type of seizure?
    y_true_bin = np.isin(y_true, seizure_labels).astype(int)
    y_pred_bin = np.isin(y_pred, seizure_labels).astype(int)

    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))

    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv          = tp / (tp + fp) if (tp + fp) > 0 else 0  # precision
    npv          = tn / (tn + fn) if (tn + fn) > 0 else 0
    false_alarm  = fp / (tn + fp) if (tn + fp) > 0 else 0

    print(f"    True Positives  (caught seizures) : {tp}")
    print(f"    False Negatives (missed seizures) : {fn}")
    print(f"    False Positives (false alarms)    : {fp}")
    print(f"    True Negatives  (correct normals) : {tn}")
    print(f"    Sensitivity (seizure recall)      : {sensitivity*100:.2f}%")
    print(f"    Specificity                       : {specificity*100:.2f}%")
    print(f"    Positive Predictive Value         : {ppv*100:.2f}%")
    print(f"    Negative Predictive Value         : {npv*100:.2f}%")
    print(f"    False Alarm Rate                  : {false_alarm*100:.2f}%")

    if fn > 0:
        print(f"\n    ⚠️  {fn} seizure epoch(s) were MISSED by the model.")
    else:
        print(f"\n    ✅ All seizure epochs were detected!")

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity":  round(float(sensitivity), 4),
        "specificity":  round(float(specificity), 4),
        "ppv":          round(float(ppv), 4),
        "npv":          round(float(npv), 4),
        "false_alarm_rate": round(float(false_alarm), 4),
    }


# =============================================================================
# STEP 7 — SAVE METRICS JSON (dashboard-compatible)
# =============================================================================

def save_metrics(metrics_4cls, metrics_3cls, sz_metrics, cv_results):
    """
    Saves a JSON file compatible with both the existing dashboard.py
    METRICS_JSON format AND the extended NeuroWatch metrics.
    """
    # Core dashboard-compatible fields (matches existing dashboard.py format)
    m3 = metrics_3cls
    output = {
        "timestamp":        datetime.now().isoformat(),
        "dataset":          "Mendeley Epileptic EEG (Nasreddine 2021)",
        "n_train":          7011,
        "n_test":           779,

        # — 3-class metrics (Normal / Pre-Seizure / Seizure) —
        # These match the existing METRICS_JSON keys in dashboard.py
        "accuracy":         m3["accuracy"],
        "precision":        m3["precision"],
        "recall":           m3["recall"],
        "f1":               m3["f1"],
        "specificity":      m3["specificity"],
        "seizure_recall":   m3["recall_per_class"][2] if len(m3["recall_per_class"]) > 2 else 0,
        "confusion_matrix": m3["confusion_matrix"],

        # — Extended metrics —
        "auc":              m3["auc"],
        "metrics_3class":   m3,
        "metrics_4class":   metrics_4cls,
        "seizure_detection": sz_metrics,
        "cross_validation": cv_results,
    }

    # Save to project folder
    with open(METRICS_JSON_OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  💾 Saved → {METRICS_JSON_OUT}")

    # Also try to save to /tmp for dashboard.py (Linux/Pi only, skip on Windows)
    try:
        with open(METRICS_JSON_DASHBOARD, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  💾 Saved → {METRICS_JSON_DASHBOARD}  (dashboard.py will auto-read this)")
    except Exception:
        print(f"  ℹ️  /tmp write skipped (Windows) — copy the output JSON to your Pi's /tmp/ when ready")

    return output


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠  NEUROWATCH METRICS PIPELINE")
    print(f"    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Allow path override from command line
    if len(sys.argv) > 1:
        NPY_PATH = sys.argv[1]

    # ── 1. Load ──────────────────────────────────────────────────────────────
    x_train, y_train, x_test, y_test = load_data(NPY_PATH)

    # ── 2. Feature extraction ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("⚙️  FEATURE EXTRACTION")
    print("="*60)
    F_train = extract_all_features(x_train, "train")
    F_test  = extract_all_features(x_test,  "test")
    print(f"\n  Feature vector size: {F_train.shape[1]} "
          f"(19 channels × 4 bands)")

    # ── 3. Train ──────────────────────────────────────────────────────────────
    scaler, svm, rf = train_hybrid(F_train, y_train)

    # ── 4. Predict ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("🔍 RUNNING PREDICTIONS")
    print("="*60)
    y_pred, proba = hybrid_predict(F_test, scaler, svm, rf)
    print(f"  Predictions done for {len(y_pred)} test epochs")

    # ── 5. 4-class metrics (original labels) ──────────────────────────────────
    print("\n" + "="*60)
    print("📊 4-CLASS METRICS (original dataset labels)")
    print("="*60)
    cls4_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
    metrics_4cls = compute_metrics(
        y_test, y_pred, proba, cls4_names, "4-class")

    # ── 6. 3-class metrics (NeuroWatch mapping) ───────────────────────────────
    print("\n" + "="*60)
    print("📊 3-CLASS METRICS (NeuroWatch: Normal / Pre-Seizure / Seizure)")
    print("="*60)
    y_test_3  = np.array([MERGE_MAP[int(v)] for v in y_test])
    y_pred_3  = np.array([MERGE_MAP[int(v)] for v in y_pred])

    # Rebuild proba for 3 classes by summing merged classes
    proba_3 = np.zeros((len(proba), 3))
    proba_3[:, 0] = proba[:, 0]                      # Normal
    proba_3[:, 1] = proba[:, 3]                       # Pre-Seizure (video)
    proba_3[:, 2] = proba[:, 1] + proba[:, 2]         # Seizure (CPS + Electro)

    metrics_3cls = compute_metrics(
        y_test_3, y_pred_3, proba_3, MERGED_NAMES, "3-class NeuroWatch")

    # ── 7. Seizure-specific metrics ───────────────────────────────────────────
    print("\n" + "="*60)
    print("🚨 SEIZURE DETECTION PERFORMANCE")
    print("="*60)
    # 4-class: seizure labels are 1, 2, 3
    sz_4cls = seizure_specific(y_test, y_pred, [1, 2, 3], "4-class")
    # 3-class: seizure label is 2
    sz_3cls = seizure_specific(y_test_3, y_pred_3, [2], "3-class NeuroWatch")

    # ── 8. Cross-validation ───────────────────────────────────────────────────
    cv_results = cross_validate(F_train, y_train, rf)

    # ── 9. Save ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("💾 SAVING RESULTS")
    print("="*60)
    final = save_metrics(
        metrics_4cls, metrics_3cls,
        {"4class": sz_4cls, "3class": sz_3cls},
        cv_results
    )

    # ── 10. Final summary ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("✅  FINAL SUMMARY (3-class NeuroWatch)")
    print("="*60)
    m = metrics_3cls
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