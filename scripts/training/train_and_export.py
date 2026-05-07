# =============================================================================
# train_and_export.py  —  NeuroWatch Model Training & Export (Bonn Dataset)
#
# WHAT THIS DOES:
#   Loads the Bonn University EEG dataset, trains a hybrid SVM + Random Forest
#   model, evaluates it on all three splits, and saves the trained model files
#   ready for use by main_pi_bios_v15.py.
#
# DATASET:
#   Bonn University EEG Dataset
#   5 folders: F/ N/ O/ S/ Z/ — each containing 100 single-channel .txt files
#   Total: 500 files
#
# LABEL MAPPING:
#   O = Eyes open,  healthy    →  0 = Normal
#   N = Eyes closed, healthy   →  0 = Normal
#   F = Inter-ictal, epil.     →  1 = Pre-Seizure
#   S = Inter-ictal, non-epil. →  1 = Pre-Seizure
#   Z = Ictal (seizure)        →  2 = Seizure
#
# SPLIT STRATEGY — Stratified 80 / 10 / 10:
#   Train      : 80%  →  400 files  — model learns from this
#   Validation : 10%  →   50 files  — used to detect overfitting
#   Test       : 10%  →   50 files  — final untouched evaluation
#   Stratified: every class keeps the same proportions in all 3 sets.
#
# FEATURES:
#   4 band power values per file: delta / theta / alpha / beta
#   Option A normalisation: each band divided by total power sum
#   → scale-invariant, works across any EEG equipment
#
# MODEL:
#   Hybrid: 40% SVM (RBF) + 60% Random Forest
#   Same blend used by main_pi_bios_v15.py at runtime
#   StandardScaler fitted on training set ONLY — no leakage into val/test
#
# OUTPUT (saved to models/MODELS_V1/):
#   scaler.pkl
#   svm_model.pkl
#   rf_model.pkl
#   training_history.json   — appended every run, tracks accuracy over time
#
# USAGE:
#   python train_and_export.py
#   python train_and_export.py --data "C:\...\Bonn Univeristy Dataset"
#   python train_and_export.py --out  "C:\...\models\MODELS_V1"
#   python train_and_export.py --no-plots
#   python train_and_export.py --reset    (clears history, trains fresh)
# =============================================================================

import os, sys, time, json, glob, argparse, warnings
from datetime import datetime

import joblib
import numpy as np
from scipy.signal import butter, filtfilt, welch, resample
from scipy.stats import skew, kurtosis as scipy_kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠  imbalanced-learn not found — install with: pip install imbalanced-learn")

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS  —  relative to wherever this file lives (project root)
# =============================================================================
# __file__ is scripts/train_and_export.py
# dirname once  → scripts/
# dirname twice → NeuroWatch_Project/  (project root where data/ and models/ live)
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(BASE_DIR, "data", "Bonn Univeristy Dataset")
DEFAULT_OUT  = os.path.join(BASE_DIR, "models", "MODELS_V1")

# =============================================================================
# CONSTANTS
# =============================================================================
TARGET_FS  = 128
FS_BONN    = 173.61
N_ORDER    = 4
F_LOW      = 0.5
F_HIGH     = 45.0
NPERSEG    = 128
N_BANDS    = 5       # delta, theta, alpha, beta, gamma
N_STATS    = 8       # activity, mobility, complexity, skewness, kurtosis, rms, zcr, spectral_entropy
N_FEATURES = N_BANDS + N_STATS   # 13 per file (single channel)

TRAIN_SIZE = 0.80
VAL_SIZE   = 0.10
TEST_SIZE  = 0.10
SEED       = 42

CLASS_MAP  = {"O": 0, "N": 0, "F": 1, "S": 1, "Z": 2}
FOLDER_DESC = {
    "O": "Eyes open — healthy",
    "N": "Eyes closed — healthy",
    "F": "Inter-ictal — epileptic focus",
    "S": "Inter-ictal — non-epileptic focus",
    "Z": "Ictal (active seizure)",
}
CLASSES = ["Normal", "Pre-Seizure", "Seizure"]
COLORS  = ["#00CC66", "#FFD633", "#FF3333"]

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
ap = argparse.ArgumentParser(description="NeuroWatch — Bonn Training & Export")
ap.add_argument("--data",     default=DEFAULT_DATA,
                help="Path to Bonn dataset root (contains F/ N/ O/ S/ Z/)")
ap.add_argument("--out",      default=DEFAULT_OUT,
                help="Where to save trained model .pkl files")
ap.add_argument("--no-plots", action="store_true",
                help="Skip all matplotlib plots")
ap.add_argument("--reset",    action="store_true",
                help="Clear training history and train from scratch")
args = ap.parse_args()

HISTORY_LOG = os.path.join(args.out, "training_history.json")
PLOT        = not args.no_plots

# =============================================================================
# MATPLOTLIB SETUP
# =============================================================================
if PLOT:
    try:
        import matplotlib; matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import seaborn as sns
        plt.rcParams.update({
            "figure.facecolor": "#1a1a2e", "axes.facecolor":  "#16213e",
            "axes.edgecolor":   "#444",    "axes.labelcolor": "#ccc",
            "xtick.color":      "#aaa",    "ytick.color":     "#aaa",
            "text.color":       "#eee",    "grid.color":      "#2a2a4a",
            "grid.linestyle":   "--",      "grid.alpha":      0.5,
        })
        PLOT_OK = True
    except Exception as e:
        print(f"  ⚠  Plots unavailable: {e}")
        PLOT_OK = False
else:
    PLOT_OK = False

# =============================================================================
# ── STEP 1: SIGNAL PROCESSING & FEATURE EXTRACTION
# =============================================================================
def _bandpass(sig, fs):
    nyq  = 0.5 * fs
    b, a = butter(N_ORDER, [F_LOW / nyq, F_HIGH / nyq], btype="band")
    return filtfilt(b, a, sig)

def _band_power(sig, fs=TARGET_FS):
    """Extract [delta, theta, alpha, beta, gamma] band power from a 1D signal."""
    filt   = _bandpass(sig, fs)
    f, psd = welch(filt, fs=fs, nperseg=NPERSEG)
    return np.array([
        np.mean(psd[(f >= 1)  & (f <= 4)]),
        np.mean(psd[(f >= 4)  & (f <= 8)]),
        np.mean(psd[(f >= 8)  & (f <= 13)]),
        np.mean(psd[(f >= 13) & (f <= 30)]),
        np.mean(psd[(f >= 30) & (f <= 45)]),
    ]), psd, f

def _normalise(feat_vec):
    total = feat_vec.sum()
    return feat_vec / total if total > 0 else feat_vec

def _hjorth(sig):
    d1, d2   = np.diff(sig), np.diff(np.diff(sig))
    var_x    = np.var(sig) + 1e-10
    var_d1   = np.var(d1)  + 1e-10
    var_d2   = np.var(d2)  + 1e-10
    activity   = float(var_x)
    mobility   = float(np.sqrt(var_d1 / var_x))
    complexity = float(np.sqrt(var_d2 / var_d1) / (mobility + 1e-10))
    return activity, mobility, complexity

def _spectral_entropy(psd):
    p = psd / (psd.sum() + 1e-10)
    return float(-np.sum(p * np.log2(p + 1e-10)) / np.log2(len(p) + 1))

def extract_features(raw_signal):
    """
    Full pipeline for one Bonn .txt file → 13 features:
      Band powers (5): delta, theta, alpha, beta, gamma  — Option A normalised
      Statistical (8): activity, mobility, complexity,
                       skewness, kurtosis, rms, zcr, spectral_entropy
    Returns (13,) float32 array, or None on failure.
    """
    try:
        n_new  = int(len(raw_signal) * TARGET_FS / FS_BONN)
        sig_rs = resample(raw_signal, n_new)

        bp, psd, _ = _band_power(sig_rs)
        bp_norm    = _normalise(bp)

        act, mob, comp = _hjorth(sig_rs)
        sk   = float(skew(sig_rs))
        kurt = float(scipy_kurtosis(sig_rs))
        rms  = float(np.sqrt(np.mean(sig_rs ** 2)))
        zcr  = float(((sig_rs[:-1] * sig_rs[1:]) < 0).sum() / len(sig_rs))
        sent = _spectral_entropy(psd)

        feats = np.concatenate([bp_norm, [act, mob, comp, sk, kurt, rms, zcr, sent]])
        return feats.astype(np.float32)
    except Exception as e:
        print(f"  ⚠  Feature extraction failed: {e}")
        return None

# =============================================================================
# ── STEP 2: LOAD ALL .TXT FILES & BUILD FEATURE MATRIX
# =============================================================================
def load_dataset(data_dir):
    print(f"\n{'─'*65}")
    print(f"  STEP 1: Loading Bonn dataset from:")
    print(f"  {data_dir}")
    print(f"{'─'*65}")

    if not os.path.isdir(data_dir):
        print(f"\n  ❌ Folder not found.")
        print(f"     Update DEFAULT_DATA at the top of this file, or use --data flag.")
        sys.exit(1)

    all_feats, all_labels = [], []

    print(f"\n  Scanning subfolders...")
    for folder in sorted(CLASS_MAP.keys()):
        lbl         = CLASS_MAP[folder]
        folder_path = os.path.join(data_dir, folder)

        if not os.path.isdir(folder_path):
            print(f"  ⚠  {folder}/ not found — skipping")
            continue

        txt_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
        if not txt_files:
            print(f"  ⚠  No .txt files in {folder}/ — skipping")
            continue

        loaded = 0
        for fpath in txt_files:
            try:
                raw  = np.loadtxt(fpath).squeeze()
                feat = extract_features(raw)
                if feat is not None:
                    all_feats.append(feat)
                    all_labels.append(lbl)
                    loaded += 1
            except Exception as e:
                print(f"  ⚠  {os.path.basename(fpath)}: {e}")

        print(f"  {folder}/  ({FOLDER_DESC[folder]:<40})  "
              f"{loaded} files  →  class {lbl} ({CLASSES[lbl]})")

    if not all_feats:
        print(f"\n  ❌ No files loaded successfully. Check the dataset path.")
        sys.exit(1)

    X = np.array(all_feats,  dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    print(f"\n  Total loaded : {len(X)} files  |  features per file: {X.shape[1]}")
    print(f"\n  Class distribution:")
    for lbl in sorted(np.unique(y)):
        cnt = int(np.sum(y == lbl))
        bar = "█" * int(cnt / len(y) * 40)
        print(f"    {lbl} = {CLASSES[lbl]:<14}: {cnt:4d}  ({cnt/len(y)*100:.0f}%)  {bar}")

    # Verify Option A normalisation
    sums = X.sum(axis=1)
    ok   = "✅" if abs(sums.mean() - 1.0) < 0.01 else "⚠"
    print(f"\n  Option A check — feature sums (should all be 1.0): "
          f"min={sums.min():.4f}  max={sums.max():.4f}  {ok}")

    # ── Stratified 80/10/10 split ─────────────────────────────────────────────
    print(f"\n  Applying stratified 80 / 10 / 10 split  (seed={SEED})...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(VAL_SIZE + TEST_SIZE),
        random_state=SEED,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=SEED,
        stratify=y_temp,
    )

    print(f"  Train : {len(X_train):4d} samples  (80%)")
    print(f"  Val   : {len(X_val):4d} samples  (10%)")
    print(f"  Test  : {len(X_test):4d} samples  (10%)")

    # Verify stratification
    print(f"\n  Stratification check — proportions should match:")
    print(f"  {'Class':<14} {'All':>7} {'Train':>7} {'Val':>7} {'Test':>7}")
    for lbl in sorted(np.unique(y)):
        ap = np.mean(y       == lbl) * 100
        tp = np.mean(y_train == lbl) * 100
        vp = np.mean(y_val   == lbl) * 100
        ep = np.mean(y_test  == lbl) * 100
        ok = "✅" if max(abs(tp-vp), abs(tp-ep)) < 2.5 else "⚠"
        print(f"  {ok} {CLASSES[lbl]:<12} "
              f"{ap:>6.1f}%  {tp:>6.1f}%  {vp:>6.1f}%  {ep:>6.1f}%")

    return X_train, y_train, X_val, y_val, X_test, y_test

# =============================================================================
# ── STEP 3: TRAIN
# =============================================================================
def train(X_train, y_train, X_val, y_val):
    print(f"\n{'─'*65}")
    print(f"  STEP 2: Training  (StandardScaler + SMOTE + SVM + Random Forest)")
    print(f"{'─'*65}")

    # Scaler — fitted on train ONLY
    print(f"\n  Fitting StandardScaler on {len(X_train)} training samples...")
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    # SMOTE — balance classes on scaled training data
    if SMOTE_AVAILABLE:
        print(f"\n  Applying SMOTE to balance classes...")
        uniq, cnts = np.unique(y_train, return_counts=True)
        print(f"  Before: { {CLASSES[u]: int(c) for u, c in zip(uniq, cnts)} }")
        sm = SMOTE(random_state=SEED, k_neighbors=3)
        X_tr_sc, y_train_bal = sm.fit_resample(X_tr_sc, y_train)
        uniq2, cnts2 = np.unique(y_train_bal, return_counts=True)
        print(f"  After : { {CLASSES[u]: int(c) for u, c in zip(uniq2, cnts2)} }")
        y_train  = y_train_bal
        X_train  = scaler.inverse_transform(X_tr_sc)
    else:
        print(f"  ⚠  SMOTE skipped — install imbalanced-learn")

    # SVM
    print(f"\n  Training SVM (RBF, C=50, balanced)...")
    t0  = time.time()
    svm = SVC(kernel="rbf", C=50, gamma="scale",
              probability=True, class_weight="balanced", random_state=SEED)
    svm.fit(X_tr_sc, y_train)
    svm_val = accuracy_score(y_val, svm.predict(X_val_sc))
    print(f"  Done in {time.time()-t0:.2f}s  |  "
          f"Support vectors: {sum(svm.n_support_)}  |  "
          f"Val accuracy: {svm_val*100:.1f}%")

    # Random Forest
    print(f"\n  Training Random Forest (500 trees, max_depth=15, balanced)...")
    t0 = time.time()
    rf  = RandomForestClassifier(n_estimators=500, max_depth=15,
                                 min_samples_leaf=2, max_features="sqrt",
                                 class_weight="balanced",
                                 random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_val = accuracy_score(y_val, rf.predict(X_val))
    print(f"  Done in {time.time()-t0:.2f}s  |  Val accuracy: {rf_val*100:.1f}%")

    # Auto-tune ensemble weights on validation set
    print(f"\n  Searching for best SVM/RF blend on validation set...")
    p_svm_val = svm.predict_proba(X_val_sc)
    p_rf_val  = rf.predict_proba(X_val)
    best_w, best_acc = 0.4, 0.0
    for w in np.arange(0.05, 1.0, 0.05):
        acc = accuracy_score(y_val, np.argmax(w * p_svm_val + (1-w) * p_rf_val, axis=1))
        if acc > best_acc:
            best_acc, best_w = acc, w
    SVM_W, RF_W = round(best_w, 2), round(1.0 - best_w, 2)
    print(f"  Best blend → SVM {SVM_W:.0%} + RF {RF_W:.0%}  "
          f"val accuracy: {best_acc*100:.1f}%")

    # Feature importances
    feat_names = ["delta","theta","alpha","beta","gamma",
                  "activity","mobility","complexity",
                  "skewness","kurtosis","rms","zcr","spectral_entropy"]
    print(f"\n  Feature importances (RF):")
    for i, (name, imp) in enumerate(zip(feat_names, rf.feature_importances_)):
        bar = "█" * int(imp * 80)
        print(f"    {i:2} = {name:<18}  {imp:.4f}  {bar}")

    return scaler, svm, rf, SVM_W, RF_W

# =============================================================================
# ── STEP 4: EVALUATE on all 3 splits
# =============================================================================
def evaluate_split(X, y_true, scaler, svm, rf, label, svm_w=0.4, rf_w=0.6):
    X_sc     = scaler.transform(X)
    p_svm    = svm.predict_proba(X_sc)
    p_rf     = rf.predict_proba(X)
    combined = svm_w * p_svm + rf_w * p_rf
    y_pred   = np.argmax(combined, axis=1)
    confs    = combined

    present = sorted(np.unique(np.concatenate([y_true, y_pred])))
    lnames  = [CLASSES[i] for i in present]

    acc    = accuracy_score(y_true, y_pred)
    prec   = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1     = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm     = confusion_matrix(y_true, y_pred)
    sz_rec = (recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)
              if 2 in y_true else 0.0)
    tn   = cm[0,0] if 0 in y_true else 0
    fp   = cm[0,1:].sum() if 0 in y_true else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

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

    # Print
    w = 14
    print(f"\n  {'─'*60}")
    print(f"  {label}")
    print(f"  {'─'*60}")
    print(f"  Samples        : {len(y_true)}")
    print(f"  Accuracy       : {acc*100:.1f}%  "
          f"{'✅' if acc>=0.90 else '⚠️ ' if acc>=0.75 else '❌'}")
    print(f"  Precision (w)  : {prec*100:.1f}%")
    print(f"  Recall (w)     : {rec*100:.1f}%")
    print(f"  F1 (w)         : {f1*100:.1f}%")
    print(f"  Specificity    : {spec*100:.1f}%")
    print(f"  Seizure Recall : {sz_rec*100:.1f}%  ← most clinically important")
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

    return dict(label=label, acc=acc, prec=prec, rec=rec, f1=f1,
                spec=spec, sz_rec=sz_rec, cm=cm, y_true=y_true,
                y_pred=y_pred, confs=confs, present=present,
                lnames=lnames, per_class=per_class)

# =============================================================================
# ── STEP 5: SAVE MODELS + LOG
# =============================================================================
def save_models(scaler, svm, rf, train_m, val_m, test_m, out_dir, svm_w=0.4, rf_w=0.6):
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(svm,    os.path.join(out_dir, "svm_model.pkl"))
    joblib.dump(rf,     os.path.join(out_dir, "rf_model.pkl"))
    joblib.dump({"svm_w": svm_w, "rf_w": rf_w},
                os.path.join(out_dir, "ensemble_weights.pkl"))

    # Build log entry
    entry = {
        "timestamp"     : datetime.now().isoformat(),
        "dataset"       : args.data,
        "split"         : "stratified 80/10/10",
        "n_features"    : N_FEATURES,
        "normalisation" : "Option A — relative band power proportions + statistical features",
        "model"         : f"Hybrid SVM ({svm_w:.0%}) + RF ({rf_w:.0%}) — auto-tuned",
        "training": {
            "samples"       : int(len(train_m["y_true"])),
            "accuracy"      : round(train_m["acc"],    4),
            "f1"            : round(train_m["f1"],     4),
            "seizure_recall": round(train_m["sz_rec"], 4),
            "per_class"     : train_m["per_class"],
        },
        "validation": {
            "samples"        : int(len(val_m["y_true"])),
            "accuracy"       : round(val_m["acc"],    4),
            "f1"             : round(val_m["f1"],     4),
            "seizure_recall" : round(val_m["sz_rec"], 4),
            "overfitting_gap": round(train_m["acc"] - val_m["acc"], 4),
            "per_class"      : val_m["per_class"],
        },
        "test": {
            "samples"       : int(len(test_m["y_true"])),
            "accuracy"      : round(test_m["acc"],    4),
            "f1"            : round(test_m["f1"],     4),
            "seizure_recall": round(test_m["sz_rec"], 4),
            "per_class"     : test_m["per_class"],
        },
        "confusion_matrices": {
            "training"  : train_m["cm"].tolist(),
            "validation": val_m["cm"].tolist(),
            "test"      : test_m["cm"].tolist(),
        }
    }

    # Append to history log
    if args.reset and os.path.exists(HISTORY_LOG):
        os.remove(HISTORY_LOG)
        print(f"  History cleared (--reset)")

    history = []
    if os.path.exists(HISTORY_LOG):
        try:
            with open(HISTORY_LOG) as f:
                history = json.load(f)
        except Exception:
            pass

    history.append(entry)
    with open(HISTORY_LOG, "w") as f:
        json.dump(history, f, indent=2)

    # Compare vs previous run
    if len(history) >= 2:
        prev_acc = history[-2]["test"]["accuracy"]
        curr_acc = entry["test"]["accuracy"]
        delta    = (curr_acc - prev_acc) * 100
        arrow    = "▲" if delta >= 0 else "▼"
        print(f"\n  {arrow} vs previous run: "
              f"{prev_acc*100:.1f}% → {curr_acc*100:.1f}% "
              f"({'+' if delta>=0 else ''}{delta:.1f}pp)")

    print(f"\n  ✅ Models saved to: {out_dir}")
    print(f"     scaler.pkl")
    print(f"     svm_model.pkl")
    print(f"     rf_model.pkl")
    print(f"     training_history.json  ({len(history)} run(s) logged)")

# =============================================================================
# ── STEP 6: PLOTS
# =============================================================================
def plot_all(train_m, val_m, test_m):
    if not PLOT_OK: return

    # Figure 1: Three confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("NeuroWatch  |  Bonn Dataset — Confusion Matrices (All Splits)",
                 fontsize=13, fontweight="bold")
    for ax, m, cmap in zip(axes, [train_m, val_m, test_m],
                           ["Blues", "Greens", "Oranges"]):
        sns.heatmap(m["cm"], annot=True, fmt="d", cmap=cmap,
                    xticklabels=m["lnames"], yticklabels=m["lnames"],
                    cbar=False, ax=ax, square=True, annot_kws={"size": 13})
        ax.set_title(f"{m['label']}\nAcc: {m['acc']*100:.1f}%  "
                     f"F1: {m['f1']*100:.1f}%", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=False); plt.pause(0.3)

    # Figure 2: Metrics comparison bar
    fig2, ax2 = plt.subplots(figsize=(13, 6))
    fig2.suptitle("NeuroWatch  |  Bonn Dataset — Metrics Comparison (All Splits)",
                  fontsize=13, fontweight="bold")
    m_labels = ["Accuracy","Precision","Recall","F1","Specificity","Seizure Recall"]
    tr_vals  = [train_m["acc"],  train_m["prec"],  train_m["rec"],
                train_m["f1"],   train_m["spec"],  train_m["sz_rec"]]
    vl_vals  = [val_m["acc"],    val_m["prec"],    val_m["rec"],
                val_m["f1"],     val_m["spec"],    val_m["sz_rec"]]
    te_vals  = [test_m["acc"],   test_m["prec"],   test_m["rec"],
                test_m["f1"],    test_m["spec"],   test_m["sz_rec"]]
    x = np.arange(len(m_labels)); w = 0.26
    b1 = ax2.bar(x - w,  tr_vals, w, label="Train (80%)",
                 color="#4fc3f7", alpha=0.9, edgecolor="#333")
    b2 = ax2.bar(x,      vl_vals, w, label="Val   (10%)",
                 color="#00CC66", alpha=0.9, edgecolor="#333")
    b3 = ax2.bar(x + w,  te_vals, w, label="Test  (10%)",
                 color="#FFD633", alpha=0.9, edgecolor="#333")
    ax2.set_xticks(x); ax2.set_xticklabels(m_labels, fontsize=10)
    ax2.set_ylim(0, 1.18); ax2.set_ylabel("Score")
    ax2.legend(fontsize=10, framealpha=0.3); ax2.grid(True, axis="y")
    for bars in [b1, b2, b3]:
        for b in bars:
            h = b.get_height()
            ax2.text(b.get_x()+b.get_width()/2, h+0.015,
                     f"{h*100:.0f}%", ha="center", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=False); plt.pause(0.3)

    # Figure 3: Confidence distributions — validation set
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle("NeuroWatch  |  Confidence Distributions — Validation Set",
                  fontsize=12, fontweight="bold")
    for col, cls_idx in enumerate(range(3)):
        ax   = axes3[col]
        mask = val_m["y_true"] == cls_idx
        if mask.any():
            c = val_m["confs"][mask, cls_idx]
            ax.hist(c, bins=20, color=COLORS[cls_idx], alpha=0.85, edgecolor="#333")
            ax.axvline(c.mean(), color="#fff", ls="--", lw=1.5,
                       label=f"mean={c.mean():.2f}")
            ax.legend(fontsize=9)
        ax.set_title(f"True {CLASSES[cls_idx]}")
        ax.set_xlabel(f"Model confidence → {CLASSES[cls_idx]}")
        ax.set_xlim(0, 1); ax.grid(True)
    plt.tight_layout()
    plt.show(block=False); plt.pause(0.3)

    # Figure 4: Overfitting check
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    fig4.suptitle("NeuroWatch  |  Overfitting Check — Train vs Validation",
                  fontsize=12, fontweight="bold")
    gaps = [t - v for t, v in zip(tr_vals, vl_vals)]
    cols = ["#00CC66" if abs(g) < 0.05 else "#FFD633" if abs(g) < 0.12 else "#FF3333"
            for g in gaps]
    bars4 = ax4.bar(m_labels, gaps, color=cols, edgecolor="#333")
    ax4.axhline(0,      color="#aaa", lw=1)
    ax4.axhline( 0.05,  color="#00CC66", lw=1, ls="--", alpha=0.5,
                label="±5% (no overfit)")
    ax4.axhline(-0.05,  color="#00CC66", lw=1, ls="--", alpha=0.5)
    ax4.axhline( 0.12,  color="#FFD633", lw=1, ls="--", alpha=0.5,
                label="±12% (mild overfit)")
    ax4.axhline(-0.12,  color="#FFD633", lw=1, ls="--", alpha=0.5)
    ax4.set_ylabel("Train − Validation gap")
    ax4.set_ylim(-0.35, 0.35)
    ax4.legend(fontsize=9, framealpha=0.3); ax4.grid(True, axis="y")
    for b, g in zip(bars4, gaps):
        ax4.text(b.get_x()+b.get_width()/2,
                 g + (0.01 if g >= 0 else -0.025),
                 f"{g:+.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.show(block=False); plt.pause(0.3)

    # Figure 5: Raw signal preview — one example per folder
    try:
        fig5, axes5 = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
        fig5.suptitle("NeuroWatch  |  Raw Signal Preview (one file per folder)",
                      fontsize=11, fontweight="bold")
        folder_cols = {"O":"#00CC66","N":"#4fc3f7",
                       "F":"#FFD633","S":"#ffb74d","Z":"#FF3333"}
        for ax, folder in zip(axes5, sorted(CLASS_MAP.keys())):
            folder_path = os.path.join(args.data, folder)
            txts = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
            if txts:
                raw = np.loadtxt(txts[0])
                t   = np.arange(len(raw)) / FS_BONN
                ax.plot(t, raw, color=folder_cols[folder], lw=0.6, alpha=0.9)
                ax.set_title(f"{folder}/  {CLASSES[CLASS_MAP[folder]]}\n"
                             f"{FOLDER_DESC[folder]}", fontsize=8)
                ax.set_xlabel("Time (s)"); ax.grid(True)
                if folder == "F": ax.set_ylabel("Amplitude (µV)")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show(block=False); plt.pause(0.3)
    except Exception:
        pass

# =============================================================================
# ── FINAL SUMMARY
# =============================================================================
def print_summary(train_m, val_m, test_m):
    overfit = train_m["acc"] - val_m["acc"]
    if overfit < 0.05:   ov_msg = "✅  No overfitting"
    elif overfit < 0.12: ov_msg = "⚠️  Mild overfitting — acceptable for 400 samples"
    else:                ov_msg = "❌  High overfitting — model memorised training data"

    print("\n" + "╔" + "═"*65 + "╗")
    print("║" + "  TRAINING COMPLETE — FINAL SUMMARY".center(65) + "║")
    print("╠" + "═"*65 + "╣")
    print(f"║  {'Metric':<20} {'Train (80%)':>14} {'Val (10%)':>12} {'Test (10%)':>12} ║")
    print("╠" + "─"*65 + "╣")
    for name, tv, vv, ev in [
        ("Accuracy",       train_m["acc"],    val_m["acc"],    test_m["acc"]),
        ("F1 (weighted)",  train_m["f1"],     val_m["f1"],     test_m["f1"]),
        ("Specificity",    train_m["spec"],   val_m["spec"],   test_m["spec"]),
        ("Seizure Recall", train_m["sz_rec"], val_m["sz_rec"], test_m["sz_rec"]),
    ]:
        print(f"║  {name:<20} {tv*100:>13.1f}%  {vv*100:>11.1f}%  {ev*100:>11.1f}% ║")
    print("╠" + "─"*65 + "╣")
    print(f"║  {'Overfitting gap':<20} {'(train - val)':>14} "
          f"{overfit*100:>+10.1f}pp  {'':>12} ║")
    print("╠" + "═"*65 + "╣")
    print(f"║  {ov_msg:<63} ║")
    print("╠" + "─"*65 + "╣")
    print(f"║  {'Dataset':<20} {'Bonn University EEG (500 files)':>43} ║")
    print(f"║  {'Features':<20} {'5 bands + 8 stats = 13 per file':>43} ║")
    print(f"║  {'Split':<20} {'Stratified 80 / 10 / 10':>43} ║")
    print(f"║  {'Normalisation':<20} {'Option A — relative band proportions':>43} ║")
    print(f"║  {'Model':<20} {'Hybrid SVM (40%) + RF (60%)':>43} ║")
    print(f"║  {'Models saved to':<20} {args.out[:43]:>43} ║")
    print("╚" + "═"*65 + "╝")
    print()
    print("  Next step:  python main_pi_bios_v15.py")
    print("              streamlit run dashboard_v2.py")

# =============================================================================
# ── MAIN
# =============================================================================
def main():
    print("\n" + "═"*65)
    print("  NeuroWatch  —  Bonn Dataset Training & Export")
    print("═"*65)
    print(f"  Data path    : {args.data}")
    print(f"  Output path  : {args.out}")
    print(f"  Features     : 5 bands + 8 stats = {N_FEATURES} per file")
    print(f"  Split        : Stratified 80 / 10 / 10")
    print(f"  SMOTE        : {'enabled' if SMOTE_AVAILABLE else 'disabled (pip install imbalanced-learn)'}")
    print(f"  History      : {'RESET' if args.reset else 'append to existing'}")
    print(f"  Plots        : {'on' if PLOT_OK else 'off'}")

    # 1. Load dataset and extract features
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(args.data)

    # 2. Train
    scaler, svm, rf, svm_w, rf_w = train(X_train, y_train, X_val, y_val)

    # 3. Evaluate all 3 splits
    print(f"\n{'─'*65}")
    print(f"  STEP 3: Evaluating all three splits...")
    print(f"{'─'*65}")
    train_m = evaluate_split(X_train, y_train, scaler, svm, rf, "Train (80%)", svm_w, rf_w)
    val_m   = evaluate_split(X_val,   y_val,   scaler, svm, rf, "Validation (10%)", svm_w, rf_w)
    test_m  = evaluate_split(X_test,  y_test,  scaler, svm, rf, "Test (10%)", svm_w, rf_w)

    # 4. Save models + log
    print(f"\n{'─'*65}")
    print(f"  STEP 4: Saving models...")
    print(f"{'─'*65}")
    save_models(scaler, svm, rf, train_m, val_m, test_m, args.out, svm_w, rf_w)

    # 5. Summary
    print_summary(train_m, val_m, test_m)

    # 6. Plots
    plot_all(train_m, val_m, test_m)

    if PLOT_OK:
        print("\n  All plots open — press Enter to close and exit...")
        input()
        plt.close("all")


if __name__ == "__main__":
    main()