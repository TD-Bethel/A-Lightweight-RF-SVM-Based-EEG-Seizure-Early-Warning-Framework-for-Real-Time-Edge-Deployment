# =============================================================================
# train_mendeley.py  —  NeuroWatch Model Training (Mendeley Dataset)
#
# DATASET:
#   Mendeley "Epileptic EEG Dataset" (Wassim Nasreddine, AUB)
#   Input files: x_train.npy, y_train.npy, x_test.npy, y_test.npy
#   Located in:  data/Mendelay dataset/Npy_files/
#
# SPLIT STRATEGY (stratified, from combined 7790 samples):
#   Train      : 80%  → 6232 samples  — model learns from this
#   Validation : 10%  → 779  samples  — used during training to detect overfitting
#   Test       : 10%  → 779  samples  — final untouched evaluation
#
#   Stratified split: every class keeps identical proportions in all 3 sets.
#   This is critical for the minority Video-detected class (only 111 samples).
#
# LABEL MAPPING (4-class Mendeley → 3-class NeuroWatch):
#   0 = Normal              →  0 = Normal
#   1 = CPS                 →  1 = Pre-Seizure  (partial seizure, subtle EEG)
#   2 = Electrographic      →  2 = Seizure      (clearest EEG signal)
#   3 = Video-detected      →  1 = Pre-Seizure  (behavioural only, no EEG change)
#
# FEATURES (per sample):
#   All 19 channels × 4 bands (delta/theta/alpha/beta) = 76 features
#   Option A normalisation: each channel's 4 bands divided by their sum
#   → relative proportions, scale-invariant across any EEG equipment
#
# MODEL:
#   Hybrid: 40% SVM (RBF) + 60% Random Forest — same blend as main_pi_bios_v15.py
#   StandardScaler fitted on training set ONLY — no leakage into val or test
#
# OUTPUT (saved to models/MODELS_MENDELEY/):
#   scaler.pkl
#   svm_model.pkl
#   rf_model.pkl
#   training_report.json   ← full metrics for all 3 splits
#
# USAGE:
#   python train_mendeley.py
#   python train_mendeley.py --data "C:\...\Npy_files" --out "C:\...\MODELS_V1"
#   python train_mendeley.py --no-plots
# =============================================================================

import os, sys, time, json, argparse, warnings
import numpy as np
import joblib
from scipy.signal import butter, filtfilt, welch, resample
from scipy.stats import skew, kurtosis as scipy_kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠  imbalanced-learn not found — install with: pip install imbalanced-learn")

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS  —  update if your folders are in different locations
# =============================================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))

# Where the npy files live  (data/Mendelay dataset/Npy_files/)
DEFAULT_DATA = os.path.join(BASE_DIR, "data", "Mendelay dataset", "Npy_files")

# Where trained models are saved  (models/MODELS_MENDELEY/)
DEFAULT_OUT  = os.path.join(BASE_DIR, "models", "MODELS_MENDELEY")

# =============================================================================
# CONSTANTS
# =============================================================================
TARGET_FS    = 128      # resample everything to this
FS_MENDELEY  = 500      # native Mendeley rate
WIN_SIZE     = 4097     # tiling window size for Welch consistency
N_ORDER      = 4
F_LOW, F_HIGH = 0.5, 45.0
NPERSEG      = 128
N_CHANNELS   = 19
N_BANDS      = 5        # delta, theta, alpha, beta, gamma
N_STATS      = 8        # activity, mobility, complexity, skewness, kurtosis, rms, zcr, spectral_entropy
N_CORR       = N_CHANNELS * (N_CHANNELS - 1) // 2  # 171 inter-channel correlation pairs
N_FEATURES   = N_CHANNELS * (N_BANDS + N_STATS) + N_CORR  # 247 + 171 = 418

TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
TEST_RATIO   = 0.10
RANDOM_SEED  = 42

# Mendeley 4-class → NeuroWatch 3-class
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
# ARGUMENT PARSING
# =============================================================================
ap = argparse.ArgumentParser(description="NeuroWatch — Mendeley Model Trainer")
ap.add_argument("--data",     default=DEFAULT_DATA,
                help="Path to folder containing x_train.npy etc.")
ap.add_argument("--out",      default=DEFAULT_OUT,
                help="Where to save trained model .pkl files")
ap.add_argument("--no-plots", action="store_true",
                help="Skip all matplotlib plots")
args = ap.parse_args()

PLOT = not args.no_plots
if PLOT:
    try:
        import matplotlib; matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import Patch
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
    """Extract [delta, theta, alpha, beta, gamma] power from one channel."""
    filt   = _bandpass(sig, fs)
    f, psd = welch(filt, fs=fs, nperseg=NPERSEG)
    return np.array([
        np.mean(psd[(f >= 1)  & (f <= 4)]),
        np.mean(psd[(f >= 4)  & (f <= 8)]),
        np.mean(psd[(f >= 8)  & (f <= 13)]),
        np.mean(psd[(f >= 13) & (f <= 30)]),
        np.mean(psd[(f >= 30) & (f <= 45)]),   # gamma (up to filter cutoff)
    ])

def _normalise(feat_vec):
    total = feat_vec.sum()
    return feat_vec / total if total > 0 else feat_vec

def _hjorth(sig):
    """Hjorth activity, mobility, complexity."""
    d1      = np.diff(sig)
    d2      = np.diff(d1)
    var_x   = np.var(sig)  + 1e-10
    var_d1  = np.var(d1)   + 1e-10
    var_d2  = np.var(d2)   + 1e-10
    activity   = float(var_x)
    mobility   = float(np.sqrt(var_d1 / var_x))
    mob_d1     = float(np.sqrt(var_d2 / var_d1))
    complexity = float(mob_d1 / (mobility + 1e-10))
    return activity, mobility, complexity

def _spectral_entropy(psd):
    """Normalised spectral entropy."""
    p = psd / (psd.sum() + 1e-10)
    return float(-np.sum(p * np.log2(p + 1e-10)) / np.log2(len(p) + 1))

def _zcr(sig):
    return float(((sig[:-1] * sig[1:]) < 0).sum() / len(sig))

def extract_features(sample):
    """
    Extract 418-feature vector from one Mendeley sample.
    sample: (19, 500) — 19 channels × 1 second at 500 Hz

    Per channel (13 features × 19 channels = 247):
      Band powers (5): delta, theta, alpha, beta, gamma  — Option A normalised
      Statistical (8): activity, mobility, complexity,
                       skewness, kurtosis, rms, zcr, spectral_entropy

    Inter-channel (171 = 19×18÷2 correlation pairs):
      Pearson correlation between every pair of channels — captures
      pre-ictal synchrony (channels correlate more before a seizure).

    Returns (418,) float32 array.
    """
    per_ch_feats = []
    resampled    = []

    for ch in range(N_CHANNELS):
        sig   = sample[ch]
        n_new = int(len(sig) * TARGET_FS / FS_MENDELEY)
        sig_r = resample(sig, n_new)
        reps  = (WIN_SIZE // len(sig_r)) + 1
        sig_t = np.tile(sig_r, reps)[:WIN_SIZE]
        resampled.append(sig_r)

        # Band powers — Option A normalised
        bp      = _band_power(sig_t)
        bp_norm = _normalise(bp)

        # Hjorth parameters
        act, mob, comp = _hjorth(sig_r)

        # Statistical features
        sk   = float(skew(sig_r))
        kurt = float(scipy_kurtosis(sig_r))
        rms  = float(np.sqrt(np.mean(sig_r ** 2)))
        zcr  = _zcr(sig_r)

        # Spectral entropy
        filt    = _bandpass(sig_t, TARGET_FS)
        _, psd  = welch(filt, fs=TARGET_FS, nperseg=NPERSEG)
        sent    = _spectral_entropy(psd)

        per_ch_feats.extend(bp_norm.tolist())
        per_ch_feats.extend([act, mob, comp, sk, kurt, rms, zcr, sent])

    # Inter-channel Pearson correlation (171 pairs)
    # Pre-ictal biomarker: channels synchronise before seizure onset
    corr_feats = []
    for i in range(N_CHANNELS):
        for j in range(i + 1, N_CHANNELS):
            # Align lengths in case of rounding
            a = resampled[i]; b = resampled[j]
            n = min(len(a), len(b))
            c = float(np.corrcoef(a[:n], b[:n])[0, 1])
            corr_feats.append(0.0 if np.isnan(c) else c)

    return np.array(per_ch_feats + corr_feats, dtype=np.float32)

# =============================================================================
# ── STEP 2: LOAD NPY FILES & BUILD FEATURE MATRIX
# =============================================================================
def load_and_extract(data_dir):
    print(f"\n{'─'*65}")
    print(f"  STEP 1: Loading npy files from:")
    print(f"  {data_dir}")
    print(f"{'─'*65}")

    def try_load(xname, yname):
        xp = os.path.join(data_dir, xname)
        yp = os.path.join(data_dir, yname)
        if os.path.exists(xp) and os.path.exists(yp):
            X = np.load(xp); Y = np.load(yp)
            print(f"  ✅ {xname:<20} {str(X.shape):<20}  "
                  f"{yname:<20} {str(Y.shape)}")
            return X, Y
        return None, None

    Xtr, Ytr = try_load("x_train.npy", "y_train.npy")
    Xte, Yte = try_load("x_test.npy",  "y_test.npy")

    if Xtr is None and Xte is None:
        print(f"  ❌ No npy files found in {data_dir}")
        print(f"     Update DEFAULT_DATA at the top of this file.")
        sys.exit(1)

    parts_x = [x for x in [Xtr, Xte] if x is not None]
    parts_y = [y for y in [Ytr, Yte] if y is not None]
    X_all   = np.concatenate(parts_x, axis=0)
    y_all   = np.concatenate(parts_y, axis=0)

    print(f"\n  Combined: {len(X_all)} samples  |  shape per sample: {X_all[0].shape}")
    print(f"\n  Original 4-class distribution:")
    for lbl in sorted(np.unique(y_all)):
        cnt = int(np.sum(y_all == lbl))
        bar = "█" * int(cnt / len(y_all) * 40)
        print(f"    {lbl} = {LABEL_NAMES[lbl]:<35} {cnt:5d}  {bar}")

    # ── Stratified 80/10/10 split ────────────────────────────────────────────
    # First split off 20% (val + test), then split that 20% in half
    print(f"\n  Applying stratified 80 / 10 / 10 split  (seed={RANDOM_SEED})...")

    X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
        X_all, y_all,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=y_all,
    )
    X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
        X_temp, y_temp,
        test_size=0.5,                    # half of the 20% = 10%
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    print(f"  Train : {len(X_train_raw):5d} samples  ({len(X_train_raw)/len(X_all)*100:.0f}%)")
    print(f"  Val   : {len(X_val_raw):5d} samples  ({len(X_val_raw)/len(X_all)*100:.0f}%)")
    print(f"  Test  : {len(X_test_raw):5d} samples  ({len(X_test_raw)/len(X_all)*100:.0f}%)")

    # Verify stratification
    print(f"\n  Stratification check — class proportions should match across splits:")
    print(f"  {'Label':<25} {'All':>7} {'Train':>7} {'Val':>7} {'Test':>7}")
    for lbl in sorted(np.unique(y_all)):
        ap  = np.mean(y_all       == lbl) * 100
        tp  = np.mean(y_train_raw == lbl) * 100
        vp  = np.mean(y_val_raw   == lbl) * 100
        ep  = np.mean(y_test_raw  == lbl) * 100
        ok  = "✅" if max(abs(tp-vp), abs(tp-ep)) < 1.5 else "⚠"
        print(f"  {ok} {LABEL_NAMES[lbl]:<23} "
              f"{ap:>6.1f}%  {tp:>6.1f}%  {vp:>6.1f}%  {ep:>6.1f}%")

    # ── Feature extraction ───────────────────────────────────────────────────
    print(f"\n  STEP 2: Extracting {N_FEATURES} features per sample "
          f"({N_CHANNELS} channels × {N_BANDS} bands)...")
    print(f"  Option A normalisation: ON  (relative band power proportions)")
    print(f"  This may take a few minutes...\n")

    def extract_split(X_raw, y_raw, name):
        feats, labels = [], []
        n = len(X_raw)
        t0 = time.time()
        for i, (sample, lbl) in enumerate(zip(X_raw, y_raw)):
            if i % 200 == 0:
                pct = i / n * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"  [{bar}] {pct:5.1f}%  {name}  ({i}/{n})", end="\r")
            feats.append(extract_features(sample))
            labels.append(LABEL_MAP[int(lbl)])
        elapsed = time.time() - t0
        print(f"  [{' █'*20}] 100.0%  {name} done  ({elapsed:.0f}s)       ")

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
# ── STEP 3: TRAIN
# =============================================================================
def train(X_train, y_train, X_val, y_val):
    print(f"\n{'─'*65}")
    print(f"  STEP 3: Training  (StandardScaler + SMOTE + SVM + Random Forest)")
    print(f"{'─'*65}")

    # ── Scaler — fitted on train ONLY ────────────────────────────────────────
    print(f"\n  Fitting StandardScaler on {len(X_train)} training samples...")
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    # ── SMOTE — balance classes on scaled training data ───────────────────────
    if SMOTE_AVAILABLE:
        print(f"\n  Applying SMOTE to balance classes...")
        uniq, cnts = np.unique(y_train, return_counts=True)
        print(f"  Before: { {CLASSES[u]: int(c) for u, c in zip(uniq, cnts)} }")
        sm = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
        X_tr_sc, y_train_bal = sm.fit_resample(X_tr_sc, y_train)
        uniq2, cnts2 = np.unique(y_train_bal, return_counts=True)
        print(f"  After : { {CLASSES[u]: int(c) for u, c in zip(uniq2, cnts2)} }")
        y_train = y_train_bal
        # Unscaled version for RF (RF doesn't need scaled input)
        X_train = scaler.inverse_transform(X_tr_sc)
    else:
        print(f"  ⚠  SMOTE skipped — install imbalanced-learn for class balancing")

    # ── SVM ──────────────────────────────────────────────────────────────────
    print(f"\n  Training SVM (RBF, C=10, balanced)...")
    t0  = time.time()
    svm = SVC(
        kernel="rbf", C=10, gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    svm.fit(X_tr_sc, y_train)
    svm_time = time.time() - t0
    svm_val  = accuracy_score(y_val, svm.predict(X_val_sc))
    print(f"  SVM done in {svm_time:.1f}s  |  "
          f"Support vectors: {sum(svm.n_support_)}  |  "
          f"Val accuracy: {svm_val*100:.1f}%")

    # ── Random Forest ─────────────────────────────────────────────────────────
    print(f"\n  Training Random Forest (500 trees, max_depth=12, balanced)...")
    t0 = time.time()
    rf  = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    rf_val  = accuracy_score(y_val, rf.predict(X_val))
    print(f"  RF done in {rf_time:.1f}s  |  Val accuracy: {rf_val*100:.1f}%")

    # ── Auto-tune ensemble weights on validation set ──────────────────────────
    print(f"\n  Searching for best SVM/RF blend on validation set...")
    p_svm_val = svm.predict_proba(X_val_sc)
    p_rf_val  = rf.predict_proba(X_val)
    best_w, best_acc = 0.4, 0.0
    for w in np.arange(0.05, 1.0, 0.05):
        acc = accuracy_score(y_val, np.argmax(w * p_svm_val + (1-w) * p_rf_val, axis=1))
        if acc > best_acc:
            best_acc, best_w = acc, w
    SVM_W, RF_W = round(best_w, 2), round(1.0 - best_w, 2)
    hybrid = np.argmax(SVM_W * p_svm_val + RF_W * p_rf_val, axis=1)
    print(f"  Best blend → SVM {SVM_W:.0%} + RF {RF_W:.0%}  "
          f"val accuracy: {accuracy_score(y_val, hybrid)*100:.1f}%")

    # ── Top 10 feature importances ────────────────────────────────────────────
    feat_names = ["delta","theta","alpha","beta","gamma",
                  "activity","mobility","complexity",
                  "skewness","kurtosis","rms","zcr","spectral_entropy"]
    top10 = np.argsort(rf.feature_importances_)[::-1][:10]
    print(f"\n  Top 10 features (RF importance):")
    for rank, idx in enumerate(top10, 1):
        ch   = idx // (N_BANDS + N_STATS)
        feat = feat_names[idx % (N_BANDS + N_STATS)]
        imp  = rf.feature_importances_[idx]
        bar  = "█" * int(imp * 300)
        print(f"    {rank:2}. ch{ch:02d}-{feat:<18}  {imp:.4f}  {bar}")

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

    tn = cm[0,0] if 0 in y_true else 0
    fp = cm[0,1:].sum() if 0 in y_true else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Per-class metrics
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
    print(f"  Seizure Recall : {sz_rec*100:.1f}%  "
          f"← most clinically important metric")
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
# ── STEP 5: SAVE MODELS + REPORT
# =============================================================================
def save_models(scaler, svm, rf, train_m, val_m, test_m, out_dir, svm_w=0.4, rf_w=0.6):
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(svm,    os.path.join(out_dir, "svm_model.pkl"))
    joblib.dump(rf,     os.path.join(out_dir, "rf_model.pkl"))
    # Save weights so the live engine can load them
    joblib.dump({"svm_w": svm_w, "rf_w": rf_w},
                os.path.join(out_dir, "ensemble_weights.pkl"))

    # JSON report
    report = {
        "trained_on"   : "Mendeley Epileptic EEG Dataset",
        "timestamp"    : __import__("datetime").datetime.now().isoformat(),
        "split"        : "stratified 80/10/10",
        "n_features"   : N_FEATURES,
        "n_channels"   : N_CHANNELS,
        "normalisation": "Option A — relative band power proportions",
        "model"        : f"Hybrid SVM ({svm_w:.0%}) + RF ({rf_w:.0%}) — auto-tuned",
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

    print(f"\n  ✅ Saved to: {out_dir}")
    print(f"     scaler.pkl")
    print(f"     svm_model.pkl")
    print(f"     rf_model.pkl")
    print(f"     training_report.json")
    print(f"\n  To use with main_pi_bios_v15.py:")
    print(f"    python main_pi_bios_v15.py --models \"{out_dir}\"")

    return report

# =============================================================================
# ── STEP 6: PLOTS
# =============================================================================
def plot_all(train_m, val_m, test_m):
    if not PLOT_OK: return

    # ── Figure 1: Three confusion matrices side by side ───────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("NeuroWatch  |  Confusion Matrices — All Splits",
                 fontsize=13, fontweight="bold")
    cmap_colors = ["Blues", "Greens", "Oranges"]
    for ax, m, cmap in zip(axes, [train_m, val_m, test_m], cmap_colors):
        sns.heatmap(m["cm"], annot=True, fmt="d", cmap=cmap,
                    xticklabels=m["lnames"], yticklabels=m["lnames"],
                    cbar=False, ax=ax, square=True, annot_kws={"size": 12})
        ax.set_title(f"{m['label']}\nAcc: {m['acc']*100:.1f}%  "
                     f"F1: {m['f1']*100:.1f}%",
                     fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=False); plt.pause(0.3)

    # ── Figure 2: Metrics comparison bar — train vs val vs test ──────────────
    fig2, ax = plt.subplots(figsize=(13, 6))
    fig2.suptitle("NeuroWatch  |  Metrics Comparison — Train / Val / Test",
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
    plt.show(block=False); plt.pause(0.3)

    # ── Figure 3: Confidence distributions — val set per class ───────────────
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle("NeuroWatch  |  Confidence Distributions — Validation Set",
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
        ax.set_xlabel("Model confidence  →  {CLASSES[cls_idx]}")
        ax.set_xlim(0, 1); ax.grid(True)
    plt.tight_layout()
    plt.show(block=False); plt.pause(0.3)

    # ── Figure 4: Overfitting check — train vs val per metric ─────────────────
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    fig4.suptitle("NeuroWatch  |  Overfitting Check — Train vs Validation",
                  fontsize=12, fontweight="bold")
    gaps = [t - v for t, v in zip(tr_vals, vl_vals)]
    cols = ["#00CC66" if abs(g) < 0.05 else "#FFD633" if abs(g) < 0.12 else "#FF3333"
            for g in gaps]
    bars4 = ax4.bar(m_labels, gaps, color=cols, edgecolor="#333")
    ax4.axhline(0, color="#aaa", lw=1)
    ax4.axhline(0.05,  color="#00CC66", lw=1, ls="--", alpha=0.5, label="±5% (good)")
    ax4.axhline(-0.05, color="#00CC66", lw=1, ls="--", alpha=0.5)
    ax4.axhline(0.12,  color="#FFD633", lw=1, ls="--", alpha=0.5, label="±12% (mild)")
    ax4.axhline(-0.12, color="#FFD633", lw=1, ls="--", alpha=0.5)
    ax4.set_ylabel("Train − Validation gap"); ax4.set_ylim(-0.3, 0.3)
    ax4.legend(fontsize=9, framealpha=0.3); ax4.grid(True, axis="y")
    for b, g in zip(bars4, gaps):
        ax4.text(b.get_x()+b.get_width()/2,
                 g + (0.01 if g >= 0 else -0.02),
                 f"{g:+.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.show(block=False); plt.pause(0.3)


# =============================================================================
# ── FINAL SUMMARY
# =============================================================================
def print_summary(train_m, val_m, test_m, report):
    overfit = train_m["acc"] - val_m["acc"]
    if overfit < 0.05:   ov_msg = "✅  No overfitting"
    elif overfit < 0.12: ov_msg = "⚠️  Mild overfitting — acceptable"
    else:                ov_msg = "❌  High overfitting — consider regularisation"

    print("\n" + "╔" + "═"*65 + "╗")
    print("║" + "  TRAINING COMPLETE — FINAL SUMMARY".center(65) + "║")
    print("╠" + "═"*65 + "╣")
    print(f"║  {'Metric':<20} {'Train (80%)':>14} {'Val (10%)':>12} {'Test (10%)':>12} ║")
    print("╠" + "─"*65 + "╣")
    rows = [
        ("Accuracy",        train_m["acc"],    val_m["acc"],    test_m["acc"]),
        ("F1 (weighted)",   train_m["f1"],     val_m["f1"],     test_m["f1"]),
        ("Specificity",     train_m["spec"],   val_m["spec"],   test_m["spec"]),
        ("Seizure Recall",  train_m["sz_rec"], val_m["sz_rec"], test_m["sz_rec"]),
    ]
    for name, tv, vv, ev in rows:
        print(f"║  {name:<20} {tv*100:>13.1f}%  {vv*100:>11.1f}%  {ev*100:>11.1f}% ║")
    print("╠" + "─"*65 + "╣")
    print(f"║  {'Overfitting gap':<20} {'(train - val)':>14} {overfit*100:>+10.1f}pp  "
          f"{'':>12} ║")
    print("╠" + "═"*65 + "╣")
    print(f"║  {ov_msg:<63} ║")
    print("╠" + "─"*65 + "╣")
    print(f"║  {'Features':<20} {'19 channels × 4 bands = 76 per sample':>43} ║")
    print(f"║  {'Split':<20} {'Stratified 80 / 10 / 10':>43} ║")
    print(f"║  {'Normalisation':<20} {'Option A — relative band proportions':>43} ║")
    print(f"║  {'Model':<20} {'Hybrid SVM (40%) + RF (60%)':>43} ║")
    print("╚" + "═"*65 + "╝")


# =============================================================================
# ── MAIN
# =============================================================================
def main():
    print("\n" + "═"*65)
    print("  NeuroWatch  —  Mendeley Model Trainer")
    print("═"*65)
    print(f"  Data path    : {args.data}")
    print(f"  Output path  : {args.out}")
    print(f"  Features     : {N_CHANNELS} channels × {N_BANDS} bands + {N_STATS} stats = {N_FEATURES}")
    print(f"  Split        : Stratified 80 / 10 / 10")
    print(f"  Normalisation: Option A (relative band power) + statistical features")
    print(f"  SMOTE        : {'enabled' if SMOTE_AVAILABLE else 'disabled (pip install imbalanced-learn)'}")
    print(f"  Random seed  : {RANDOM_SEED}")

    # 1. Load and extract features
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_extract(args.data)

    # 2. Train
    scaler, svm, rf, svm_w, rf_w = train(X_train, y_train, X_val, y_val)

    # 3. Evaluate all 3 splits
    print(f"\n{'─'*65}")
    print(f"  STEP 4: Evaluating all three splits...")
    print(f"{'─'*65}")
    train_m = evaluate_split(X_train, y_train, scaler, svm, rf, "Train (80%)", svm_w, rf_w)
    val_m   = evaluate_split(X_val,   y_val,   scaler, svm, rf, "Validation (10%)", svm_w, rf_w)
    test_m  = evaluate_split(X_test,  y_test,  scaler, svm, rf, "Test (10%)", svm_w, rf_w)

    # 4. Save
    print(f"\n{'─'*65}")
    print(f"  STEP 5: Saving models...")
    print(f"{'─'*65}")
    report = save_models(scaler, svm, rf, train_m, val_m, test_m, args.out, svm_w, rf_w)

    # 5. Summary
    print_summary(train_m, val_m, test_m, report)

    # 6. Plots
    plot_all(train_m, val_m, test_m)

    if PLOT_OK:
        print("\n  All plots open — press Enter to close and exit...")
        input()
        plt.close("all")


if __name__ == "__main__":
    main()
