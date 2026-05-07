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
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("WARN  mlflow not found -- pip install mlflow  (tracking disabled)")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("WARN  imbalanced-learn not found -- install with: pip install imbalanced-learn")

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from neurowatch_selector import RFImportanceSelector

# =============================================================================
# PATHS  --  update if your folders are in different locations
# =============================================================================
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
N_CORR       = 0  # correlation features removed to reduce overfitting
N_FEATURES   = N_CHANNELS * (N_BANDS + N_STATS)  # 19 x 13 = 247
N_SELECT     = 75  # top-K features kept after SelectKBest(f_classif)

TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
TEST_RATIO   = 0.10
RANDOM_SEED  = 42

# Mendeley 4-class -> NeuroWatch 3-class
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
ap = argparse.ArgumentParser(description="NeuroWatch -- Mendeley Model Trainer")
ap.add_argument("--data",       default=DEFAULT_DATA,
                help="Path to folder containing x_train.npy etc.")
ap.add_argument("--out",        default=DEFAULT_OUT,
                help="Where to save trained model .pkl files")
ap.add_argument("--no-plots",   action="store_true",
                help="Skip all matplotlib plots")
ap.add_argument("--svm-weight", type=float, default=None,
                help="Fix SVM blend weight (0.0–1.0). Skips auto-tuning. "
                     "e.g. --svm-weight 0.40 gives SVM 40%% + RF 60%%")
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
        print(f"  WARN  Plots unavailable: {e}")
        PLOT_OK = False
else:
    PLOT_OK = False

# =============================================================================
# -- STEP 1: SIGNAL PROCESSING & FEATURE EXTRACTION
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
        n_new = int(len(sig) * TARGET_FS / FS_MENDELEY)
        sig_r = resample(sig, n_new)
        reps  = (WIN_SIZE // len(sig_r)) + 1
        sig_t = np.tile(sig_r, reps)[:WIN_SIZE]
        resampled.append(sig_r)

        # Band powers -- Option A normalised
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

    return np.array(per_ch_feats, dtype=np.float32)

# =============================================================================
# -- STEP 2: LOAD NPY FILES & BUILD FEATURE MATRIX
# =============================================================================
def load_and_extract(data_dir):
    print(f"\n{'-'*65}")
    print(f"  STEP 1: Loading npy files from:")
    print(f"  {data_dir}")
    print(f"{'-'*65}")

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

    parts_x = [x for x in [Xtr, Xte] if x is not None]
    parts_y = [y for y in [Ytr, Yte] if y is not None]
    X_all   = np.concatenate(parts_x, axis=0)
    y_all   = np.concatenate(parts_y, axis=0)

    print(f"\n  Combined: {len(X_all)} samples  |  shape per sample: {X_all[0].shape}")
    print(f"\n  Original 4-class distribution:")
    for lbl in sorted(np.unique(y_all)):
        cnt = int(np.sum(y_all == lbl))
        bar = "#" * int(cnt / len(y_all) * 40)
        print(f"    {lbl} = {LABEL_NAMES[lbl]:<35} {cnt:5d}  {bar}")

    # -- Stratified 80/10/10 split --------------------------------------------
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
        test_size=0.5,                    # half of the 20% = 10% each
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    print(f"  Train : {len(X_train_raw):5d} samples  ({len(X_train_raw)/len(X_all)*100:.0f}%)")
    print(f"  Val   : {len(X_val_raw):5d} samples  ({len(X_val_raw)/len(X_all)*100:.0f}%)")
    print(f"  Test  : {len(X_test_raw):5d} samples  ({len(X_test_raw)/len(X_all)*100:.0f}%)")

    # Verify stratification
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

    # -- Feature extraction ---------------------------------------------------
    print(f"\n  STEP 2: Extracting {N_FEATURES} features per sample "
          f"({N_CHANNELS} channels x {N_BANDS} bands)...")
    print(f"  Option A normalisation: ON  (relative band power proportions)")
    print(f"  This may take a few minutes...\n")

    def extract_split(X_raw, y_raw, name):
        feats, labels = [], []
        n = len(X_raw)
        t0 = time.time()
        for i, (sample, lbl) in enumerate(zip(X_raw, y_raw)):
            if i % 200 == 0:
                pct = i / n * 100
                bar = "#" * int(pct / 5) + "#" * (20 - int(pct / 5))
                print(f"  [{bar}] {pct:5.1f}%  {name}  ({i}/{n})", end="\r")
            feats.append(extract_features(sample))
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
# -- STEP 3: TRAIN
# =============================================================================
def train(X_train, y_train, X_val, y_val, forced_svm_w=None):
    print(f"\n{'-'*65}")
    print(f"  STEP 3: Training  (StandardScaler + RF-importance selection + SVM + RF)")
    print(f"{'-'*65}")

    # -- Scaler -- fitted on train ONLY ----------------------------------------
    print(f"\n  Fitting StandardScaler on {len(X_train)} training samples...")
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)

    # -- Bootstrap RF to rank features by importance ---------------------------
    print(f"\n  Bootstrap RF (300 trees) for RF-importance feature selection...")
    selector = RFImportanceSelector(k=N_SELECT)
    selector.fit(X_train, y_train,          # unscaled — RF is scale-invariant
                 n_estimators=300, max_depth=10, random_state=RANDOM_SEED)

    feat_names = ["delta","theta","alpha","beta","gamma",
                  "activity","mobility","complexity",
                  "skewness","kurtosis","rms","zcr","spectral_entropy"]
    top10_boot = np.argsort(selector.importances_)[::-1][:10]
    print(f"  Top 10 features selected (out of {N_FEATURES}):")
    for rank, idx in enumerate(top10_boot, 1):
        ch   = idx // (N_BANDS + N_STATS)
        feat = feat_names[idx % (N_BANDS + N_STATS)]
        imp  = selector.importances_[idx]
        bar  = "#" * int(imp * 300)
        print(f"    {rank:2}. ch{ch:02d}-{feat:<18}  {imp:.4f}  {bar}")

    # -- Apply selection to all splits -----------------------------------------
    X_tr_sel  = selector.transform(X_tr_sc)          # scaled+selected  (SVM)
    X_tr_rf   = selector.transform(X_train)           # unscaled+selected (RF)
    X_val_sc  = scaler.transform(X_val)
    X_val_sel = selector.transform(X_val_sc)
    X_val_rf  = selector.transform(X_val)
    print(f"\n  Feature count: {N_FEATURES} → {N_SELECT} selected")

    # -- SVM — class_weight gives Normal 1.5× penalty for misclassification ----
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

    # -- Random Forest — trains on unscaled + selected features ----------------
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

    # -- Ensemble weights: fixed if forced, otherwise auto-tuned ---------------
    p_svm_val = svm.predict_proba(X_val_sel)
    p_rf_val  = rf.predict_proba(X_val_rf)
    if forced_svm_w is not None:
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
    p_combined_val = SVM_W * p_svm_val + RF_W * p_rf_val
    print(f"  Val accuracy (argmax): {accuracy_score(y_val, np.argmax(p_combined_val, axis=1))*100:.1f}%")

    # -- Normal threshold tuning -----------------------------------------------
    # Only override Pre-Seizure → Normal (never downgrade a Seizure prediction).
    # Searches validation set to find the threshold that maximises accuracy.
    print(f"\n  Tuning Normal threshold on validation set "
          f"(Pre-Seizure → Normal only, Seizure protected)...")
    argmax_val = np.argmax(p_combined_val, axis=1)
    best_thresh, best_thresh_acc = 0.50, 0.0
    for t in np.arange(0.30, 0.65, 0.01):
        y_t = argmax_val.copy()
        y_t[(argmax_val == 1) & (p_combined_val[:, 0] >= t)] = 0
        acc_t = accuracy_score(y_val, y_t)
        if acc_t > best_thresh_acc:
            best_thresh_acc, best_thresh = acc_t, t
    NORMAL_THRESHOLD = round(float(best_thresh), 2)
    print(f"  Best Normal threshold: {NORMAL_THRESHOLD:.2f}  "
          f"(val acc with threshold: {best_thresh_acc*100:.1f}%)")

    return scaler, selector, svm, rf, SVM_W, RF_W, NORMAL_THRESHOLD

# =============================================================================
# -- STEP 4: EVALUATE on all 3 splits
# =============================================================================
def evaluate_split(X, y_true, scaler, selector, svm, rf, label,
                   svm_w=0.4, rf_w=0.6, normal_threshold=0.50):
    X_sc     = scaler.transform(X)
    X_sc_sel = selector.transform(X_sc)
    X_rf_sel = selector.transform(X)
    p_svm    = svm.predict_proba(X_sc_sel)
    p_rf     = rf.predict_proba(X_rf_sel)
    combined = svm_w * p_svm + rf_w * p_rf
    # Apply Normal threshold: override Pre-Seizure → Normal when confident enough.
    # Seizure predictions are never downgraded.
    argmax   = np.argmax(combined, axis=1)
    y_pred   = argmax.copy()
    y_pred[(argmax == 1) & (combined[:, 0] >= normal_threshold)] = 0
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
    print(f"\n  {'-'*60}")
    print(f"  {label}")
    print(f"  {'-'*60}")
    print(f"  Samples        : {len(y_true)}")
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
# -- STEP 5: SAVE MODELS + REPORT
# =============================================================================
def save_models(scaler, selector, svm, rf, train_m, val_m, test_m, out_dir,
                svm_w=0.4, rf_w=0.6, normal_threshold=0.50):
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(scaler,   os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(selector, os.path.join(out_dir, "selector.pkl"))
    joblib.dump(svm,      os.path.join(out_dir, "svm_model.pkl"))
    joblib.dump(rf,       os.path.join(out_dir, "rf_model.pkl"))
    joblib.dump({"svm_w": svm_w, "rf_w": rf_w, "normal_threshold": normal_threshold},
                os.path.join(out_dir, "ensemble_weights.pkl"))

    # JSON report
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

    # Write neurowatch_metrics.json so the dashboard updates automatically
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

    # Write HOW_TO_USE.txt so it's always clear how to activate this model
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

    return report

# =============================================================================
# -- STEP 6: PLOTS
# =============================================================================
def plot_all(train_m, val_m, test_m, out_dir=None):
    if not PLOT_OK: return

    def _save(fig, name):
        if out_dir:
            path = os.path.join(out_dir, name)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")

    # -- Figure 1: Three confusion matrices side by side -----------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("NeuroWatch  |  Confusion Matrices -- All Splits",
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
    _save(fig, "confusion_matrices.png")
    plt.show(block=False); plt.pause(0.3)

    # -- Figure 2: Metrics comparison bar -- train vs val vs test --------------
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

    # -- Figure 3: Confidence distributions -- val set per class ---------------
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

    # -- Figure 4: Overfitting check -- train vs val per metric -----------------
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
# -- FINAL SUMMARY
# =============================================================================
def print_summary(train_m, val_m, test_m, report, svm_w=0.5, rf_w=0.5):
    overfit = train_m["acc"] - val_m["acc"]
    if overfit < 0.05:   ov_msg = "OK  No overfitting"
    elif overfit < 0.12: ov_msg = "WARN  Mild overfitting -- acceptable"
    else:                ov_msg = "FAIL  High overfitting -- consider regularisation"

    print("\n" + "+" + "="*65 + "+")
    print("|" + "  TRAINING COMPLETE -- FINAL SUMMARY".center(65) + "|")
    print("+" + "="*65 + "+")
    print(f"|  {'Metric':<20} {'Train (80%)':>14} {'Val (10%)':>12} {'Test (10%)':>12} |")
    print("+" + "-"*65 + "+")
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
# -- MAIN
# =============================================================================
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

    # 1. Load and extract features
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_extract(args.data)

    # 2. Train
    scaler, selector, svm, rf, svm_w, rf_w, normal_threshold = train(
        X_train, y_train, X_val, y_val, forced_svm_w=args.svm_weight)

    # 3. Evaluate all 3 splits
    print(f"\n{'-'*65}")
    print(f"  STEP 4: Evaluating all three splits...")
    print(f"{'-'*65}")
    train_m = evaluate_split(X_train, y_train, scaler, selector, svm, rf, "Train (80%)",
                             svm_w, rf_w, normal_threshold)
    val_m   = evaluate_split(X_val,   y_val,   scaler, selector, svm, rf, "Validation (10%)",
                             svm_w, rf_w, normal_threshold)
    test_m  = evaluate_split(X_test,  y_test,  scaler, selector, svm, rf, "Test (10%)",
                             svm_w, rf_w, normal_threshold)

    # 4. Save
    print(f"\n{'-'*65}")
    print(f"  STEP 5: Saving models...")
    print(f"{'-'*65}")
    report = save_models(scaler, selector, svm, rf, train_m, val_m, test_m, args.out,
                         svm_w, rf_w, normal_threshold)

    # 5. MLflow experiment tracking
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("NeuroWatch-EEG")
        with mlflow.start_run(run_name=os.path.basename(args.out)):
            # -- Parameters
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

            # -- Metrics: train
            mlflow.log_metric("train_accuracy",      round(train_m["acc"],    4))
            mlflow.log_metric("train_f1",            round(train_m["f1"],     4))
            mlflow.log_metric("train_seizure_recall", round(train_m["sz_rec"], 4))

            # -- Metrics: validation
            mlflow.log_metric("val_accuracy",        round(val_m["acc"],    4))
            mlflow.log_metric("val_f1",              round(val_m["f1"],     4))
            mlflow.log_metric("val_seizure_recall",  round(val_m["sz_rec"], 4))

            # -- Metrics: test
            mlflow.log_metric("test_accuracy",       round(test_m["acc"],    4))
            mlflow.log_metric("test_f1",             round(test_m["f1"],     4))
            mlflow.log_metric("test_seizure_recall", round(test_m["sz_rec"], 4))

            # -- Overfitting gap (most important single number)
            mlflow.log_metric("overfitting_gap",
                              round(train_m["acc"] - val_m["acc"], 4))

            # -- Model artifacts
            mlflow.sklearn.log_model(svm, "svm_model")
            mlflow.sklearn.log_model(rf,  "rf_model")
            mlflow.sklearn.log_model(scaler, "scaler")

            # -- Training report JSON as artifact
            report_path = os.path.join(args.out, "training_report.json")
            mlflow.log_artifact(report_path)

            print(f"\n  MLflow run logged  ->  mlflow ui  (localhost:5000)")

    # 6. Summary
    print_summary(train_m, val_m, test_m, report)

    # 7. Plots
    plot_all(train_m, val_m, test_m, args.out)

    if PLOT_OK:
        print("\n  All plots open -- press Enter to close and exit...")
        input()
        plt.close("all")


if __name__ == "__main__":
    main()
