"""
eval_preictal_backup.py  --  Evaluate MODELS_PREICTAL_BACKUP on the Mendeley test split

Loads the backup pre-ictal models (~83% test accuracy) and re-runs them against
the same Mendeley npy data used during training, producing a full metrics report.

USAGE:
  python eval_preictal_backup.py
  python eval_preictal_backup.py --data "path/to/Npy_files" --model-dir "path/to/models"
"""

import os, sys, argparse, warnings, time
import numpy as np
import joblib

warnings.filterwarnings("ignore")

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
from neurowatch_selector import RFImportanceSelector  # needed for joblib unpickling
DEFAULT_DATA = os.path.join(BASE_DIR, "data", "Mendelay dataset", "Npy_files_preictal")
DEFAULT_MDL  = os.path.join(BASE_DIR, "models", "MODELS_FS75")
RANDOM_SEED  = 42

ap = argparse.ArgumentParser()
ap.add_argument("--data",      default=DEFAULT_DATA)
ap.add_argument("--model-dir", default=DEFAULT_MDL)
ap.add_argument("--no-plots",  action="store_true")
args = ap.parse_args()

CLASSES    = ["Normal", "Pre-Seizure", "Seizure"]
LABEL_MAP  = {0: 0, 1: 1, 2: 2, 3: 1}
LABEL_NAMES = {0:"Normal", 1:"CPS", 2:"Electrographic", 3:"Video-detected"}

# ── Feature extraction (identical to train_mendeley.py) ────────────────────────
from scipy.signal import butter, filtfilt, welch, resample
from scipy.stats import skew, kurtosis as scipy_kurtosis

TARGET_FS = 128
FS_SRC    = 500
WIN_SIZE  = 4097
N_BANDS   = 5
N_STATS   = 8
N_CH      = 19
N_CORR    = N_CH * (N_CH - 1) // 2
NPERSEG   = 128

def _bandpass(sig, fs=TARGET_FS):
    nyq  = 0.5 * fs
    b, a = butter(4, [0.5/nyq, 45.0/nyq], btype="band")
    return filtfilt(b, a, sig)

def _band_power(sig):
    filt   = _bandpass(sig)
    f, psd = welch(filt, fs=TARGET_FS, nperseg=NPERSEG)
    return np.array([
        np.mean(psd[(f>=1)  & (f<=4)]),
        np.mean(psd[(f>=4)  & (f<=8)]),
        np.mean(psd[(f>=8)  & (f<=13)]),
        np.mean(psd[(f>=13) & (f<=30)]),
        np.mean(psd[(f>=30) & (f<=45)]),
    ])

def _hjorth(sig):
    d1    = np.diff(sig)
    d2    = np.diff(d1)
    vx    = np.var(sig)+1e-10; vd1 = np.var(d1)+1e-10; vd2 = np.var(d2)+1e-10
    mob   = float(np.sqrt(vd1/vx))
    return float(vx), mob, float(np.sqrt(vd2/vd1)/(mob+1e-10))

def _spectral_entropy(psd):
    p = psd/(psd.sum()+1e-10)
    return float(-np.sum(p*np.log2(p+1e-10))/np.log2(len(p)+1))

def _zcr(sig):
    return float(((sig[:-1]*sig[1:])<0).sum()/len(sig))

def extract_features(sample):
    per_ch = []; resampled = []
    for ch in range(N_CH):
        sig   = sample[ch]
        n_new = int(len(sig)*TARGET_FS/FS_SRC)
        sig_r = resample(sig, n_new)
        reps  = (WIN_SIZE//len(sig_r))+1
        sig_t = np.tile(sig_r, reps)[:WIN_SIZE]
        resampled.append(sig_r)
        bp      = _band_power(sig_t)
        bp_norm = bp/(bp.sum()+1e-10)
        act, mob, comp = _hjorth(sig_r)
        filt   = _bandpass(sig_t)
        _, psd = welch(filt, fs=TARGET_FS, nperseg=NPERSEG)
        per_ch.extend(bp_norm.tolist())
        per_ch.extend([act, mob, comp,
                        float(skew(sig_r)), float(scipy_kurtosis(sig_r)),
                        float(np.sqrt(np.mean(sig_r**2))), _zcr(sig_r),
                        _spectral_entropy(psd)])
    return np.array(per_ch, dtype=np.float32)

# ── Load models ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  NeuroWatch -- Pre-ictal Backup Model Evaluation")
print(f"{'='*60}")
print(f"  Model dir : {args.model_dir}")
print(f"  Data dir  : {args.data}")

def require(path):
    if not os.path.exists(path):
        print(f"  FAIL: not found: {path}"); sys.exit(1)
    return path

scaler   = joblib.load(require(os.path.join(args.model_dir, "scaler.pkl")))
svm      = joblib.load(require(os.path.join(args.model_dir, "svm_model.pkl")))
rf       = joblib.load(require(os.path.join(args.model_dir, "rf_model.pkl")))
ew       = joblib.load(require(os.path.join(args.model_dir, "ensemble_weights.pkl")))
SVM_W    = ew["svm_w"]; RF_W = ew["rf_w"]
NORMAL_THRESHOLD = float(ew.get("normal_threshold", 0.50))

selector_path = os.path.join(args.model_dir, "selector.pkl")
selector = joblib.load(selector_path) if os.path.exists(selector_path) else None

print(f"\n  Models loaded OK")
print(f"  Ensemble       : SVM {SVM_W:.0%} + RF {RF_W:.0%}")
print(f"  Normal threshold: {NORMAL_THRESHOLD:.2f}")
if selector is not None:
    print(f"  Feature selector: {selector.k} features selected")
else:
    print(f"  Feature selector: none")

# ── Load + split data (same seed as training) ──────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

def load_npy(name):
    p = os.path.join(args.data, name)
    if not os.path.exists(p): return None
    return np.load(p)

Xtr = load_npy("x_train.npy"); Ytr = load_npy("y_train.npy")
Xte = load_npy("x_test.npy");  Yte = load_npy("y_test.npy")

parts_x = [x for x in [Xtr, Xte] if x is not None]
parts_y = [y for y in [Ytr, Yte] if y is not None]
if not parts_x:
    print(f"  FAIL: no npy files in {args.data}"); sys.exit(1)

X_all = np.concatenate(parts_x); y_all = np.concatenate(parts_y)
print(f"\n  Dataset: {len(X_all)} samples, shape per sample: {X_all[0].shape}")

X_train_r, X_temp, y_train_r, y_temp = train_test_split(
    X_all, y_all, test_size=0.20, random_state=RANDOM_SEED, stratify=y_all)
X_val_r, X_test_r, y_val_r, y_test_r = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp)

print(f"  Split   : train={len(X_train_r)}  val={len(X_val_r)}  test={len(X_test_r)}")

# ── Feature extraction ─────────────────────────────────────────────────────────
def extract_split(X_raw, y_raw, tag):
    feats = []; labels = []; n = len(X_raw); t0 = time.time()
    for i, (s, l) in enumerate(zip(X_raw, y_raw)):
        if i % 200 == 0:
            pct = i/n*100
            print(f"  [{('#'*int(pct/5)):20s}] {pct:5.1f}%  {tag} ({i}/{n})", end="\r")
        feats.append(extract_features(s))
        labels.append(LABEL_MAP[int(l)])
    elapsed = time.time()-t0
    print(f"  [{'#'*20}] 100.0%  {tag} done ({elapsed:.0f}s)        ")
    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int32)

CACHE_DIR = os.path.join(BASE_DIR, "_feat_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def cached_extract(X_raw, y_raw, tag):
    key  = f"{tag}_{len(X_raw)}_{RANDOM_SEED}"
    cx   = os.path.join(CACHE_DIR, f"{key}_X.npy")
    cy   = os.path.join(CACHE_DIR, f"{key}_y.npy")
    if os.path.exists(cx) and os.path.exists(cy):
        print(f"  [cache] Loading {tag} features from disk...")
        return np.load(cx), np.load(cy)
    F, L = extract_split(X_raw, y_raw, tag)
    np.save(cx, F); np.save(cy, L)
    return F, L

print(f"\n  Extracting features (cached after first run)...")
X_test, y_test   = cached_extract(X_test_r,  y_test_r,  "Test ")

# ── Predict ────────────────────────────────────────────────────────────────────
def predict(X, y_true, tag):
    X_sc = scaler.transform(X)
    if selector is not None:
        X_svm = selector.transform(X_sc)
        X_rf  = selector.transform(X)
    else:
        X_svm = X_sc
        X_rf  = X
    p_svm    = svm.predict_proba(X_svm)
    p_rf     = rf.predict_proba(X_rf)
    combined = SVM_W * p_svm + RF_W * p_rf
    argmax   = np.argmax(combined, axis=1)
    y_pred   = argmax.copy()
    y_pred[(argmax == 1) & (combined[:, 0] >= NORMAL_THRESHOLD)] = 0

    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm    = confusion_matrix(y_true, y_pred)
    sz_r  = recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0) \
            if 2 in y_true else 0.0

    present = sorted(np.unique(np.concatenate([y_true, y_pred])))
    w = 14
    print(f"\n  {'─'*55}")
    print(f"  {tag}")
    print(f"  {'─'*55}")
    print(f"  Samples        : {len(y_true)}")
    print(f"  Accuracy       : {acc*100:.1f}%  {'OK' if acc>=0.80 else 'WARN'}")
    print(f"  F1 (weighted)  : {f1*100:.1f}%")
    print(f"  Seizure Recall : {sz_r*100:.1f}%")
    print(f"\n  Confusion Matrix  (rows=True, cols=Predicted):")
    print(f"  {'':>{w}}", end="")
    for i in present: print(f"  {CLASSES[i]:>{w}}", end="")
    print()
    for i, row in zip(present, cm):
        print(f"  {CLASSES[i]:>{w}}", end="")
        for v in row: print(f"  {v:>{w}d}", end="")
        print()
    print(f"\n  Per-class (test set):")
    for i in present:
        mask = y_true == i
        if not mask.any(): continue
        p = precision_score(y_true==i, y_pred==i, zero_division=0)
        r = recall_score(y_true==i,    y_pred==i, zero_division=0)
        f = f1_score(y_true==i,        y_pred==i, zero_division=0)
        print(f"    {CLASSES[i]:<14}  n={mask.sum():4d}  "
              f"P={p:.2f}  R={r:.2f}  F1={f:.2f}")
    return y_pred, combined, acc, f1

y_pred_test, confs_test, acc_test, f1_test = predict(X_test, y_test, "TEST SET")

print(f"\n{'='*60}")
print(f"  Summary: Test accuracy={acc_test*100:.1f}%  F1={f1_test*100:.1f}%")
print(f"{'='*60}\n")

# ── Optional plots ─────────────────────────────────────────────────────────────
if not args.no_plots:
    try:
        import matplotlib; matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.rcParams.update({
            "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
            "axes.edgecolor": "#444", "axes.labelcolor": "#ccc",
            "xtick.color": "#aaa", "ytick.color": "#aaa",
            "text.color": "#eee", "grid.color": "#2a2a4a",
        })
        present = sorted(np.unique(np.concatenate([y_test, y_pred_test])))
        cls_names = [CLASSES[i] for i in present]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("MODELS_PREICTAL_BACKUP -- Test Set Evaluation", fontsize=13, fontweight="bold")

        cm = confusion_matrix(y_test, y_pred_test)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=cls_names, yticklabels=cls_names,
                    cbar=False, ax=axes[0], square=True, annot_kws={"size": 13})
        axes[0].set_title(f"Confusion Matrix\nAcc: {acc_test*100:.1f}%  F1: {f1_test*100:.1f}%")
        axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

        avg_conf = confs_test.mean(axis=0)
        axes[1].bar(CLASSES[:len(avg_conf)], avg_conf,
                    color=["#00CC66","#FFD633","#FF3333"], edgecolor="#333", alpha=0.9)
        axes[1].set_title("Average Model Confidence Per Class")
        axes[1].set_ylabel("Mean Probability"); axes[1].set_ylim(0, 1)
        for i, v in enumerate(avg_conf):
            axes[1].text(i, v+0.02, f"{v*100:.1f}%", ha="center", fontsize=11)

        plt.tight_layout()
        plt.show()
    except ImportError as e:
        print(f"  Plots skipped: {e}")
