"""
threshold_comparison.py -- Before vs after threshold correction metrics
Run from repo root:
    python REPORT/threshold_comparison.py
"""

import os, sys, io, json
import joblib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from scipy.signal import welch, butter, filtfilt, resample
from scipy.stats import skew, kurtosis as _kurtosis
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

TARGET_FS = 128; MENDELEY_FS = 500; WIN_SIZE = 4097; NPERSEG = 128; N_CH = 19

def extract_247(sample):
    nyq  = 0.5 * TARGET_FS
    b, a = butter(4, [0.5/nyq, 45/nyq], btype="band")
    m = np.amax(np.abs(sample))
    if m > 0: sample = sample / m
    all_feats = []
    for ch in range(N_CH):
        sig   = sample[ch]
        n_new = int(len(sig) * TARGET_FS / MENDELEY_FS)
        sig_r = resample(sig, n_new)
        reps  = (WIN_SIZE // len(sig_r)) + 1
        sig_t = np.tile(sig_r, reps)[:WIN_SIZE]
        filt  = filtfilt(b, a, sig_t)
        f, psd = welch(filt, fs=TARGET_FS, nperseg=NPERSEG)
        bp = np.array([np.mean(psd[(f>=1)&(f<=4)]), np.mean(psd[(f>=4)&(f<=8)]),
                       np.mean(psd[(f>=8)&(f<=13)]), np.mean(psd[(f>=13)&(f<=30)]),
                       np.mean(psd[(f>=30)&(f<=45)])])
        total = bp.sum(); bp_norm = bp/total if total>0 else bp
        d1, d2 = np.diff(sig_r), np.diff(np.diff(sig_r))
        var_x=np.var(sig_r)+1e-10; var_d1=np.var(d1)+1e-10; var_d2=np.var(d2)+1e-10
        mobility = float(np.sqrt(var_d1/var_x))
        all_feats.extend(bp_norm.tolist())
        all_feats.extend([float(var_x), mobility,
                          float(np.sqrt(var_d2/var_d1)/(mobility+1e-10)),
                          float(skew(sig_r)), float(_kurtosis(sig_r)),
                          float(np.sqrt(np.mean(sig_r**2))),
                          float(((sig_r[:-1]*sig_r[1:])<0).sum()/len(sig_r)),
                          float(-np.sum((psd/(psd.sum()+1e-10))*
                                        np.log2(psd/(psd.sum()+1e-10)+1e-10))/
                                np.log2(len(psd)+1))])
    return np.array(all_feats, dtype=np.float32)

MODEL_DIR = os.path.join(ROOT, "models", "RF60+SVM40_C2")
DATA_DIR  = os.path.join(ROOT, "data", "Mendelay dataset", "Npy_files_preictal")
THETA     = 0.39
CLASS_NAMES = ["Normal", "Pre-Seizure", "Seizure"]

print("Loading models...")
scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
svm      = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
rf       = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
selector = joblib.load(os.path.join(MODEL_DIR, "selector.pkl"))

print("Loading test data...")
X_raw  = np.load(os.path.join(DATA_DIR, "x_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy")).ravel()

CACHE = os.path.join(ROOT, "_feat_cache", "x_test_247features.npy")
if os.path.exists(CACHE):
    print("  Loading cached features...")
    X_flat = np.load(CACHE)
else:
    print(f"  Extracting features from {X_raw.shape[0]} epochs...")
    X_flat = np.vstack([extract_247(X_raw[i]) for i in range(X_raw.shape[0])])
    np.save(CACHE, X_flat)

X_sc     = scaler.transform(X_flat)
X_svm_in = selector.transform(X_sc)
X_rf_in  = selector.transform(X_flat)

print("Computing probabilities...")
svm_proba = svm.predict_proba(X_svm_in)
rf_proba  = rf.predict_proba(X_rf_in)
proba     = 0.60 * rf_proba + 0.40 * svm_proba   # shape (n, 3)

# ── Without threshold correction ─────────────────────────────────────────────
y_before = np.argmax(proba, axis=1)

# ── With threshold correction ─────────────────────────────────────────────────
y_after = y_before.copy()
# override Pre-Seizure → Normal when Normal prob > theta AND raw pred != Seizure
mask = (y_before == 1) & (proba[:, 0] > THETA)
y_after[mask] = 0

def metrics(y_true, y_pred, label):
    acc  = accuracy_score(y_true, y_pred)
    wf1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    # false alarms = Normal windows predicted as Pre-Seizure
    false_alarms = cm[0, 1] if cm.shape[0] > 1 else 0
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Weighted F1    : {wf1:.4f}")
    print(f"  Normal prec    : {prec[0]:.4f}")
    print(f"  Pre-ictal rec  : {rec[1]:.4f}")
    print(f"  Seizure rec    : {rec[2]:.4f}")
    print(f"  False alarms   : {false_alarms}  (Normal→Pre-Seizure)")
    print(f"  Confusion matrix:\n{cm}")
    return dict(accuracy=round(float(acc),4), weighted_f1=round(float(wf1),4),
                normal_precision=round(float(prec[0]),4),
                preictal_recall=round(float(rec[1]),4),
                seizure_recall=round(float(rec[2]),4),
                false_alarms=int(false_alarms))

r_before = metrics(y_test, y_before, "WITHOUT threshold correction")
r_after  = metrics(y_test, y_after,  "WITH threshold correction (theta=0.39)")

out = {"without": r_before, "with": r_after}
out_path = os.path.join(ROOT, "REPORT", "figures", "threshold_comparison.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")
