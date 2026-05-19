"""
sweep_weights.py  --  Evaluate SVM/RF ensemble across weight pairs
Outputs a LaTeX table + JSON results for the report.

Run from repo root:
    python REPORT/sweep_weights.py
"""

import os, sys, json, io
import joblib

# fix Windows CP1252 console so emoji in imported modules don't crash
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from scipy.signal import welch, butter, filtfilt, resample
from scipy.stats import skew, kurtosis as _kurtosis

TARGET_FS   = 128
MENDELEY_FS = 500
WIN_SIZE    = 4097
NPERSEG     = 128
N_CH        = 19

def extract_247(sample):
    """247-feature extractor matching main_pi_bios_v15 / training pipeline."""
    nyq  = 0.5 * TARGET_FS
    b, a = butter(4, [0.5 / nyq, 45 / nyq], btype="band")
    m = np.amax(np.abs(sample))
    if m > 0:
        sample = sample / m
    all_feats = []
    for ch in range(N_CH):
        sig   = sample[ch]
        n_new = int(len(sig) * TARGET_FS / MENDELEY_FS)
        sig_r = resample(sig, n_new)
        reps  = (WIN_SIZE // len(sig_r)) + 1
        sig_t = np.tile(sig_r, reps)[:WIN_SIZE]
        filt  = filtfilt(b, a, sig_t)
        f, psd = welch(filt, fs=TARGET_FS, nperseg=NPERSEG)
        bp = np.array([
            np.mean(psd[(f >= 1)  & (f <= 4)]),
            np.mean(psd[(f >= 4)  & (f <= 8)]),
            np.mean(psd[(f >= 8)  & (f <= 13)]),
            np.mean(psd[(f >= 13) & (f <= 30)]),
            np.mean(psd[(f >= 30) & (f <= 45)]),
        ])
        total   = bp.sum()
        bp_norm = bp / total if total > 0 else bp
        d1, d2   = np.diff(sig_r), np.diff(np.diff(sig_r))
        var_x    = np.var(sig_r) + 1e-10
        var_d1   = np.var(d1)    + 1e-10
        var_d2   = np.var(d2)    + 1e-10
        activity   = float(var_x)
        mobility   = float(np.sqrt(var_d1 / var_x))
        complexity = float(np.sqrt(var_d2 / var_d1) / (mobility + 1e-10))
        sk   = float(skew(sig_r))
        kurt = float(_kurtosis(sig_r))
        rms  = float(np.sqrt(np.mean(sig_r ** 2)))
        zcr  = float(((sig_r[:-1] * sig_r[1:]) < 0).sum() / len(sig_r))
        p_n  = psd / (psd.sum() + 1e-10)
        sent = float(-np.sum(p_n * np.log2(p_n + 1e-10)) / np.log2(len(p_n) + 1))
        all_feats.extend(bp_norm.tolist())
        all_feats.extend([activity, mobility, complexity, sk, kurt, rms, zcr, sent])
    return np.array(all_feats, dtype=np.float32)
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix)

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(ROOT, "models", "RF60+SVM40_C2")
DATA_DIR  = os.path.join(ROOT, "data", "Mendelay dataset", "Npy_files_preictal")
OUT_DIR   = os.path.join(ROOT, "REPORT", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models …")
scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
svm      = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
rf       = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
selector = joblib.load(os.path.join(MODEL_DIR, "selector.pkl"))

# ── Load test data ────────────────────────────────────────────────────────────
print("Loading test data …")
X_raw  = np.load(os.path.join(DATA_DIR, "x_test.npy"))   # (n, 19, 500)
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy")).ravel()

# Extract 247 engineered features per epoch (band-powers, Hjorth, etc.)
CACHE = os.path.join(ROOT, "_feat_cache", "x_test_247features.npy")
os.makedirs(os.path.dirname(CACHE), exist_ok=True)
if os.path.exists(CACHE):
    print("  Loading cached 247-features …")
    X_flat = np.load(CACHE)
else:
    print(f"  Extracting 247 features from {X_raw.shape[0]} epochs …")
    X_flat = np.vstack([extract_247(X_raw[i]) for i in range(X_raw.shape[0])
                        if not print(f"    {i+1}/{X_raw.shape[0]}", end="\r")
                        or True])
    np.save(CACHE, X_flat)
    print(f"\n  Cached to {CACHE}")
print(f"  Feature matrix: {X_flat.shape}")

# Pipeline order matches main_pi_bios_v15.py:
#   1. scale full 247-feature matrix
#   2. selector on scaled  → SVM input (75 features)
#   3. selector on raw     → RF  input (75 features)
X_sc      = scaler.transform(X_flat)          # (n, 247) scaled
X_svm_in  = selector.transform(X_sc)          # (n, 75)  scaled, for SVM
X_rf_in   = selector.transform(X_flat)        # (n, 75)  raw,    for RF

print(f"Test set: {X_flat.shape[0]} samples  |  classes: {np.unique(y_test, return_counts=True)}")

# ── Load training data (for overfitting gap) ──────────────────────────────────
print("\nLoading training data …")
X_tr_raw = np.load(os.path.join(DATA_DIR, "x_train.npy"))
y_train  = np.load(os.path.join(DATA_DIR, "y_train.npy")).ravel()

CACHE_TR = os.path.join(ROOT, "_feat_cache", "x_train_247features.npy")
if os.path.exists(CACHE_TR):
    print("  Loading cached training 247-features …")
    X_tr_flat = np.load(CACHE_TR)
else:
    print(f"  Extracting 247 features from {X_tr_raw.shape[0]} epochs …")
    X_tr_flat = np.vstack([extract_247(X_tr_raw[i]) for i in range(X_tr_raw.shape[0])
                           if not print(f"    {i+1}/{X_tr_raw.shape[0]}", end="\r")
                           or True])
    np.save(CACHE_TR, X_tr_flat)
    print(f"\n  Cached to {CACHE_TR}")
print(f"  Train feature matrix: {X_tr_flat.shape}")

X_tr_sc     = scaler.transform(X_tr_flat)
X_tr_svm_in = selector.transform(X_tr_sc)
X_tr_rf_in  = selector.transform(X_tr_flat)

print("Computing training SVM probabilities …")
svm_tr_proba = svm.predict_proba(X_tr_svm_in)
print("Computing training RF probabilities …")
rf_tr_proba  = rf.predict_proba(X_tr_rf_in)

# ── Weight pairs to evaluate ─────────────────────────────────────────────────
# (rf_weight, svm_weight)
PAIRS = [
    (0.70, 0.30),
    (0.60, 0.40),   # current production setting
    (0.50, 0.50),
    (0.40, 0.60),
    (0.30, 0.70),
]
CLASS_NAMES = ["Normal", "Pre-Seizure", "Seizure"]

# ── Pre-compute raw probabilities (expensive — do once) ──────────────────────
print("Computing SVM probabilities …")
svm_proba = svm.predict_proba(X_svm_in)      # SVM: scaled + selected
print("Computing RF probabilities …")
rf_proba  = rf.predict_proba(X_rf_in)        # RF:  raw + selected

# ── Optional MLflow tracking (silent if mlflow not installed) ─────────────────
try:
    import mlflow
    mlflow.set_tracking_uri(f"file:{os.path.join(ROOT, 'mlruns')}")
    mlflow.set_experiment("neurowatch_weight_sweep")
    _MLFLOW_OK = True
except ImportError:
    _MLFLOW_OK = False

# ── Sweep ─────────────────────────────────────────────────────────────────────
results = []
print("\n{:<16} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
    "RF/SVM", "Tr-Acc", "Te-Acc", "Gap", "Macro-F1", "Sz-Rec"))
print("-" * 70)

for rf_w, svm_w in PAIRS:
    # ── Test metrics ──
    proba  = rf_w * rf_proba + svm_w * svm_proba
    y_pred = np.argmax(proba, axis=1)
    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    prec     = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec      = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_cls   = f1_score(y_test, y_pred, average=None, zero_division=0)
    cm       = confusion_matrix(y_test, y_pred).tolist()

    # ── Training metrics (for overfitting gap) ──
    tr_proba  = rf_w * rf_tr_proba + svm_w * svm_tr_proba
    tr_pred   = np.argmax(tr_proba, axis=1)
    tr_acc    = accuracy_score(y_train, tr_pred)
    tr_f1     = f1_score(y_train, tr_pred, average="macro", zero_division=0)
    gap       = tr_acc - acc

    tag = f"RF{int(rf_w*100)}/SVM{int(svm_w*100)}"
    print(f"{tag:<16} {tr_acc:>8.4f}  {acc:>8.4f}  {gap:>8.4f}  "
          f"{macro_f1:>8.4f}  {rec[2]:>8.4f}")

    results.append({
        "rf_weight": rf_w,
        "svm_weight": svm_w,
        "label": f"RF {int(rf_w*100)} / SVM {int(svm_w*100)}",
        "train_accuracy": round(float(tr_acc), 4),
        "train_macro_f1": round(float(tr_f1), 4),
        "overfitting_gap": round(float(gap), 4),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": {
            cn: {
                "precision": round(float(prec[i]), 4),
                "recall":    round(float(rec[i]),  4),
                "f1":        round(float(f1_cls[i]), 4),
            }
            for i, cn in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": cm,
        "seizure_recall":    round(float(rec[2]), 4),
        "preictal_recall":   round(float(rec[1]), 4),
    })

    # ── MLflow: log this weight pair as its own run ──────────────────────────
    if _MLFLOW_OK:
        try:
            with mlflow.start_run(run_name=tag):
                mlflow.log_params({"rf_weight": rf_w, "svm_weight": svm_w})
                mlflow.log_metric("train_accuracy",  float(tr_acc))
                mlflow.log_metric("test_accuracy",   float(acc))
                mlflow.log_metric("overfitting_gap", float(gap))
                mlflow.log_metric("macro_f1",        float(macro_f1))
                mlflow.log_metric("seizure_recall",  float(rec[2]))
                mlflow.log_metric("preictal_recall", float(rec[1]))
        except Exception as e:
            print(f"  WARN  MLflow logging failed for {tag}: {e}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
json_path = os.path.join(OUT_DIR, "weight_sweep_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved JSON → {json_path}")

# ── Print LaTeX table ──────────────────────────────────────────────────────────
print("\n% ── LaTeX table ──────────────────────────────────────────")
print(r"\begin{table}[ht]")
print(r"\centering")
print(r"\caption{Validation metrics for the hybrid SVM\,+\,RF ensemble")
print(r"         across five weight configurations on the held-out test set.}")
print(r"\label{tab:weight_sweep}")
print(r"\small")
print(r"\begin{tabularx}{0.98\linewidth}{l *{6}{>{\centering\arraybackslash}X}}")
print(r"\toprule")
print(r"\textbf{RF\,/\,SVM} & \textbf{Train Acc.} & \textbf{Test Acc.} "
      r"& \textbf{Gap} & \textbf{Macro-F1} "
      r"& \textbf{Pre-ictal F1} & \textbf{Seizure F1} \\")
print(r"\midrule")

best_acc = max(r["accuracy"]  for r in results)
best_f1  = max(r["macro_f1"] for r in results)
min_gap  = min(r["overfitting_gap"] for r in results)

for r in results:
    tag  = r["label"]
    tacc = r["train_accuracy"]
    acc  = r["accuracy"]
    gap  = r["overfitting_gap"]
    mf1  = r["macro_f1"]
    pf1  = r["per_class"]["Pre-Seizure"]["f1"]
    sf1  = r["per_class"]["Seizure"]["f1"]
    prod = r["rf_weight"] == 0.60

    acc_s = f"\\textbf{{{acc:.4f}}}" if acc == best_acc else f"{acc:.4f}"
    mf1_s = f"\\textbf{{{mf1:.4f}}}" if mf1 == best_f1  else f"{mf1:.4f}"
    gap_s = f"\\textbf{{{gap:.4f}}}" if gap == min_gap   else f"{gap:.4f}"
    row = (f"{tag} & {tacc:.4f} & {acc_s} & {gap_s} & {mf1_s} "
           f"& {pf1:.4f} & {sf1:.4f} \\\\")
    if prod:
        row += "  % production"
    print(row)

print(r"\bottomrule")
print(r"\end{tabularx}")
print(r"\end{table}")
