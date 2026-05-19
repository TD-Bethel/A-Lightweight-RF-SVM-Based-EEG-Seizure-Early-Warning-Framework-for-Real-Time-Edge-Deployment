"""
ablation_feature_k.py — RF60+SVM40_C2 ablation over feature-count k.

Trains the final model configuration for k in {25, 50, 75, 100, 247} and
reports validation-set Accuracy, Weighted F1, and Seizure Recall for each k.
Same preprocessing, split, and hyperparameters as the production model.
"""

import os, sys
import numpy as np
from scipy.signal import butter, filtfilt, welch, resample
from scipy.stats import skew, kurtosis as scipy_kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# ── Constants (match train_mendeley.py exactly) ───────────────────────────────
FS_MENDELEY = 500
TARGET_FS   = 128
WIN_SIZE    = 4097
NPERSEG     = 128
N_ORDER     = 4
F_LOW, F_HIGH = 0.5, 45.0
N_CHANNELS  = 19
RANDOM_SEED = 42
LABEL_MAP   = {0: 0, 1: 1, 2: 2, 3: 1}

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "Mendelay dataset", "Npy_files"
)

# ── Signal processing helpers ─────────────────────────────────────────────────
def _bandpass(sig, fs=TARGET_FS):
    nyq  = 0.5 * fs
    b, a = butter(N_ORDER, [F_LOW / nyq, F_HIGH / nyq], btype="band")
    return filtfilt(b, a, sig)

def _band_power(sig, fs=TARGET_FS):
    filt   = _bandpass(sig, fs)
    f, psd = welch(filt, fs=fs, nperseg=NPERSEG)
    return np.array([
        np.mean(psd[(f >= 1)  & (f <= 4)]),
        np.mean(psd[(f >= 4)  & (f <= 8)]),
        np.mean(psd[(f >= 8)  & (f <= 13)]),
        np.mean(psd[(f >= 13) & (f <= 30)]),
        np.mean(psd[(f >= 30) & (f <= 45)]),
    ])

def _normalise(v):
    t = v.sum(); return v / t if t > 0 else v

def _hjorth(sig):
    d1 = np.diff(sig); d2 = np.diff(d1)
    vx = np.var(sig) + 1e-10; vd1 = np.var(d1) + 1e-10; vd2 = np.var(d2) + 1e-10
    mob = float(np.sqrt(vd1 / vx))
    return float(vx), mob, float(np.sqrt(vd2 / vd1) / (mob + 1e-10))

def _spectral_entropy(psd):
    p = psd / (psd.sum() + 1e-10)
    return float(-np.sum(p * np.log2(p + 1e-10)) / np.log2(len(p) + 1))

def _zcr(sig):
    return float(((sig[:-1] * sig[1:]) < 0).sum() / len(sig))

def extract_features(sample):
    feats = []
    for ch in range(N_CHANNELS):
        sig   = sample[ch]
        n_new = int(len(sig) * TARGET_FS / FS_MENDELEY)
        sig_r = resample(sig, n_new)
        reps  = (WIN_SIZE // len(sig_r)) + 1
        sig_t = np.tile(sig_r, reps)[:WIN_SIZE]
        bp    = _normalise(_band_power(sig_t))
        act, mob, comp = _hjorth(sig_r)
        filt  = _bandpass(sig_t, TARGET_FS)
        _, psd = welch(filt, fs=TARGET_FS, nperseg=NPERSEG)
        feats.extend(bp.tolist())
        feats.extend([act, mob, comp,
                      float(skew(sig_r)), float(scipy_kurtosis(sig_r)),
                      float(np.sqrt(np.mean(sig_r**2))), _zcr(sig_r),
                      _spectral_entropy(psd)])
    return np.array(feats, dtype=np.float32)

# ── Load & split data ─────────────────────────────────────────────────────────
print("Loading NPY files...")
X_all = np.concatenate([np.load(os.path.join(DATA_DIR, "x_train.npy")),
                         np.load(os.path.join(DATA_DIR, "x_test.npy"))], axis=0)
y_raw = np.concatenate([np.load(os.path.join(DATA_DIR, "y_train.npy")),
                         np.load(os.path.join(DATA_DIR, "y_test.npy"))], axis=0)
y_all = np.array([LABEL_MAP[int(l)] for l in y_raw])
print(f"  Total samples: {len(X_all)}")

X_tr_raw, X_tmp, y_tr, y_tmp = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=RANDOM_SEED)
X_val_raw, _, y_val, _ = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_SEED)

print("Extracting features (train)...")
X_tr  = np.array([extract_features(s) for s in X_tr_raw])
print("Extracting features (val)...")
X_val = np.array([extract_features(s) for s in X_val_raw])
print(f"  Feature matrix: train={X_tr.shape}, val={X_val.shape}")

# ── Ablation loop ─────────────────────────────────────────────────────────────
K_VALUES = [25, 50, 75, 100, 247]
print("\n{:<6}  {:>10}  {:>12}  {:>14}".format(
    "k", "Val Acc %", "Weighted F1", "Seizure Recall"))
print("-" * 48)

results = []
for k in K_VALUES:
    # Feature selection via RF importance
    scout = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    scout.fit(X_tr, y_tr)
    top_idx = np.argsort(scout.feature_importances_)[::-1][:k]
    top_idx = np.sort(top_idx)

    X_tr_sel  = X_tr[:, top_idx]
    X_val_sel = X_val[:, top_idx]

    # Scale
    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_tr_sel)
    X_val_sc  = scaler.transform(X_val_sel)

    # SVM
    svm = SVC(kernel="rbf", C=2, gamma="scale", probability=True,
              class_weight={0: 1.0, 1: 1.8, 2: 1.0}, random_state=RANDOM_SEED)
    svm.fit(X_tr_sc, y_tr)

    # RF
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=5, min_samples_leaf=10,
        max_features="sqrt", class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_tr_sel, y_tr)

    # Blend (SVM 40%, RF 60%)
    p_svm = svm.predict_proba(X_val_sc)
    p_rf  = rf.predict_proba(X_val_sel)
    p_comb = 0.40 * p_svm + 0.60 * p_rf
    y_pred = np.argmax(p_comb, axis=1)

    acc   = accuracy_score(y_val, y_pred) * 100
    wf1   = f1_score(y_val, y_pred, average="weighted") * 100
    # Seizure recall: class 2
    mask  = y_val == 2
    sz_rec = (y_pred[mask] == 2).sum() / mask.sum() * 100 if mask.sum() > 0 else 0.0

    results.append((k, acc, wf1, sz_rec))
    print(f"{k:<6}  {acc:>9.1f}  {wf1:>11.1f}  {sz_rec:>13.1f}")

print("\nDone.")
