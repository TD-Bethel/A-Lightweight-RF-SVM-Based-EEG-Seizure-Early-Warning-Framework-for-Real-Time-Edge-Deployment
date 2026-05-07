"""
quick_viz.py  --  Lightweight prediction visualizer (no Streamlit)

Loads MODELS_FS75, runs predictions on the preictal test split,
and shows a 4-panel matplotlib figure instantly.

Uses the feature cache from eval_preictal_backup.py if it exists
(avoids re-extracting features). Otherwise extracts a small subset.

USAGE:
  python scripts/inference/quick_viz.py
  python scripts/inference/quick_viz.py --n 500
  python scripts/inference/quick_viz.py --model-dir "models/MODELS_FS75"
"""

import os, sys, argparse, warnings
import numpy as np
import joblib
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from neurowatch_selector import RFImportanceSelector  # needed for unpickling

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CLASSES = ["Normal", "Pre-Seizure", "Seizure"]
COLORS  = ["#00CC66", "#FFD633", "#FF3333"]
SEED    = 42

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default=os.path.join(PROJECT_ROOT, "models", "MODELS_FS75"))
ap.add_argument("--data",      default=os.path.join(PROJECT_ROOT, "data", "Mendelay dataset", "Npy_files_preictal"))
ap.add_argument("--n",         type=int, default=400, help="Max windows to show (default 400)")
args = ap.parse_args()

# ── Load models ────────────────────────────────────────────────────────────────
print(f"\nLoading models from: {args.model_dir}")
scaler   = joblib.load(os.path.join(args.model_dir, "scaler.pkl"))
svm      = joblib.load(os.path.join(args.model_dir, "svm_model.pkl"))
rf       = joblib.load(os.path.join(args.model_dir, "rf_model.pkl"))
ew       = joblib.load(os.path.join(args.model_dir, "ensemble_weights.pkl"))
SVM_W    = float(ew["svm_w"]); RF_W = float(ew["rf_w"])
THRESH   = float(ew.get("normal_threshold", 0.50))
sel_path = os.path.join(args.model_dir, "selector.pkl")
selector = joblib.load(sel_path) if os.path.exists(sel_path) else None
print(f"SVM {SVM_W:.0%} + RF {RF_W:.0%}  |  threshold={THRESH:.2f}  |  "
      f"selector={'yes (%d feats)' % selector.k if selector else 'no'}")

# ── Load features (from cache or extract small subset) ────────────────────────
from sklearn.model_selection import train_test_split

def load_npy(name):
    p = os.path.join(args.data, name)
    return np.load(p) if os.path.exists(p) else None

CACHE_DIR = os.path.join(PROJECT_ROOT, "_feat_cache")
X_feat = y_true = None

# Try loading from the eval cache first (same seed/split)
Xtr = load_npy("x_train.npy"); Ytr = load_npy("y_train.npy")
Xte = load_npy("x_test.npy");  Yte = load_npy("y_test.npy")
parts_x = [x for x in [Xtr, Xte] if x is not None]
parts_y = [y for y in [Ytr, Yte] if y is not None]

if not parts_x:
    print(f"ERROR: no npy files found in {args.data}"); sys.exit(1)

X_all = np.concatenate(parts_x); y_all = np.concatenate(parts_y)
LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 1}

_, X_temp, _, y_temp = train_test_split(
    X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all)
_, X_test_r, _, y_test_r = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)
y_test_r = np.array([LABEL_MAP[int(l)] for l in y_test_r])

# Check for cached features from eval_preictal_backup.py
cache_key = f"Test _{len(X_test_r)}_{SEED}"
cx = os.path.join(CACHE_DIR, f"{cache_key}_X.npy")
cy = os.path.join(CACHE_DIR, f"{cache_key}_y.npy")

if os.path.exists(cx) and os.path.exists(cy):
    print("Found feature cache — loading instantly...")
    X_feat = np.load(cx)
    y_true = np.load(cy)
else:
    # Extract features for a small subset quickly
    print(f"No cache found. Extracting features for {min(args.n, len(X_test_r))} windows...")
    from scipy.signal import butter, filtfilt, welch, resample as sp_resample
    from scipy.stats import skew, kurtosis as sp_kurtosis

    TARGET_FS = 128; FS_SRC = 500; WIN_SIZE = 4097; NPERSEG = 128

    def _bp(sig):
        nyq = 0.5*TARGET_FS; b,a = butter(4,[0.5/nyq,45./nyq],btype="band")
        return filtfilt(b,a,sig)
    def _band_power(sig):
        f,p = welch(_bp(sig),fs=TARGET_FS,nperseg=NPERSEG)
        return np.array([p[(f>=1)&(f<=4)].mean(),p[(f>=4)&(f<=8)].mean(),
                         p[(f>=8)&(f<=13)].mean(),p[(f>=13)&(f<=30)].mean(),
                         p[(f>=30)&(f<=45)].mean()])
    def _hjorth(s):
        d1=np.diff(s);d2=np.diff(d1)
        vx=np.var(s)+1e-10;vd1=np.var(d1)+1e-10;vd2=np.var(d2)+1e-10
        mob=float(np.sqrt(vd1/vx))
        return float(vx),mob,float(np.sqrt(vd2/vd1)/(mob+1e-10))
    def _sent(psd):
        p=psd/(psd.sum()+1e-10)
        return float(-np.sum(p*np.log2(p+1e-10))/np.log2(len(p)+1))
    def _zcr(s): return float(((s[:-1]*s[1:])<0).sum()/len(s))

    def extract(sample):
        feats=[]
        for ch in range(19):
            sig=sample[ch]; n_new=int(len(sig)*TARGET_FS/FS_SRC)
            sr=sp_resample(sig,n_new); reps=(WIN_SIZE//len(sr))+1
            st=np.tile(sr,reps)[:WIN_SIZE]
            bp=_band_power(st); bp_n=bp/(bp.sum()+1e-10)
            act,mob,comp=_hjorth(sr)
            _,psd=welch(_bp(st),fs=TARGET_FS,nperseg=NPERSEG)
            feats.extend(bp_n.tolist())
            feats.extend([act,mob,comp,float(skew(sr)),float(sp_kurtosis(sr)),
                          float(np.sqrt(np.mean(sr**2))),_zcr(sr),_sent(psd)])
        return np.array(feats,dtype=np.float32)

    n_extract = min(args.n, len(X_test_r))
    X_feat = np.array([extract(X_test_r[i]) for i in range(n_extract)], dtype=np.float32)
    y_true = y_test_r[:n_extract]
    print("Done.")

# ── Trim to requested N ────────────────────────────────────────────────────────
n = min(args.n, len(X_feat))
X_feat = X_feat[:n]; y_true = y_true[:n]

# ── Predict ────────────────────────────────────────────────────────────────────
print(f"Running predictions on {n} windows...")
X_sc = scaler.transform(X_feat)
if selector is not None:
    X_svm = selector.transform(X_sc)
    X_rf  = selector.transform(X_feat)
else:
    X_svm = X_sc; X_rf = X_feat

confs  = SVM_W * svm.predict_proba(X_svm) + RF_W * rf.predict_proba(X_rf)
argmax = np.argmax(confs, axis=1)
y_pred = argmax.copy()
y_pred[(argmax == 1) & (confs[:, 0] >= THRESH)] = 0

correct = y_pred == y_true
acc     = correct.mean()
print(f"Accuracy: {acc*100:.1f}%  |  Windows: {n}")

# ── Plot ───────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
    "axes.edgecolor": "#444",      "axes.labelcolor": "#ccc",
    "xtick.color": "#aaa",         "ytick.color": "#aaa",
    "text.color": "#eee",          "grid.color": "#2a2a4a",
    "grid.linestyle": "--",        "grid.alpha": 0.4,
})

fig, axes = plt.subplots(4, 1, figsize=(15, 11),
                         gridspec_kw={"height_ratios": [1.2, 1.2, 1.2, 1.4]})
fig.suptitle(f"NeuroWatch  |  MODELS_FS75  |  {n} windows  |  Acc: {acc*100:.1f}%",
             fontsize=13, fontweight="bold")

# ── Panel 1: Prediction timeline ──────────────────────────────────────────────
ax1 = axes[0]
for i in range(n):
    ax1.bar(i, 1, color=COLORS[y_pred[i]], alpha=0.85, width=1.0, linewidth=0)
    if not correct[i]:
        ax1.bar(i, 1, color="white", alpha=0.25, width=1.0, linewidth=0)
ax1.set_xlim(0, n); ax1.set_ylim(0, 1); ax1.set_yticks([])
ax1.set_title("Prediction Timeline  (white overlay = wrong)", fontsize=10)
ax1.set_xlabel("Window")
legend1 = [mpatches.Patch(color=COLORS[i], label=CLASSES[i]) for i in range(3)]
ax1.legend(handles=legend1, loc="upper right", fontsize=8, framealpha=0.3)

# ── Panel 2: Ground truth timeline ────────────────────────────────────────────
ax2 = axes[1]
for i in range(n):
    ax2.bar(i, 1, color=COLORS[y_true[i]], alpha=0.85, width=1.0, linewidth=0)
ax2.set_xlim(0, n); ax2.set_ylim(0, 1); ax2.set_yticks([])
ax2.set_title("Ground Truth Timeline", fontsize=10)
ax2.set_xlabel("Window")

# ── Panel 3: Confidence over time ─────────────────────────────────────────────
ax3 = axes[2]
for i, (color, cls) in enumerate(zip(COLORS, CLASSES)):
    ax3.plot(confs[:, i], color=color, lw=1.1, alpha=0.9, label=cls)
ax3.axhline(THRESH, color="#aaa", ls="--", lw=1, alpha=0.6,
            label=f"Normal threshold ({THRESH:.2f})")
ax3.set_xlim(0, n); ax3.set_ylim(0, 1)
ax3.set_ylabel("Confidence"); ax3.set_xlabel("Window")
ax3.set_title("Model Confidence Per Class Over Time", fontsize=10)
ax3.legend(fontsize=8, framealpha=0.3); ax3.grid(True)

# ── Panel 4: Per-class accuracy bars ─────────────────────────────────────────
ax4 = axes[3]
cls_counts = []; cls_accs = []
for i, (cls, color) in enumerate(zip(CLASSES, COLORS)):
    mask = y_true == i
    if mask.any():
        ca = (y_pred[mask] == i).mean()
        ax4.bar(cls, ca, color=color, alpha=0.9, edgecolor="#333", width=0.5)
        ax4.text(i, ca + 0.02, f"{ca*100:.1f}%\n(n={mask.sum()})",
                 ha="center", fontsize=10)
        cls_accs.append(ca); cls_counts.append(mask.sum())
ax4.set_ylim(0, 1.2); ax4.set_ylabel("Accuracy")
ax4.set_title("Per-Class Accuracy on Test Split", fontsize=10)
ax4.axhline(acc, color="#aaa", ls="--", lw=1, alpha=0.6,
            label=f"Overall {acc*100:.1f}%")
ax4.legend(fontsize=8, framealpha=0.3); ax4.grid(True, axis="y")

plt.tight_layout()
plt.show()
