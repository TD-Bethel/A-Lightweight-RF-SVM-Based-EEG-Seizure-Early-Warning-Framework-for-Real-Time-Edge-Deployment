# =============================================================================
# plot_results.py  —  Show training plots from a saved report JSON
#
# USAGE:
#   python scripts/plot_results.py                          # auto-detects latest
#   python scripts/plot_results.py --model mendeley
#   python scripts/plot_results.py --model bonn
#   python scripts/plot_results.py --report "path/to/report.json"
# =============================================================================

import os, sys, json, argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--model",  choices=["mendeley","bonn"], default=None)
ap.add_argument("--report", default=None, help="Direct path to a report JSON file")
args = ap.parse_args()

# ── Locate report ──────────────────────────────────────────────────────────────
if args.report:
    report_path = args.report
elif args.model == "mendeley":
    report_path = os.path.join(BASE_DIR, "models", "MODELS_MENDELEY", "training_report.json")
elif args.model == "bonn":
    # Bonn uses training_history.json — last entry is the most recent run
    report_path = os.path.join(BASE_DIR, "models", "MODELS_V1", "training_history.json")
else:
    # Auto-detect: prefer Mendeley if both exist
    mp = os.path.join(BASE_DIR, "models", "MODELS_MENDELEY", "training_report.json")
    bp = os.path.join(BASE_DIR, "models", "MODELS_V1", "training_history.json")
    report_path = mp if os.path.exists(mp) else bp

if not os.path.exists(report_path):
    print(f"❌ Report not found: {report_path}")
    sys.exit(1)

print(f"📂 Loading: {report_path}")
with open(report_path) as f:
    data = json.load(f)

# Bonn history is a list — take the last entry
if isinstance(data, list):
    data = data[-1]

dataset   = data.get("trained_on", data.get("dataset", "Unknown"))
model_str = data.get("model", "Hybrid SVM + RF")
n_feat    = data.get("n_features", "?")
split     = data.get("split", "80/10/10")
timestamp = data.get("timestamp", "")

print(f"  Dataset  : {dataset}")
print(f"  Model    : {model_str}")
print(f"  Features : {n_feat}")
print(f"  Trained  : {timestamp[:19]}")

# ── Pull metrics ───────────────────────────────────────────────────────────────
splits = ["training", "validation", "test"]
labels = ["Train (80%)", "Val (10%)", "Test (10%)"]
colors = ["#4fc3f7", "#00CC66", "#FFD633"]

metric_keys = ["accuracy", "f1", "seizure_recall"]
metric_labels = ["Accuracy", "F1 (weighted)", "Seizure Recall"]

vals = {s: data.get(s, {}) for s in splits}

# ── Matplotlib setup ───────────────────────────────────────────────────────────
try:
    import matplotlib; matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
except ImportError as e:
    print(f"❌ Plot dependencies missing: {e}")
    print("   pip install matplotlib seaborn")
    sys.exit(1)

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e", "axes.facecolor":  "#16213e",
    "axes.edgecolor":   "#444",    "axes.labelcolor": "#ccc",
    "xtick.color":      "#aaa",    "ytick.color":     "#aaa",
    "text.color":       "#eee",    "grid.color":      "#2a2a4a",
    "grid.linestyle":   "--",      "grid.alpha":      0.5,
})
CLASSES = ["Normal", "Pre-Seizure", "Seizure"]

# ── Figure 1: Confusion matrices ───────────────────────────────────────────────
cms = data.get("confusion_matrices", {})
if cms:
    fig1, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle(f"Confusion Matrices  |  {dataset}", fontsize=13, fontweight="bold")
    for ax, key, lbl, cmap in zip(axes,
                                   ["training","validation","test"],
                                   labels,
                                   ["Blues","Greens","Oranges"]):
        cm = np.array(cms.get(key, []))
        if cm.size == 0:
            ax.set_visible(False); continue
        n = cm.shape[0]
        cls = CLASSES[:n]
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                    xticklabels=cls, yticklabels=cls,
                    cbar=False, ax=ax, square=True, annot_kws={"size": 13})
        acc = vals[key.replace("training","training").replace("test","test")].get("accuracy", 0)
        f1  = vals[key.replace("training","training").replace("test","test")].get("f1", 0)
        ax.set_title(f"{lbl}\nAcc: {acc*100:.1f}%  F1: {f1*100:.1f}%", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show(block=False); plt.pause(0.3)

# ── Figure 2: Metrics bar chart ────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(13, 6))
fig2.suptitle(f"Metrics Comparison  |  {dataset}", fontsize=13, fontweight="bold")

x  = np.arange(len(metric_labels))
w  = 0.26
split_keys = ["training", "validation", "test"]
bars_list  = []
for i, (sk, lbl, col) in enumerate(zip(split_keys, labels, colors)):
    row = vals.get(sk, {})
    bar_vals = [row.get(m, 0) for m in metric_keys]
    b = ax2.bar(x + (i-1)*w, bar_vals, w, label=lbl, color=col, alpha=0.9, edgecolor="#333")
    bars_list.append(b)

ax2.set_xticks(x); ax2.set_xticklabels(metric_labels, fontsize=11)
ax2.set_ylim(0, 1.18); ax2.set_ylabel("Score")
ax2.legend(fontsize=10, framealpha=0.3); ax2.grid(True, axis="y")
for bars in bars_list:
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x()+b.get_width()/2, h+0.015,
                 f"{h*100:.0f}%", ha="center", fontsize=9)
plt.tight_layout(rect=[0,0,1,0.95])
plt.show(block=False); plt.pause(0.3)

# ── Figure 3: Overfitting gap ──────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 5))
fig3.suptitle(f"Overfitting Check (Train − Val)  |  {dataset}", fontsize=12, fontweight="bold")
tr = vals.get("training",   {})
vl = vals.get("validation", {})
gaps = [tr.get(m,0) - vl.get(m,0) for m in metric_keys]
gap_cols = ["#00CC66" if abs(g)<0.05 else "#FFD633" if abs(g)<0.12 else "#FF3333"
            for g in gaps]
ax3.bar(metric_labels, gaps, color=gap_cols, edgecolor="#333")
ax3.axhline(0,     color="#aaa", lw=1)
ax3.axhline( 0.05, color="#00CC66", lw=1, ls="--", alpha=0.6, label="±5% good")
ax3.axhline(-0.05, color="#00CC66", lw=1, ls="--", alpha=0.6)
ax3.axhline( 0.12, color="#FFD633", lw=1, ls="--", alpha=0.6, label="±12% mild")
ax3.axhline(-0.12, color="#FFD633", lw=1, ls="--", alpha=0.6)
ax3.set_ylabel("Train − Val gap"); ax3.set_ylim(-0.35, 0.35)
ax3.legend(fontsize=9, framealpha=0.3); ax3.grid(True, axis="y")
for rect, g in zip(ax3.patches, gaps):
    ax3.text(rect.get_x()+rect.get_width()/2,
             g+(0.01 if g>=0 else -0.025),
             f"{g:+.2f}", ha="center", fontsize=10)
plt.tight_layout()
plt.show(block=False); plt.pause(0.3)

# ── Figure 4: Per-class breakdown ──────────────────────────────────────────────
per_class = vals.get("test", {}).get("per_class", {})
if per_class:
    class_names = list(per_class.keys())
    p_vals = [per_class[c].get("precision",0) for c in class_names]
    r_vals = [per_class[c].get("recall",   0) for c in class_names]
    f_vals = [per_class[c].get("f1",       0) for c in class_names]
    counts = [per_class[c].get("count",    0) for c in class_names]

    fig4, ax4 = plt.subplots(figsize=(11, 5))
    fig4.suptitle(f"Per-Class Breakdown — Test Set  |  {dataset}",
                  fontsize=12, fontweight="bold")
    xc = np.arange(len(class_names)); wc = 0.25
    ax4.bar(xc - wc, p_vals, wc, label="Precision", color="#4fc3f7", alpha=0.9, edgecolor="#333")
    ax4.bar(xc,      r_vals, wc, label="Recall",    color="#00CC66", alpha=0.9, edgecolor="#333")
    ax4.bar(xc + wc, f_vals, wc, label="F1",        color="#FFD633", alpha=0.9, edgecolor="#333")
    ax4.set_xticks(xc)
    ax4.set_xticklabels([f"{n}\n(n={c})" for n, c in zip(class_names, counts)], fontsize=10)
    ax4.set_ylim(0, 1.18); ax4.set_ylabel("Score")
    ax4.legend(fontsize=10, framealpha=0.3); ax4.grid(True, axis="y")
    plt.tight_layout()
    plt.show(block=False); plt.pause(0.3)

print("\n  All plots open — press Enter to close...")
input()
plt.close("all")
