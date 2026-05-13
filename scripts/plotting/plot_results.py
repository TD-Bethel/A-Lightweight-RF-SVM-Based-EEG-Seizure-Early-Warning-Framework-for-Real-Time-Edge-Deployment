# =============================================================================
# plot_results.py  |  Show training plots from a saved report JSON
#
# PURPOSE:
#   After you train a model, the training script saves a JSON report file
#   containing accuracy, F1 score, confusion matrices, etc. This script reads
#   that report and draws four diagnostic charts without re-running training.
#
# USAGE:
#   python scripts/plot_results.py                          # auto-detects latest
#   python scripts/plot_results.py --model mendeley         # use Mendeley report
#   python scripts/plot_results.py --model bonn             # use Bonn report
#   python scripts/plot_results.py --report "path/to/report.json"  # explicit path
#
# OUTPUT (4 matplotlib windows):
#   Figure 1 — Confusion matrices for train / val / test splits
#   Figure 2 — Grouped bar chart of accuracy, F1, seizure recall
#   Figure 3 — Overfitting gap (train score minus val score per metric)
#   Figure 4 — Per-class precision / recall / F1 on the test set
# =============================================================================

import os, sys, json, argparse
import numpy as np

# ---------------------------------------------------------------------------
# Resolve the project root directory (two levels up from this script).
# e.g. this file lives at  <root>/scripts/plotting/plot_results.py
#       dirname(__file__)  →  scripts/plotting
#       dirname(...)       →  scripts
#       dirname(...)       →  <root>
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Command-line argument setup.
# --model  : choose between "mendeley" or "bonn" dataset report
# --report : bypass auto-detection and point directly to a JSON file
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--model",  choices=["mendeley","bonn"], default=None)
ap.add_argument("--report", default=None, help="Direct path to a report JSON file")
args = ap.parse_args()

# =============================================================================
# SECTION 1: Locate the correct report JSON file
# Priority order:
#   1. --report flag (explicit path)
#   2. --model flag (known preset path)
#   3. Auto-detect: prefer Mendeley if both exist, else fall back to Bonn
# =============================================================================
if args.report:
    # User gave us a specific file — use it directly
    report_path = args.report
elif args.model == "mendeley":
    # Mendeley model saves a single flat JSON report
    report_path = os.path.join(BASE_DIR, "models", "MODELS_MENDELEY", "training_report.json")
elif args.model == "bonn":
    # Bonn model saves a LIST of runs in training_history.json;
    # we will pick the last entry (most recent run) below
    report_path = os.path.join(BASE_DIR, "models", "MODELS_V1", "training_history.json")
else:
    # Auto-detect: prefer Mendeley if both exist
    mp = os.path.join(BASE_DIR, "models", "MODELS_MENDELEY", "training_report.json")
    bp = os.path.join(BASE_DIR, "models", "MODELS_V1", "training_history.json")
    report_path = mp if os.path.exists(mp) else bp

# Exit early if the chosen file doesn't exist
if not os.path.exists(report_path):
    print(f"# Report not found: {report_path}")
    sys.exit(1)

print(f"## Loading: {report_path}")
with open(report_path) as f:
    data = json.load(f)  # parse the JSON into a Python dict (or list)

# ---------------------------------------------------------------------------
# Handle the Bonn history format: it's a JSON list where each item is one
# training run. We always inspect the LAST (most recent) entry.
# ---------------------------------------------------------------------------
if isinstance(data, list):
    data = data[-1]

# ---------------------------------------------------------------------------
# Extract top-level metadata fields from the report.
# .get(key, default) prevents a KeyError if the field is absent.
# ---------------------------------------------------------------------------
dataset   = data.get("trained_on", data.get("dataset", "Unknown"))  # which dataset was used
model_str = data.get("model", "Hybrid SVM + RF")                    # model description string
n_feat    = data.get("n_features", "?")                             # number of input features
split     = data.get("split", "80/10/10")                           # train/val/test split ratio
timestamp = data.get("timestamp", "")                               # ISO timestamp of training run

print(f"  Dataset  : {dataset}")
print(f"  Model    : {model_str}")
print(f"  Features : {n_feat}")
print(f"  Trained  : {timestamp[:19]}")   # slice to drop sub-second precision

# =============================================================================
# SECTION 2: Pull per-split metrics from the report
# The JSON has three sub-dicts: "training", "validation", "test"
# Each contains keys like "accuracy", "f1", "seizure_recall", etc.
# =============================================================================
splits = ["training", "validation", "test"]
labels = ["Train (80%)", "Val (10%)", "Test (10%)"]
colors = ["#4fc3f7", "#00CC66", "#FFD633"]   # blue, green, yellow — one per split

# Metric keys to compare across splits and their display names
metric_keys   = ["accuracy", "f1", "seizure_recall"]
metric_labels = ["Accuracy", "F1 (weighted)", "Seizure Recall"]

# Build a dict of {split_name: metrics_dict}
vals = {s: data.get(s, {}) for s in splits}

# =============================================================================
# SECTION 3: Import matplotlib with the TkAgg backend
# TkAgg is the interactive windowed backend (creates a real GUI window).
# If your system only has a headless display, swap to "Agg" and call savefig().
# =============================================================================
try:
    import matplotlib; matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
except ImportError as e:
    print(f"# Plot dependencies missing: {e}")
    print("   pip install matplotlib seaborn")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Global matplotlib style: dark navy theme matching the NeuroWatch dashboard.
# Change any hex colour here to restyle ALL four figures at once.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",   # outer window background (very dark navy)
    "axes.facecolor":   "#16213e",   # plot area background (slightly lighter)
    "axes.edgecolor":   "#444",      # axis border colour
    "axes.labelcolor":  "#ccc",      # x/y axis label text colour
    "xtick.color":      "#aaa",      # x-axis tick mark colour
    "ytick.color":      "#aaa",      # y-axis tick mark colour
    "text.color":       "#eee",      # default text colour (titles, annotations)
    "grid.color":       "#2a2a4a",   # grid line colour
    "grid.linestyle":   "--",        # dashed grid lines
    "grid.alpha":       0.5,         # grid transparency (0=invisible, 1=solid)
})

# Class names used for confusion matrix axis labels.
# Must match the label integers (0=Normal, 1=Pre-Seizure, 2=Seizure).
CLASSES = ["Normal", "Pre-Seizure", "Seizure"]

# =============================================================================
# FIGURE 1: Confusion Matrices (one per split)
#
# A confusion matrix shows how often the model predicted each class correctly.
# - Rows = true (actual) class
# - Columns = predicted class
# - Diagonal cells = correct predictions (want these to be high)
# - Off-diagonal cells = mistakes (want these to be near 0)
# =============================================================================
cms = data.get("confusion_matrices", {})   # dict keyed by split name
if cms:
    # Create a 1×3 grid of subplots (one confusion matrix per split)
    fig1, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle(f"Confusion Matrices  |  {dataset}", fontsize=13, fontweight="bold")

    for ax, key, lbl, cmap in zip(axes,
                                   ["training","validation","test"],
                                   labels,
                                   ["Blues","Greens","Oranges"]):
        cm = np.array(cms.get(key, []))   # get matrix for this split (may be empty)
        if cm.size == 0:
            ax.set_visible(False)   # hide the subplot if there's no data
            continue

        n   = cm.shape[0]       # number of classes in this matrix
        cls = CLASSES[:n]       # slice class names to match matrix size

        # sns.heatmap draws the coloured grid with cell-count annotations
        # annot=True  — write the count inside each cell
        # fmt="d"     — format counts as integers (not floats)
        # cmap        — colour gradient (Blues/Greens/Oranges per split)
        # square=True — force square cells so the matrix isn't distorted
        # cbar=False  — hide the colour scale bar (counts speak for themselves)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                    xticklabels=cls, yticklabels=cls,
                    cbar=False, ax=ax, square=True, annot_kws={"size": 13})

        # Pull accuracy and F1 for this split to show in the subplot title
        acc = vals[key.replace("training","training").replace("test","test")].get("accuracy", 0)
        f1  = vals[key.replace("training","training").replace("test","test")].get("f1", 0)
        ax.set_title(f"{lbl}\nAcc: {acc*100:.1f}%  F1: {f1*100:.1f}%", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    plt.tight_layout(rect=[0,0,1,0.95])   # rect leaves space for the suptitle
    plt.show(block=False); plt.pause(0.3) # non-blocking show so multiple figures can open

# =============================================================================
# FIGURE 2: Metrics Bar Chart
#
# Grouped bars comparing Accuracy, F1, and Seizure Recall across all three
# splits side by side. Lets you quickly spot if train/val/test are consistent.
#
# - Accuracy    : fraction of all predictions that are correct
# - F1 (wt.)   : harmonic mean of precision & recall, weighted by class size
# - Seizure Recall : of all real seizure windows, how many did we catch?
#                    (Critical — missing a seizure is dangerous)
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(13, 6))
fig2.suptitle(f"Metrics Comparison  |  {dataset}", fontsize=13, fontweight="bold")

x  = np.arange(len(metric_labels))   # [0, 1, 2] — one position per metric
w  = 0.26                             # bar width (three bars fit side by side)
split_keys = ["training", "validation", "test"]
bars_list  = []                       # store bar containers to annotate later

for i, (sk, lbl, col) in enumerate(zip(split_keys, labels, colors)):
    row = vals.get(sk, {})                       # metrics dict for this split
    bar_vals = [row.get(m, 0) for m in metric_keys]  # [accuracy, f1, seizure_recall]
    # i-1 offsets bars: train(-1), val(0), test(+1) around each metric position
    b = ax2.bar(x + (i-1)*w, bar_vals, w, label=lbl, color=col, alpha=0.9, edgecolor="#333")
    bars_list.append(b)

ax2.set_xticks(x); ax2.set_xticklabels(metric_labels, fontsize=11)
ax2.set_ylim(0, 1.18)         # extra headroom above 1.0 for percentage labels
ax2.set_ylabel("Score")
ax2.legend(fontsize=10, framealpha=0.3)   # legend identifies which split each colour is
ax2.grid(True, axis="y")      # horizontal guide lines only (not vertical)

# Add percentage label on top of each bar
for bars in bars_list:
    for b in bars:
        h = b.get_height()    # the bar's score value (0–1)
        ax2.text(b.get_x()+b.get_width()/2, h+0.015,
                 f"{h*100:.0f}%",            # convert 0–1 to percentage string
                 ha="center", fontsize=9)

plt.tight_layout(rect=[0,0,1,0.95])
plt.show(block=False); plt.pause(0.3)

# =============================================================================
# FIGURE 3: Overfitting Gap (Train minus Val)
#
# If the model memorised the training data rather than learning general
# patterns, its training score will be much higher than validation score.
# This chart shows that gap: positive = model did better on train than val.
#
# Colour thresholds:
#   Green  (<5%)  — healthy, generalising well
#   Yellow (5–12%) — mild overfitting, monitor it
#   Red    (>12%) — significant overfitting, consider regularisation
# =============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 5))
fig3.suptitle(f"Overfitting Check (Train # Val)  |  {dataset}", fontsize=12, fontweight="bold")

tr = vals.get("training",   {})   # training metrics
vl = vals.get("validation", {})   # validation metrics

# Compute gap = train_score - val_score for each metric
gaps = [tr.get(m,0) - vl.get(m,0) for m in metric_keys]

# Colour each bar based on how large the gap is
gap_cols = ["#00CC66" if abs(g)<0.05 else "#FFD633" if abs(g)<0.12 else "#FF3333"
            for g in gaps]

ax3.bar(metric_labels, gaps, color=gap_cols, edgecolor="#333")
ax3.axhline(0, color="#aaa", lw=1)   # zero line — a gap of 0 is perfect

# Reference lines at ±5% and ±12% to visually show the thresholds
ax3.axhline( 0.05, color="#00CC66", lw=1, ls="--", alpha=0.6, label="#5% good")
ax3.axhline(-0.05, color="#00CC66", lw=1, ls="--", alpha=0.6)
ax3.axhline( 0.12, color="#FFD633", lw=1, ls="--", alpha=0.6, label="#12% mild")
ax3.axhline(-0.12, color="#FFD633", lw=1, ls="--", alpha=0.6)

ax3.set_ylabel("Train # Val gap"); ax3.set_ylim(-0.35, 0.35)
ax3.legend(fontsize=9, framealpha=0.3); ax3.grid(True, axis="y")

# Annotate each bar with its numeric gap value
for rect, g in zip(ax3.patches, gaps):
    ax3.text(rect.get_x()+rect.get_width()/2,
             g+(0.01 if g>=0 else -0.025),   # shift label above/below bar
             f"{g:+.2f}",                    # show sign (+/-) explicitly
             ha="center", fontsize=10)

plt.tight_layout()
plt.show(block=False); plt.pause(0.3)

# =============================================================================
# FIGURE 4: Per-Class Breakdown on the Test Set
#
# Breaks down the three key metrics for EACH class individually:
#
#   Precision  : of all windows we predicted as class X, how many actually were X?
#                (High precision = few false positives)
#   Recall     : of all windows that truly are class X, how many did we find?
#                (High recall = few false negatives — especially important for Seizure)
#   F1         : geometric balance between precision and recall for that class
#
# The "(n=...)" labels show how many test samples belong to each class.
# =============================================================================
per_class = vals.get("test", {}).get("per_class", {})   # nested dict: {class_name: {p, r, f1, count}}
if per_class:
    class_names = list(per_class.keys())

    # Extract metric lists in class order
    p_vals  = [per_class[c].get("precision", 0) for c in class_names]
    r_vals  = [per_class[c].get("recall",    0) for c in class_names]
    f_vals  = [per_class[c].get("f1",        0) for c in class_names]
    counts  = [per_class[c].get("count",     0) for c in class_names]   # sample count per class

    fig4, ax4 = plt.subplots(figsize=(11, 5))
    fig4.suptitle(f"Per-Class Breakdown # Test Set  |  {dataset}",
                  fontsize=12, fontweight="bold")

    xc = np.arange(len(class_names))   # [0, 1, 2] — one cluster per class
    wc = 0.25                           # bar width

    # Three bars per class: Precision (left), Recall (centre), F1 (right)
    ax4.bar(xc - wc, p_vals, wc, label="Precision", color="#4fc3f7", alpha=0.9, edgecolor="#333")
    ax4.bar(xc,      r_vals, wc, label="Recall",    color="#00CC66", alpha=0.9, edgecolor="#333")
    ax4.bar(xc + wc, f_vals, wc, label="F1",        color="#FFD633", alpha=0.9, edgecolor="#333")

    # x-tick labels show class name and sample count, e.g. "Seizure\n(n=50)"
    ax4.set_xticks(xc)
    ax4.set_xticklabels([f"{n}\n(n={c})" for n, c in zip(class_names, counts)], fontsize=10)
    ax4.set_ylim(0, 1.18); ax4.set_ylabel("Score")
    ax4.legend(fontsize=10, framealpha=0.3); ax4.grid(True, axis="y")

    plt.tight_layout()
    plt.show(block=False); plt.pause(0.3)

# ---------------------------------------------------------------------------
# All four figures are now open. Block on input() so the windows stay visible
# until you press Enter. plt.close("all") then disposes them.
# ---------------------------------------------------------------------------
print("\n  All plots open # press Enter to close...")
input()
plt.close("all")
