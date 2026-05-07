# =============================================================================
# plot_test_samples.py  —  Show the exact Bonn test-split samples
#
# Recreates the same stratified 80/10/10 split (seed=42) used during
# training and plots the 50 test samples with a slider.
#
# USAGE:
#   python scripts/plot_test_samples.py
#   python scripts/plot_test_samples.py --data "data/Bonn Univeristy Dataset"
# =============================================================================

import os, sys, glob, argparse
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA = os.path.join(BASE_DIR, "data", "Bonn Univeristy Dataset")

CLASS_MAP   = {"O": 0, "N": 0, "F": 1, "S": 1, "Z": 2}
CLASSES     = ["Normal", "Pre-Seizure", "Seizure"]
CLASS_COLOR = {0: "#00e676", 1: "#ffca28", 2: "#ff5252"}
SEED        = 42
VAL_SIZE    = 0.10
TEST_SIZE   = 0.10

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=DEFAULT_DATA)
args = ap.parse_args()

# ── Load files, keeping track of path + label ─────────────────────────────────
print(f"\nLoading dataset from: {args.data}")
filepaths, labels, signals = [], [], []

for folder in sorted(CLASS_MAP.keys()):
    lbl         = CLASS_MAP[folder]
    folder_path = os.path.join(args.data, folder)
    if not os.path.isdir(folder_path):
        print(f"  ⚠  {folder}/ not found — skipping")
        continue
    txt_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
    for fpath in txt_files:
        try:
            sig = np.loadtxt(fpath).squeeze()
            filepaths.append(fpath)
            labels.append(lbl)
            signals.append(sig)
        except Exception as e:
            print(f"  ⚠  {os.path.basename(fpath)}: {e}")
    print(f"  {folder}/  ({len(txt_files)} files → {CLASSES[lbl]})")

if not filepaths:
    print("❌ No files loaded. Check --data path.")
    sys.exit(1)

filepaths = np.array(filepaths)
labels    = np.array(labels)
signals   = np.array(signals, dtype=object)
indices   = np.arange(len(filepaths))

# ── Recreate exact 80/10/10 stratified split ─────────────────────────────────
idx_train, idx_temp = train_test_split(
    indices, test_size=(VAL_SIZE + TEST_SIZE),
    random_state=SEED, stratify=labels,
)
idx_val, idx_test = train_test_split(
    idx_temp, test_size=0.5,
    random_state=SEED, stratify=labels[idx_temp],
)

test_files  = filepaths[idx_test]
test_labels = labels[idx_test]
test_sigs   = signals[idx_test]

print(f"\n{'─'*60}")
print(f"  Test split: {len(idx_test)} samples")
for c in range(3):
    n = int(np.sum(test_labels == c))
    print(f"    {CLASSES[c]:<14}: {n}")
print(f"{'─'*60}")

# ── Print full list ───────────────────────────────────────────────────────────
print(f"\n{'#':>4}  {'Label':<14}  {'Folder':<6}  File")
print("─" * 60)
for i, (fp, lbl) in enumerate(zip(test_files, test_labels)):
    folder = os.path.basename(os.path.dirname(fp))
    fname  = os.path.basename(fp)
    print(f"  {i+1:>2}  {CLASSES[lbl]:<14}  {folder:<6}  {fname}")

# ── Plot with slider ──────────────────────────────────────────────────────────
try:
    import matplotlib; matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
except ImportError:
    print("\n❌ matplotlib not installed — pip install matplotlib")
    sys.exit(1)

plt.rcParams.update({
    "figure.facecolor": "#0d0f1a", "axes.facecolor":  "#13162a",
    "axes.edgecolor":   "#444",    "axes.labelcolor": "#ccc",
    "xtick.color":      "#aaa",    "ytick.color":     "#aaa",
    "text.color":       "#eee",    "grid.color":      "#2a2a4a",
    "grid.linestyle":   "--",      "grid.alpha":      0.5,
})

N = len(test_files)

def make_title(i):
    folder = os.path.basename(os.path.dirname(test_files[i]))
    fname  = os.path.basename(test_files[i])
    lbl    = test_labels[i]
    return (f"Test Sample {i+1}/{N}  |  {CLASSES[lbl]}  "
            f"|  Folder: {folder}/  |  {fname}")

fig, ax = plt.subplots(figsize=(14, 5))
plt.subplots_adjust(bottom=0.22, top=0.88)
fig.patch.set_facecolor("#0d0f1a")

sig0  = test_sigs[0]
col0  = CLASS_COLOR[test_labels[0]]
[line] = ax.plot(sig0, color=col0, lw=0.9)
ax.set_title(make_title(0), fontsize=11, pad=10)
ax.set_xlabel("Sample index")
ax.set_ylabel("Amplitude (µV)")
ax.grid(True)

# Colour band at top showing class
band = ax.axhspan(sig0.max()*0.97, sig0.max(), color=col0, alpha=0.35, lw=0)

# Label badge
badge = ax.text(
    0.01, 0.95, f"  {CLASSES[test_labels[0]]}  ",
    transform=ax.transAxes, fontsize=11, fontweight="bold",
    color="#111", backgroundcolor=col0,
    verticalalignment="top",
)

# Slider
ax_slider = plt.axes([0.12, 0.08, 0.76, 0.03], facecolor="#181c30")
slider    = Slider(ax_slider, "Sample", 1, N, valinit=1, valstep=1,
                   color="#4fc3f7")

def update(val):
    i   = int(slider.val) - 1
    sig = test_sigs[i]
    col = CLASS_COLOR[test_labels[i]]
    line.set_ydata(sig)
    line.set_color(col)
    ax.relim(); ax.autoscale_view()
    ax.set_title(make_title(i), fontsize=11, pad=10)
    band.set_xy([[0, sig.max()*0.97], [0, sig.max()],
                 [1, sig.max()],      [1, sig.max()*0.97]])
    band.set_facecolor(col)
    badge.set_text(f"  {CLASSES[test_labels[i]]}  ")
    badge.set_backgroundcolor(col)
    fig.canvas.draw_idle()

slider.on_changed(update)

# Save button
ax_save = plt.axes([0.89, 0.04, 0.09, 0.04], facecolor="#181c30")
btn     = Button(ax_save, "Save", color="#181c30", hovercolor="#252a45")
btn.label.set_color("#eee")

save_dir = os.path.join(BASE_DIR, "Plot", "Bonn Test Split")
os.makedirs(save_dir, exist_ok=True)

def save(event):
    i     = int(slider.val) - 1
    folder = os.path.basename(os.path.dirname(test_files[i]))
    fname  = os.path.splitext(os.path.basename(test_files[i]))[0]
    out    = os.path.join(save_dir, f"test_{i+1:02d}_{folder}_{fname}.png")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved: {out}")

btn.on_clicked(save)

print(f"\n  Showing {N} test samples — use the slider to browse.")
print(f"  Saved plots go to: Plot/Bonn Test Split/")
plt.show()
