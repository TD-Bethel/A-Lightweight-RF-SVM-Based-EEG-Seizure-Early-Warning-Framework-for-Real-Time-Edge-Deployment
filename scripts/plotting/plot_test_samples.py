# =============================================================================
# plot_test_samples.py  —  Show the exact Bonn test-split samples
#
# PURPOSE:
#   During training the Bonn dataset was split 80/10/10 (train/val/test) using
#   a fixed random seed (seed=42). This script recreates that EXACT same split
#   so you can inspect the real test samples — the ones the model was evaluated
#   on but never trained with.
#
#   An interactive slider lets you scroll through all 50 test samples.
#   Each sample is colour-coded by its class:
#     Green  (#00e676) = Normal
#     Yellow (#ffca28) = Pre-Seizure
#     Red    (#ff5252) = Seizure
#
# USAGE:
#   python scripts/plotting/plot_test_samples.py
#   python scripts/plotting/plot_test_samples.py --data "data/Bonn Univeristy Dataset"
#
# OUTPUT:
#   - Console: numbered list of all test samples with their class and filename
#   - GUI window: interactive EEG signal viewer with slider + Save button
#   - Saved PNGs: written to  Plot/Bonn Test Split/  when you click Save
# =============================================================================

import os, sys, glob, argparse
import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Resolve project root (two levels above this script's directory).
# DEFAULT_DATA points to the standard Bonn dataset folder inside the project.
# Change --data on the command line if your data lives elsewhere.
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA = os.path.join(BASE_DIR, "data", "Bonn Univeristy Dataset")

# ---------------------------------------------------------------------------
# Dataset configuration — must match EXACTLY what train_and_export.py used
# so the split is reproduced identically.
#
# CLASS_MAP maps folder letters to integer class labels:
#   O and N → 0 (Normal)
#   F and S → 1 (Pre-Seizure / interictal)
#   Z       → 2 (Seizure / ictal)
# ---------------------------------------------------------------------------
CLASS_MAP   = {"O": 0, "N": 0, "F": 1, "S": 1, "Z": 2}
CLASSES     = ["Normal", "Pre-Seizure", "Seizure"]   # index matches label integer
CLASS_COLOR = {0: "#00e676", 1: "#ffca28", 2: "#ff5252"}   # per-class plot colour
SEED        = 42     # random seed — MUST match the training script exactly
VAL_SIZE    = 0.10   # 10% of data reserved for validation
TEST_SIZE   = 0.10   # 10% of data reserved for testing

# ---------------------------------------------------------------------------
# Parse command-line arguments.
# If --data is not supplied, DEFAULT_DATA above is used.
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=DEFAULT_DATA)
args = ap.parse_args()

# =============================================================================
# SECTION 1: Load all EEG files from the Bonn dataset
#
# The Bonn dataset is organised as five sub-folders (F, N, O, S, Z).
# Each folder contains 100 .txt files, one recording per file.
# Each .txt file is a single-column array of voltage samples (~4097 values).
# =============================================================================
print(f"\nLoading dataset from: {args.data}")
filepaths, labels, signals = [], [], []

for folder in sorted(CLASS_MAP.keys()):
    lbl         = CLASS_MAP[folder]           # integer label for this folder
    folder_path = os.path.join(args.data, folder)

    if not os.path.isdir(folder_path):
        print(f"  ⚠  {folder}/ not found — skipping")
        continue

    # glob finds every .txt file in the folder and sorts alphabetically
    txt_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))

    for fpath in txt_files:
        try:
            # np.loadtxt reads the file as a 1-D array of floats.
            # .squeeze() removes any extra dimensions (e.g. shape (4097,1) → (4097,))
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

# Convert Python lists to NumPy arrays so we can use boolean indexing later
filepaths = np.array(filepaths)
labels    = np.array(labels)
signals   = np.array(signals, dtype=object)   # dtype=object: signals may differ in length
indices   = np.arange(len(filepaths))         # [0, 1, 2, ... N-1]

# =============================================================================
# SECTION 2: Recreate the exact 80/10/10 stratified split
#
# train_test_split from scikit-learn is deterministic given the same:
#   - data order (indices)
#   - test_size fraction
#   - random_state (SEED)
#   - stratify array (labels) — ensures each class is proportionally represented
#
# Step 1: split into train (80%) and temp (20%)
# Step 2: split temp equally into val (10%) and test (10%)
# =============================================================================
idx_train, idx_temp = train_test_split(
    indices, test_size=(VAL_SIZE + TEST_SIZE),    # 20% goes to temp
    random_state=SEED, stratify=labels,           # reproducible + class-balanced
)
idx_val, idx_test = train_test_split(
    idx_temp, test_size=0.5,                      # 50% of 20% = 10% test
    random_state=SEED, stratify=labels[idx_temp], # keep class balance in val/test
)

# Use the test indices to select only the test-split samples
test_files  = filepaths[idx_test]
test_labels = labels[idx_test]
test_sigs   = signals[idx_test]

# Print a breakdown of how many samples per class ended up in the test split
print(f"\n{'─'*60}")
print(f"  Test split: {len(idx_test)} samples")
for c in range(3):
    n = int(np.sum(test_labels == c))
    print(f"    {CLASSES[c]:<14}: {n}")
print(f"{'─'*60}")

# =============================================================================
# SECTION 3: Print the full list of test files to the console
#
# Format: index | label name | source folder | filename
# This lets you verify which specific recordings went into the test set.
# =============================================================================
print(f"\n{'#':>4}  {'Label':<14}  {'Folder':<6}  File")
print("─" * 60)
for i, (fp, lbl) in enumerate(zip(test_files, test_labels)):
    folder = os.path.basename(os.path.dirname(fp))   # e.g. "Z" from "...Bonn/Z/Z001.txt"
    fname  = os.path.basename(fp)                    # e.g. "Z001.txt"
    print(f"  {i+1:>2}  {CLASSES[lbl]:<14}  {folder:<6}  {fname}")

# =============================================================================
# SECTION 4: Build the interactive matplotlib viewer
#
# The viewer shows one EEG signal at a time.
# A slider at the bottom lets you browse all test samples.
# A "Save" button exports the current view as a PNG.
# =============================================================================
try:
    import matplotlib; matplotlib.use("TkAgg")   # windowed backend (requires Tk/Tcl)
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
except ImportError:
    print("\n❌ matplotlib not installed — pip install matplotlib")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Dark colour theme for the viewer window.
# These settings override matplotlib defaults for a cleaner look.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#0d0f1a",   # outer window background
    "axes.facecolor":   "#13162a",   # plot area background
    "axes.edgecolor":   "#444",      # axis border
    "axes.labelcolor":  "#ccc",      # axis label text
    "xtick.color":      "#aaa",      # tick marks
    "ytick.color":      "#aaa",
    "text.color":       "#eee",      # general text
    "grid.color":       "#2a2a4a",   # grid lines
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})

N = len(test_files)   # total number of test samples (typically 50)

def make_title(i):
    """
    Build the title string for sample i.
    Shows: sample index, class name, source folder, and filename.
    """
    folder = os.path.basename(os.path.dirname(test_files[i]))
    fname  = os.path.basename(test_files[i])
    lbl    = test_labels[i]
    return (f"Test Sample {i+1}/{N}  |  {CLASSES[lbl]}  "
            f"|  Folder: {folder}/  |  {fname}")

# ---------------------------------------------------------------------------
# Create the main figure and axes.
# bottom=0.22 leaves vertical space for the slider widget below the plot.
# top=0.88    leaves space for the title above.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))
plt.subplots_adjust(bottom=0.22, top=0.88)
fig.patch.set_facecolor("#0d0f1a")

# Draw the first sample's EEG waveform.
# [line] unpacks the list returned by ax.plot into a single Line2D object
# so we can update it later without redrawing the whole axes.
sig0  = test_sigs[0]
col0  = CLASS_COLOR[test_labels[0]]
[line] = ax.plot(sig0, color=col0, lw=0.9)   # lw=line width in points
ax.set_title(make_title(0), fontsize=11, pad=10)
ax.set_xlabel("Sample index")    # x-axis: time as discrete sample number
ax.set_ylabel("Amplitude (µV)") # y-axis: EEG voltage in microvolts
ax.grid(True)

# ---------------------------------------------------------------------------
# Colour band at the top of the plot area — a thin horizontal strip that
# shows the class colour even before reading the title text.
# axhspan draws a horizontal shaded band between two y values.
# ---------------------------------------------------------------------------
band = ax.axhspan(sig0.max()*0.97, sig0.max(), color=col0, alpha=0.35, lw=0)

# ---------------------------------------------------------------------------
# Class label badge — a small text box in the top-left corner with a coloured
# background matching the class. backgroundcolor is set dynamically on update.
# transform=ax.transAxes means coordinates are in axis fractions (0–1),
# not data units, so the badge stays fixed regardless of the signal range.
# ---------------------------------------------------------------------------
badge = ax.text(
    0.01, 0.95, f"  {CLASSES[test_labels[0]]}  ",
    transform=ax.transAxes, fontsize=11, fontweight="bold",
    color="#111", backgroundcolor=col0,
    verticalalignment="top",
)

# =============================================================================
# SECTION 5: Slider widget
#
# plt.axes([left, bottom, width, height]) creates a new Axes object at the
# given position in figure-fraction coordinates (0=left/bottom, 1=right/top).
# The Slider widget lives inside that Axes rectangle.
#
# valinit=1 — start at sample 1
# valstep=1 — only integer steps (no fractional sample index)
# =============================================================================
ax_slider = plt.axes([0.12, 0.08, 0.76, 0.03], facecolor="#181c30")
slider    = Slider(ax_slider, "Sample", 1, N, valinit=1, valstep=1,
                   color="#4fc3f7")   # slider handle colour

def update(val):
    """
    Called automatically every time the slider value changes.
    Updates the waveform, title, colour band, and class badge to match
    the newly selected sample index.
    """
    i   = int(slider.val) - 1       # slider is 1-indexed; array is 0-indexed
    sig = test_sigs[i]
    col = CLASS_COLOR[test_labels[i]]

    line.set_ydata(sig)              # replace waveform data (faster than re-plotting)
    line.set_color(col)              # change line colour to match class
    ax.relim()                       # recompute axis limits from the new data
    ax.autoscale_view()              # apply those new limits to the view
    ax.set_title(make_title(i), fontsize=11, pad=10)

    # Update the colour band corners: [[x0,y0],[x0,y1],[x1,y1],[x1,y0]]
    band.set_xy([[0, sig.max()*0.97], [0, sig.max()],
                 [1, sig.max()],      [1, sig.max()*0.97]])
    band.set_facecolor(col)

    badge.set_text(f"  {CLASSES[test_labels[i]]}  ")
    badge.set_backgroundcolor(col)

    fig.canvas.draw_idle()   # queue a redraw (more efficient than draw())

slider.on_changed(update)   # register the callback

# =============================================================================
# SECTION 6: Save button
#
# Clicking "Save" exports the current figure as a PNG to Plot/Bonn Test Split/.
# The filename encodes the index, source folder, and original filename so saved
# plots are self-documenting.
# =============================================================================
ax_save = plt.axes([0.89, 0.04, 0.09, 0.04], facecolor="#181c30")
btn     = Button(ax_save, "Save", color="#181c30", hovercolor="#252a45")
btn.label.set_color("#eee")   # white button text

save_dir = os.path.join(BASE_DIR, "Plot", "Bonn Test Split")
os.makedirs(save_dir, exist_ok=True)   # create folder if it doesn't exist yet

def save(event):
    """
    Saves the currently displayed sample as a PNG.
    event is the mouse-click event object (not used directly).
    """
    i     = int(slider.val) - 1
    folder = os.path.basename(os.path.dirname(test_files[i]))
    fname  = os.path.splitext(os.path.basename(test_files[i]))[0]   # strip .txt
    out    = os.path.join(save_dir, f"test_{i+1:02d}_{folder}_{fname}.png")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())   # dpi=150 = good screen resolution
    print(f"Saved: {out}")

btn.on_clicked(save)   # register the save callback

print(f"\n  Showing {N} test samples — use the slider to browse.")
print(f"  Saved plots go to: Plot/Bonn Test Split/")
plt.show()   # blocking: keeps the window open until closed
