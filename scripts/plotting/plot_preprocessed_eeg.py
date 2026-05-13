"""
plot_preprocessed_eeg.py  —  Browse ALL preprocessed EEG windows after pre-ictal
extraction. All 19 channels on one single graph with vertical offsets.

Slider walks through every sample. Background and channel colour change to
show which class the current window belongs to:
    green  = Normal    (class 0)
    orange = Pre-ictal (class 1)
    red    = Ictal     (class 2)

USAGE:
    python scripts/plotting/plot_preprocessed_eeg.py
    python scripts/plotting/plot_preprocessed_eeg.py --start 500
    python scripts/plotting/plot_preprocessed_eeg.py --split train
    python scripts/plotting/plot_preprocessed_eeg.py --split test
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NPY_DIR  = os.path.join(BASE_DIR, "data", "Mendelay dataset", "Npy_files_preictal")

CLASSES     = {0: "Normal", 1: "Pre-ictal", 2: "Ictal"}
CLASS_COLOR = {0: "#22aa44", 1: "#ff8800", 2: "#ee3333"}
BG_COLOR    = {0: "#0d1f10",  1: "#1f1500",  2: "#1f0000"}

CH_NAMES = [
    "Fp2","Fp1","F8","F4","Fz","F3","F7",
    "A2","T4","C4","C3","T3","A1",
    "T6","P4","P3","T5","O2","O1",
]

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["train","test","both"], default="both")
ap.add_argument("--start", type=int, default=0)
args = ap.parse_args()

# ── Load ALL classes ──────────────────────────────────────────────────────────
parts_x, parts_y = [], []
for split in (["train","test"] if args.split == "both" else [args.split]):
    xp = os.path.join(NPY_DIR, f"x_{split}.npy")
    yp = os.path.join(NPY_DIR, f"y_{split}.npy")
    if not os.path.exists(xp):
        print(f"WARNING: {xp} not found — skipping.")
        continue
    px, py = np.load(xp), np.load(yp)
    parts_x.append(px)
    parts_y.append(py)
    print(f"Loaded x_{split}.npy  shape={px.shape}")

if not parts_x:
    print(f"ERROR: No .npy files found in:\n  {NPY_DIR}")
    print("Run preprocess_preictal.py first.")
    sys.exit(1)

X = np.concatenate(parts_x, axis=0)   # (N, 19, 500)
y = np.concatenate(parts_y, axis=0)   # (N,)

print(f"\nTotal samples loaded: {len(X):,}")
for c, name in CLASSES.items():
    cnt = int(np.sum(y == c))
    print(f"  Class {c} — {name:<12}: {cnt:>6,}  ({cnt/len(y)*100:.1f}%)")

N_SAMPLES = len(X)
N_CH      = X.shape[1]   # 19
N_TIME    = X.shape[2]   # 500
t         = np.arange(N_TIME) / 500 * 1000   # ms
OFFSET    = 2.5

# ── Figure — single axes ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9), facecolor="#0d0d0d")
ax  = fig.add_axes([0.08, 0.15, 0.89, 0.78])
ax.set_facecolor(BG_COLOR[1])
ax.set_xlim(0, t[-1])
ax.set_ylim(-OFFSET, N_CH * OFFSET)
ax.set_xlabel("Time (ms)", color="white", fontsize=9)
ax.tick_params(colors="white")
ax.set_yticks([i * OFFSET for i in range(N_CH)])
ax.set_yticklabels(CH_NAMES[:N_CH], fontsize=7)
for spine in ax.spines.values():
    spine.set_color("#333333")

# Horizontal dividers between channels
for i in range(N_CH):
    ax.axhline(i * OFFSET - OFFSET * 0.45, color="#333333",
               linewidth=0.3, alpha=0.5)

# One line per channel — all on the same axes
lines = []
for i in range(N_CH):
    ln, = ax.plot(t, np.zeros(N_TIME) + i * OFFSET,
                  linewidth=0.6, alpha=0.9, rasterized=True, color=CLASS_COLOR[1])
    lines.append(ln)

title_obj = fig.suptitle("", fontsize=11, y=0.97)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=CLASS_COLOR[0], label="Normal (class 0)"),
    mpatches.Patch(color=CLASS_COLOR[1], label="Pre-ictal (class 1)"),
    mpatches.Patch(color=CLASS_COLOR[2], label="Ictal (class 2)"),
]
fig.legend(handles=legend_patches, loc="lower right", ncol=3,
           facecolor="#1a1a1a", labelcolor="white", fontsize=8,
           framealpha=0.9, bbox_to_anchor=(0.99, 0.00))

# ── Controls ──────────────────────────────────────────────────────────────────
ax_slider = fig.add_axes([0.08, 0.07, 0.65, 0.025])
ax_prev   = fig.add_axes([0.08, 0.02, 0.07, 0.035])
ax_next   = fig.add_axes([0.17, 0.02, 0.07, 0.035])
ax_save   = fig.add_axes([0.80, 0.02, 0.09, 0.035])

slider = Slider(ax_slider, "Sample", 0, N_SAMPLES - 1,
                valinit=min(args.start, N_SAMPLES - 1), valstep=1,
                color="#555555")
slider.label.set_color("white")
slider.valtext.set_color("white")

btn_prev = Button(ax_prev, "◀ Prev", color="#222222", hovercolor="#444444")
btn_next = Button(ax_next, "Next ▶", color="#222222", hovercolor="#444444")
btn_save = Button(ax_save, "Save PNG", color="#222222", hovercolor="#444444")
for b in [btn_prev, btn_next, btn_save]:
    b.label.set_color("white")

SAVE_DIR = os.path.join(BASE_DIR, "scripts", "plotting", "preprocessed_samples")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Draw ──────────────────────────────────────────────────────────────────────
def draw(idx):
    idx    = int(idx)
    sample = X[idx]        # (19, 500)
    label  = int(y[idx])
    col    = CLASS_COLOR[label]
    bg     = BG_COLOR[label]

    ax.set_facecolor(bg)
    ax.set_yticklabels(CH_NAMES[:N_CH], color=col, fontsize=7)

    for i in range(N_CH):
        sig   = sample[i]
        rng   = np.ptp(sig) or 1.0
        sig_n = (sig - sig.mean()) / rng * (OFFSET * 0.45)
        lines[i].set_ydata(sig_n + i * OFFSET)
        lines[i].set_color(col)

    title_obj.set_text(
        f"Sample {idx + 1:,} / {N_SAMPLES:,}   │   "
        f"Class {label} — {CLASSES[label]}   │   "
        f"19 channels  ×  500 samples  (1 s at 500 Hz)"
    )
    title_obj.set_color(col)
    fig.canvas.draw_idle()

def on_slider(val): draw(int(val))
def on_prev(_):     slider.set_val(max(0, int(slider.val) - 1))
def on_next(_):     slider.set_val(min(N_SAMPLES - 1, int(slider.val) + 1))

def on_save(_):
    idx  = int(slider.val)
    lbl  = int(y[idx])
    path = os.path.join(SAVE_DIR, f"sample_{idx:05d}_class{lbl}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {path}")

slider.on_changed(on_slider)
btn_prev.on_clicked(on_prev)
btn_next.on_clicked(on_next)
btn_save.on_clicked(on_save)

draw(min(args.start, N_SAMPLES - 1))
plt.show()
