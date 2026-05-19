"""
generate_figures.py  --  Generate all report figures for NeuroWatch thesis report.

Run from the REPORT directory:
    python generate_figures.py

Outputs (saved to REPORT/figures/):
    fig1_system_workflow.pdf
    fig2_feature_pipeline.pdf
    fig3_signal_processing.pdf
    fig4_model_architecture.pdf
    fig5_results_comparison.pdf
    fig6_confusion_matrices.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

DARK_BG   = "#1a1a2e"
MID_BG    = "#16213e"
ACCENT    = "#0f3460"
GREEN     = "#00cc66"
YELLOW    = "#ffd633"
RED       = "#ff3333"
BLUE      = "#4da6ff"
PURPLE    = "#cc66ff"
TEXT      = "#eeeeee"
SUBTEXT   = "#aaaaaa"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   MID_BG,
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  TEXT,
    "xtick.color":      SUBTEXT,
    "ytick.color":      SUBTEXT,
    "text.color":       TEXT,
    "grid.color":       "#2a2a4a",
    "font.family":      "DejaVu Sans",
    "font.size":        9,
})

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1  System Workflow
# ─────────────────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, sublabel="", color=ACCENT, fontsize=8):
    rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                          linewidth=1.2, edgecolor=BLUE, facecolor=color, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.06 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize, fontweight="bold",
            color=TEXT, zorder=4)
    if sublabel:
        ax.text(x, y - 0.14, sublabel,
                ha="center", va="center", fontsize=6.5, color=SUBTEXT, zorder=4)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5), zorder=2)

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)

ax.text(6, 4.7, "NeuroWatch — System Workflow",
        ha="center", va="center", fontsize=13, fontweight="bold", color=TEXT)

nodes = [
    (1.2, 2.5, 1.6, 0.9, "EEG Acquisition", "OpenBCI / EDF files"),
    (3.3, 2.5, 1.6, 0.9, "Preprocessing", "Bandpass 0.5–45 Hz\nDownsample 500→128 Hz"),
    (5.4, 2.5, 1.6, 0.9, "Feature Extraction", "247 features / window\n(13 per channel × 19 ch)"),
    (7.5, 2.5, 1.6, 0.9, "Feature Selection", "RF Importance\n75 of 247 features"),
    (9.6, 2.5, 1.6, 0.9, "Classification", "Hybrid SVM+RF\nEnsemble"),
    (11.2, 3.8, 1.4, 0.7, "Alert / Log", "Seizure detected"),
    (11.2, 1.2, 1.4, 0.7, "Dashboard", "Remote monitoring"),
]
for (x, y, w, h, lbl, sub) in nodes:
    box(ax, x, y, w, h, lbl, sub)

for i in range(len(nodes) - 3):
    x1 = nodes[i][0] + nodes[i][2]/2
    x2 = nodes[i+1][0] - nodes[i+1][2]/2
    arrow(ax, x1, 2.5, x2, 2.5)

# branches from Classification
arrow(ax, 9.6, 3.0, 10.5, 3.8)
arrow(ax, 9.6, 2.0, 10.5, 1.2)

# threshold annotation
ax.text(7.5, 1.7, "Normal threshold\ncorrection applied", ha="center",
        fontsize=7, color=YELLOW, style="italic")

plt.tight_layout(pad=0.3)
plt.savefig(os.path.join(OUT, "fig1_system_workflow.pdf"), bbox_inches="tight", dpi=150)
plt.savefig(os.path.join(OUT, "fig1_system_workflow.png"), bbox_inches="tight", dpi=150)
plt.close()
print("  fig1_system_workflow  saved")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2  Feature Extraction Pipeline (per channel)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.set_xlim(0, 12); ax.set_ylim(0, 4.5); ax.axis("off")
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)
ax.text(6, 4.2, "Feature Extraction Pipeline (per EEG channel)",
        ha="center", fontsize=12, fontweight="bold", color=TEXT)

ch_x = 1.0
box(ax, ch_x, 2.2, 1.4, 2.4, "EEG Channel\n(500 samples\n@ 500 Hz)", color="#1e3a5f", fontsize=8)

arrow(ax, ch_x + 0.7, 2.2, 2.3, 2.2)

# Frequency features
box(ax, 3.1, 3.2, 1.6, 0.7, "Welch PSD", "nperseg=128, fs=128", color=ACCENT, fontsize=7.5)
ax.text(3.1, 2.6, "δ [1–4]\nθ [4–8]\nα [8–13]\nβ [13–30]\nγ [30–45]",
        ha="center", va="top", fontsize=7, color=GREEN, family="monospace")
ax.text(3.1, 1.4, "5 band-power\nfeatures (norm.)", ha="center", fontsize=7, color=SUBTEXT)

# Time domain
box(ax, 5.8, 3.2, 1.6, 0.7, "Time Domain", "on resampled signal", color=ACCENT, fontsize=7.5)
ax.text(5.8, 2.6, "Hjorth Activity\nHjorth Mobility\nHjorth Complexity\nSkewness\nKurtosis",
        ha="center", va="top", fontsize=7, color=GREEN, family="monospace")

# More time domain
box(ax, 8.5, 3.2, 1.6, 0.7, "Signal Stats", "on resampled signal", color=ACCENT, fontsize=7.5)
ax.text(8.5, 2.6, "RMS\nZero-Crossing Rate\nSpectral Entropy",
        ha="center", va="top", fontsize=7, color=GREEN, family="monospace")

for bx in [3.1, 5.8, 8.5]:
    arrow(ax, 2.3 + (bx - 3.1) * 0.0, 2.2, bx - 0.8, 2.85)

box(ax, 10.8, 2.2, 1.4, 0.7, "13 features\nper channel", color="#1e5f3a", fontsize=8)
for bx in [3.1, 5.8, 8.5]:
    arrow(ax, bx + 0.8, 2.2, 10.1, 2.2)

ax.text(6, 0.4, "19 channels × 13 features = 247 features per 1-second EEG window",
        ha="center", fontsize=9, color=YELLOW, fontweight="bold")

plt.tight_layout(pad=0.3)
plt.savefig(os.path.join(OUT, "fig2_feature_pipeline.pdf"), bbox_inches="tight", dpi=150)
plt.savefig(os.path.join(OUT, "fig2_feature_pipeline.png"), bbox_inches="tight", dpi=150)
plt.close()
print("  fig2_feature_pipeline saved")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3  Signal Processing (bandpass + downsampling visual)
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(0)
t500 = np.linspace(0, 1, 500)
t128 = np.linspace(0, 1, 128)
raw  = (np.sin(2*np.pi*10*t500) + 0.5*np.sin(2*np.pi*30*t500)
        + 0.3*np.sin(2*np.pi*60*t500) + 0.8*np.random.randn(500))
filt = np.sin(2*np.pi*10*t128) + 0.5*np.sin(2*np.pi*30*t128)

fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=False)
fig.patch.set_facecolor(DARK_BG)

axes[0].plot(t500, raw, color=BLUE, lw=0.7, alpha=0.85)
axes[0].set_title("Raw EEG Signal (500 Hz, unfiltered)", color=TEXT, pad=4)
axes[0].set_ylabel("Amplitude (µV)", color=SUBTEXT)
axes[0].set_facecolor(MID_BG)
axes[0].axhline(0, color="#444", lw=0.5)
axes[0].text(0.98, 0.92, "Contains muscle artefacts\n& high-freq noise",
             transform=axes[0].transAxes, ha="right", fontsize=7.5, color=YELLOW,
             style="italic")

axes[1].plot(t128, filt, color=GREEN, lw=1.2)
axes[1].set_title("After Bandpass (0.5–45 Hz) & Downsampling to 128 Hz", color=TEXT, pad=4)
axes[1].set_ylabel("Amplitude (µV)", color=SUBTEXT)
axes[1].set_xlabel("Time (s)", color=SUBTEXT)
axes[1].set_facecolor(MID_BG)
axes[1].axhline(0, color="#444", lw=0.5)

fig.suptitle("EEG Preprocessing: Filtering and Downsampling", fontsize=11,
             fontweight="bold", color=TEXT, y=1.01)
plt.tight_layout(pad=0.8)
plt.savefig(os.path.join(OUT, "fig3_signal_processing.pdf"), bbox_inches="tight", dpi=150)
plt.savefig(os.path.join(OUT, "fig3_signal_processing.png"), bbox_inches="tight", dpi=150)
plt.close()
print("  fig3_signal_processing saved")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4  Model Architecture (ensemble diagram)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)
ax.text(5.5, 4.75, "NeuroWatch Ensemble Classifier Architecture",
        ha="center", fontsize=12, fontweight="bold", color=TEXT)

box(ax, 1.3, 2.5, 2.0, 0.8, "75 Selected Features", "(StandardScaler applied)", color="#1e3a5f")
arrow(ax, 2.3, 3.0, 3.5, 3.5)
arrow(ax, 2.3, 2.0, 3.5, 1.5)

box(ax, 4.5, 3.6, 2.0, 0.8, "SVM", "C=2, RBF kernel\nclass_weight={0:1, 1:1.8, 2:1}", color="#3a1e5f")
box(ax, 4.5, 1.4, 2.0, 0.8, "Random Forest", "300 trees, max_depth=8\nclass_weight=balanced", color="#1e5f3a")

ax.text(4.5, 3.6+0.55, "40%", ha="center", fontsize=9, color=PURPLE, fontweight="bold")
ax.text(4.5, 1.4+0.55, "60%", ha="center", fontsize=9, color=GREEN, fontweight="bold")

box(ax, 7.5, 3.6, 2.0, 0.8, "P_SVM", "P(Normal | Pre-Seizure | Seizure)", color="#3a1e5f", fontsize=7.5)
box(ax, 7.5, 1.4, 2.0, 0.8, "P_RF", "P(Normal | Pre-Seizure | Seizure)", color="#1e5f3a", fontsize=7.5)

arrow(ax, 5.5, 3.6, 6.5, 3.6)
arrow(ax, 5.5, 1.4, 6.5, 1.4)
arrow(ax, 8.5, 3.6, 9.2, 2.7)
arrow(ax, 8.5, 1.4, 9.2, 2.3)

box(ax, 9.8, 2.5, 1.8, 1.2, "Weighted\nCombination\n0.4·P_SVM\n+0.6·P_RF", color=ACCENT, fontsize=7.5)

ax.text(5.5, 0.3, "Normal threshold correction: if argmax = Pre-Seizure AND P(Normal) ≥ θ  →  predict Normal",
        ha="center", fontsize=8, color=YELLOW, style="italic")

plt.tight_layout(pad=0.3)
plt.savefig(os.path.join(OUT, "fig4_model_architecture.pdf"), bbox_inches="tight", dpi=150)
plt.savefig(os.path.join(OUT, "fig4_model_architecture.png"), bbox_inches="tight", dpi=150)
plt.close()
print("  fig4_model_architecture saved")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5  Model Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
models    = ["MODELS_V1\n(ANOVA)", "MODELS_FS75\n(RF sel.)", "RF60+SVM40\n(C=1)", "RF60+SVM40_C2\n(C=2, final)"]
accuracy  = [72.5, 76.1, 75.0, 77.1]
sz_recall = [78.0, 85.1, 84.9, 86.7]
f1_scores = [72.0, 75.8, 74.8, 77.1]

x = np.arange(len(models))
w = 0.25

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

b1 = ax.bar(x - w, accuracy,  w, label="Overall Accuracy (%)", color="#2563eb", alpha=0.9, edgecolor="#1e3a8a")
b2 = ax.bar(x,     f1_scores, w, label="F1-score (weighted %)", color="#16a34a", alpha=0.9, edgecolor="#14532d")
b3 = ax.bar(x + w, sz_recall, w, label="Seizure Recall (%)",    color="#dc2626", alpha=0.9, edgecolor="#7f1d1d")

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}",
                ha="center", va="bottom", fontsize=7.5, color="#111111")

ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8.5, color="#111111")
ax.set_ylim(60, 95)
ax.set_ylabel("Score (%)", color="#111111")
ax.tick_params(colors="#111111")
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#cccccc")
ax.set_title("Model Performance Comparison Across Training Iterations", fontsize=11,
             fontweight="bold", color="#111111", pad=8)
ax.legend(loc="lower right", fontsize=8, framealpha=0.8, edgecolor="#cccccc")
ax.yaxis.grid(True, alpha=0.4, color="#dddddd"); ax.set_axisbelow(True)

plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(OUT, "fig5_results_comparison.pdf"), bbox_inches="tight", dpi=150)
plt.savefig(os.path.join(OUT, "fig5_results_comparison.png"), bbox_inches="tight", dpi=150)
plt.close()
print("  fig5_results_comparison saved")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6  Final Confusion Matrix  (RF60+SVM40_C2)
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib.colors as mcolors

cm = np.array([[280, 87, 22],
               [83, 283, 24],
               [16,  36, 338]])

labels = ["Normal", "Pre-Seizure", "Seizure"]
fig, ax = plt.subplots(figsize=(7, 5.5))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(MID_BG)

cmap = plt.cm.Blues
norm = mcolors.Normalize(vmin=0, vmax=cm.max())
im   = ax.imshow(cm, cmap=cmap, norm=norm, aspect="auto")

for i in range(3):
    for j in range(3):
        val = cm[i, j]
        col = "white" if val < cm.max()*0.5 else "black"
        ax.text(j, i, str(val), ha="center", va="center",
                fontsize=14, fontweight="bold", color=col)

ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
ax.set_xticklabels(labels, fontsize=10); ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Predicted Label", fontsize=11, color=TEXT)
ax.set_ylabel("True Label", fontsize=11, color=TEXT)
ax.set_title(f"Confusion Matrix — RF60+SVM40\\_C2 (Test set, n=1169)\nAcc: 77.1%  |  Seizure Recall: 86.7%",
             fontsize=10, fontweight="bold", color=TEXT, pad=10)
plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(OUT, "fig6_confusion_matrix.pdf"), bbox_inches="tight", dpi=150)
plt.savefig(os.path.join(OUT, "fig6_confusion_matrix.png"), bbox_inches="tight", dpi=150)
plt.close()
print("  fig6_confusion_matrix saved")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7  Feature-count ablation (k vs validation metrics)
# ─────────────────────────────────────────────────────────────────────────────
k_vals   = [25,   50,   75,   100,  247]
abl_acc  = [88.7, 94.0, 96.0, 96.3, 96.3]
abl_f1   = [88.7, 94.0, 96.0, 96.3, 96.3]
abl_rec  = [85.3, 90.7, 93.3, 94.7, 94.7]

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.plot(k_vals, abl_acc, "o-",  color="#2563eb", linewidth=2, markersize=7,
        label="Overall Accuracy (%)")
ax.plot(k_vals, abl_f1,  "s--", color="#16a34a", linewidth=2, markersize=7,
        label="Weighted F1 (%)")
ax.plot(k_vals, abl_rec, "^:",  color="#dc2626", linewidth=2, markersize=7,
        label="Seizure Recall (%)")

# Annotate each point
for k, a, f, r in zip(k_vals, abl_acc, abl_f1, abl_rec):
    ax.annotate(f"{a:.1f}", (k, a), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color="#2563eb")
    ax.annotate(f"{r:.1f}", (k, r), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=8, color="#dc2626")

# Highlight the selected k=75
ax.axvline(75, color="#888888", linestyle="--", linewidth=1.2, alpha=0.7)
ax.text(75 + 3, 85.5, "Selected\n$k=75$", fontsize=8.5, color="#444444",
        va="bottom")

ax.set_xticks(k_vals)
ax.set_xticklabels([str(k) for k in k_vals], color="#111111")
ax.tick_params(colors="#111111")
ax.set_xlabel("Number of features retained ($k$)", color="#111111", fontsize=11)
ax.set_ylabel("Score (%)", color="#111111", fontsize=11)
ax.set_ylim(82, 99)
ax.set_title("Feature-Count Ablation — RF60+SVM40\\_C2 (Validation Set)",
             fontsize=11, fontweight="bold", color="#111111", pad=8)
ax.legend(loc="lower right", fontsize=9, framealpha=0.9, edgecolor="#cccccc")
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#cccccc")
ax.yaxis.grid(True, alpha=0.4, color="#dddddd")
ax.set_axisbelow(True)

plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(OUT, "fig7_feature_ablation.pdf"), bbox_inches="tight", dpi=150)
plt.savefig(os.path.join(OUT, "fig7_feature_ablation.png"), bbox_inches="tight", dpi=150)
plt.close()
print("  fig7_feature_ablation saved")


print("\n  All figures saved to:", OUT)
