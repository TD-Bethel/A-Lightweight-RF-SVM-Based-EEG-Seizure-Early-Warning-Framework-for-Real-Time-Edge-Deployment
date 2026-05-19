"""
generate_weight_figures.py  --  Table image + bar chart for weight sweep results
Outputs:
  REPORT/figures/fig_weight_sweep_table.png
  REPORT/figures/fig_weight_sweep_chart.png

Run from repo root:
    python REPORT/generate_weight_figures.py
"""

import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE    = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

JSON_PATH = os.path.join(FIG_DIR, "weight_sweep_results.json")
with open(JSON_PATH) as f:
    data = [d for d in json.load(f) if d["rf_weight"] != 0.4 or d["svm_weight"] != 0.6]

LABELS      = [d["label"]            for d in data]
TRAIN_ACC   = [d["train_accuracy"]   for d in data]
TEST_ACC    = [d["accuracy"]         for d in data]
GAP         = [d["overfitting_gap"]  for d in data]
MACRO_F1    = [d["macro_f1"]         for d in data]
PREC_F1     = [d["per_class"]["Pre-Seizure"]["f1"] for d in data]
SZ_F1       = [d["per_class"]["Seizure"]["f1"]     for d in data]
SZ_REC      = [d["seizure_recall"]   for d in data]

PROD_IDX = 1   # RF 60 / SVM 40

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
COL_PROD   = "#2196F3"   # blue  — production row
COL_OTHER  = "#90CAF9"   # light blue
COL_BEST   = "#4CAF50"   # green — best value cell
COL_BEST_T = "#FFFFFF"   # white text on green
COL_PROD_T = "#FFFFFF"   # white text on blue
COL_HDR_BG = "#1565C0"   # dark blue header
COL_HDR_T  = "#FFFFFF"
COL_ALT    = "#E3F2FD"   # alternating row tint

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Table image
# ─────────────────────────────────────────────────────────────────────────────
COL_HEADERS = [
    "RF / SVM\nWeights",
    "Train\nAcc.",
    "Test\nAcc.",
    "Macro\nF1",
    "Pre-ictal\nF1",
    "Seizure\nF1",
    "Seizure\nRecall",
]

table_data = []
for d in data:
    row = [
        d["label"],
        f"{d['train_accuracy']:.4f}",
        f"{d['accuracy']:.4f}",
        f"{d['macro_f1']:.4f}",
        f"{d['per_class']['Pre-Seizure']['f1']:.4f}",
        f"{d['per_class']['Seizure']['f1']:.4f}",
        f"{d['seizure_recall']:.4f}",
    ]
    table_data.append(row)

n_rows = len(table_data)
n_cols = len(COL_HEADERS)

fig_t, ax_t = plt.subplots(figsize=(14, 3.8))
fig_t.patch.set_facecolor("white")
ax_t.set_facecolor("white")
ax_t.axis("off")

tbl = ax_t.table(
    cellText=table_data,
    colLabels=COL_HEADERS,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.9)

# Style header row
for c in range(n_cols):
    cell = tbl[0, c]
    cell.set_facecolor(COL_HDR_BG)
    cell.set_text_props(color=COL_HDR_T, fontweight="bold")
    cell.set_edgecolor("#BBBBBB")

# Identify best value per column (excluding label col)
best_train  = max(TRAIN_ACC)
best_test   = max(TEST_ACC)
best_mf1    = max(MACRO_F1)
best_pf1    = max(PREC_F1)
best_sf1    = max(SZ_F1)
best_srec   = max(SZ_REC)

bests = [None, best_train, best_test, best_mf1, best_pf1, best_sf1, best_srec]

for r in range(n_rows):
    is_prod = (r == PROD_IDX)
    row_bg  = COL_PROD if is_prod else (COL_ALT if r % 2 == 0 else "white")
    row_fg  = COL_PROD_T if is_prod else "#111111"

    for c in range(n_cols):
        cell = tbl[r + 1, c]
        cell.set_edgecolor("#CCCCCC")

        # check if best value cell
        is_best = False
        if bests[c] is not None:
            val_str = table_data[r][c]
            try:
                val = float(val_str)
                is_best = (val == bests[c])
            except ValueError:
                pass

        if is_best and not is_prod:
            cell.set_facecolor(COL_BEST)
            cell.set_text_props(color=COL_BEST_T, fontweight="bold")
        elif is_best and is_prod:
            cell.set_facecolor("#1E88E5")
            cell.set_text_props(color=COL_BEST_T, fontweight="bold")
        elif is_prod:
            cell.set_facecolor(COL_PROD)
            cell.set_text_props(color=row_fg, fontweight="bold")
        else:
            cell.set_facecolor(row_bg)
            cell.set_text_props(color=row_fg)

# Legend
prod_patch = mpatches.Patch(facecolor=COL_PROD,  label="Production setting (RF 60 / SVM 40)")
best_patch = mpatches.Patch(facecolor=COL_BEST,  label="Best value in column")
ax_t.legend(handles=[prod_patch, best_patch],
            loc="lower center", bbox_to_anchor=(0.5, -0.18),
            ncol=2, frameon=False, fontsize=9,
            labelcolor="#222222")

fig_t.suptitle("Ensemble Weight Sweep — Validation Metrics",
               fontsize=13, fontweight="bold", color="#111111", y=1.01)

table_png = os.path.join(FIG_DIR, "fig_weight_sweep_table.png")
fig_t.savefig(table_png, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig_t)
print(f"Saved table: {table_png}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────
x = np.arange(len(LABELS))
width = 0.16

METRICS = [
    ("Test Accuracy",   TEST_ACC,  "#2196F3"),
    ("Macro F1",        MACRO_F1,  "#9C27B0"),
    ("Pre-ictal F1",    PREC_F1,   "#FF9800"),
    ("Seizure F1",      SZ_F1,     "#4CAF50"),
    ("Seizure Recall",  SZ_REC,    "#F44336"),
]

fig_c, ax_c = plt.subplots(figsize=(13, 5.5))
fig_c.patch.set_facecolor("white")
ax_c.set_facecolor("white")

offsets = np.linspace(-(len(METRICS) - 1) / 2, (len(METRICS) - 1) / 2, len(METRICS))

bars_all = []
for idx, (label, vals, colour) in enumerate(METRICS):
    bars = ax_c.bar(
        x + offsets[idx] * width, vals,
        width=width * 0.92,
        color=colour, alpha=0.85,
        label=label,
        zorder=3,
    )
    bars_all.append(bars)

# Annotate production bars with a marker
prod_x = PROD_IDX
for idx, (_, vals, _) in enumerate(METRICS):
    bar_x = prod_x + offsets[idx] * width
    ax_c.annotate("★", xy=(bar_x, vals[prod_x]),
                  xytext=(bar_x, vals[prod_x] + 0.006),
                  ha="center", va="bottom", fontsize=8,
                  color="#1565C0", fontweight="bold")

# Shade the production column
ax_c.axvspan(PROD_IDX - 0.45, PROD_IDX + 0.45,
             facecolor="#2196F3", alpha=0.07, zorder=0)

ax_c.set_xticks(x)
ax_c.set_xticklabels(LABELS, fontsize=10, color="#222222")
ax_c.set_ylabel("Score", fontsize=11)
ax_c.set_ylim(0.60, 0.96)
ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax_c.tick_params(axis="both", labelsize=9)

ax_c.grid(axis="y", color="#CCCCCC", linewidth=0.5, alpha=0.7, zorder=0)
ax_c.set_axisbelow(True)
for spine in ax_c.spines.values():
    spine.set_edgecolor("#CCCCCC")

handles1, labels1 = ax_c.get_legend_handles_labels()
ax_c.legend(handles1, labels1,
            loc="lower right", frameon=True, fontsize=8.5,
            framealpha=0.9, edgecolor="#CCCCCC")

ax_c.annotate("★ = Production\n   (RF 60 / SVM 40)",
              xy=(PROD_IDX, 0.614), ha="center", fontsize=8,
              color="#1565C0", style="italic")

fig_c.suptitle("Ensemble Weight Sweep — Performance Metrics",
               fontsize=13, fontweight="bold", color="#111111")
ax_c.set_title("Metrics across RF / SVM blend ratios  |  held-out test set",
               fontsize=9, color="#555555", pad=5)

chart_png = os.path.join(FIG_DIR, "fig_weight_sweep_chart.png")
fig_c.savefig(chart_png, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig_c)
print(f"Saved chart: {chart_png}")
