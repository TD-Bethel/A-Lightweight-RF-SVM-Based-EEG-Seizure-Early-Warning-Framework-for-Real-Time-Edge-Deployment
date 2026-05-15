"""
generate_gantt.py  --  NeuroWatch project Gantt chart (static PNG/PDF)

Run from REPORT/ or anywhere:
    python generate_gantt.py

Output: REPORT/figures/fig_gantt_timeline.png  (also .pdf)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Output path ──────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# ── Colour palette (matches neurowatch_animations.py) ────────────────────────
DARK_BG     = "#1a1a2e"
MID_BG      = "#16213e"
BLUE_ACC    = "#4cc9f0"
PURPLE_ACC  = "#c77dff"
TEAL        = "#2a9d8f"
ORANGE_ACC  = "#e76f51"
NEON_GREEN  = "#00CC66"
GOLD_ACC    = "#e9c46a"
AMBER       = "#f4a261"
NEON_YELLOW = "#FFD633"

# ── Data ─────────────────────────────────────────────────────────────────────
MONTHS = ["Aug\n2025", "Sep", "Oct", "Nov", "Jan\n2026", "Feb", "Mar", "Apr", "May"]
N = len(MONTHS)   # 9 columns

# (task label, start_col, end_col_excl, colour)
TASKS = [
    ("Literature Review",        0, 2, BLUE_ACC),
    ("Methodology Design",       1, 4, PURPLE_ACC),
    ("ML Model Development",     2, 4, TEAL),
    ("Circuit Simulation",       3, 4, ORANGE_ACC),
    ("Hardware Implementation",  4, 6, NEON_GREEN),
    ("Monitoring Dashboard",     5, 7, BLUE_ACC),
    ("Patient User Interface",   6, 8, GOLD_ACC),
    ("Results & Analysis",       7, 9, AMBER),
    ("Report Writing",           7, 9, NEON_YELLOW),
]
T = len(TASKS)

# ── Figure setup ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ── Semester background shading ───────────────────────────────────────────────
SEM1_COLS = 4   # columns 0-3  (Aug–Nov 2025)
SEM2_COLS = 5   # columns 4-8  (Jan–May 2026)

ax.axvspan(0, SEM1_COLS, ymin=0, ymax=1,
           facecolor=BLUE_ACC, alpha=0.07, zorder=0)
ax.axvspan(SEM1_COLS, N, ymin=0, ymax=1,
           facecolor=NEON_GREEN, alpha=0.07, zorder=0)

# Semester border lines
for x, clr in [(0, BLUE_ACC), (SEM1_COLS, GOLD_ACC), (N, NEON_GREEN)]:
    ax.axvline(x, color=clr, linewidth=0.9, alpha=0.45, zorder=1)

# Semester divider dashed
ax.axvline(SEM1_COLS, color=GOLD_ACC, linewidth=1.6,
           linestyle="--", alpha=0.80, zorder=2)

# ── Grid lines (faint vertical) ───────────────────────────────────────────────
for i in range(1, N):
    ax.axvline(i, color="#aaaaaa", linewidth=0.3, alpha=0.4, zorder=1)

# ── Task bars ─────────────────────────────────────────────────────────────────
BAR_H = 0.55

for j, (name, s, e, clr) in enumerate(TASKS):
    y = T - 1 - j          # top task at highest y so it reads top-to-bottom
    width = e - s

    bar = FancyBboxPatch(
        (s + 0.04, y - BAR_H / 2),
        width - 0.08, BAR_H,
        boxstyle="round,pad=0.04",
        facecolor=clr, edgecolor="none",
        alpha=0.88, zorder=3,
    )
    ax.add_patch(bar)

    # Task label inside bar if wide enough, else to the right
    label_x = s + width / 2
    ax.text(label_x, y, name,
            ha="center", va="center",
            fontsize=8.8, color=DARK_BG, fontweight="bold", zorder=4)

# ── Row separator lines ───────────────────────────────────────────────────────
for j in range(T + 1):
    y = T - j - 0.5
    ax.axhline(y, color="#aaaaaa", linewidth=0.25, alpha=0.5, zorder=1)

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xlim(0, N)
ax.set_ylim(-0.5, T - 0.5)

ax.set_xticks([i + 0.5 for i in range(N)])
ax.set_xticklabels(MONTHS, fontsize=10, color="#222222")
ax.tick_params(axis="x", length=0, pad=6)

ax.set_yticks([])

# Top x-axis (semester labels)
ax2 = ax.twiny()
ax2.set_xlim(0, N)
ax2.set_facecolor("white")
ax2.set_xticks([SEM1_COLS / 2, SEM1_COLS + SEM2_COLS / 2])
ax2.set_xticklabels(
    ["SEMESTER 1  (Aug – Nov 2025)", "SEMESTER 2  (Jan – May 2026)"],
    fontsize=10, fontweight="bold",
)
ax2.tick_params(axis="x", length=0, pad=6)
ax2.xaxis.set_tick_params(labelcolor=BLUE_ACC)
ax2.get_xticklabels()[1].set_color(NEON_GREEN)

# Remove all spines
for spine in ax.spines.values():
    spine.set_visible(False)
for spine in ax2.spines.values():
    spine.set_visible(False)

# ── Milestone markers ─────────────────────────────────────────────────────────
ax.annotate("✓ Circuit Simulation\n    complete",
            xy=(SEM1_COLS, T - 1 - 3),
            xytext=(SEM1_COLS - 0.15, T - 1 - 3 - 1.1),
            fontsize=8, color=NEON_GREEN, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=NEON_GREEN,
                            lw=1.2, connectionstyle="arc3,rad=-0.25"),
            ha="center", zorder=5)

ax.annotate("✓ Hardware & Report\n    complete",
            xy=(N, T - 1 - 8),
            xytext=(N - 0.8, T - 1 - 8 + 1.2),
            fontsize=8, color=NEON_GREEN, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=NEON_GREEN,
                            lw=1.2, connectionstyle="arc3,rad=0.25"),
            ha="center", zorder=5)

# ── Legend patches ────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=BLUE_ACC,    label="Literature & Theory"),
    mpatches.Patch(facecolor=PURPLE_ACC,  label="Design & Methodology"),
    mpatches.Patch(facecolor=TEAL,        label="Model Development"),
    mpatches.Patch(facecolor=ORANGE_ACC,  label="Simulation"),
    mpatches.Patch(facecolor=NEON_GREEN,  label="Hardware"),
    mpatches.Patch(facecolor=GOLD_ACC,    label="UI & Dashboard"),
    mpatches.Patch(facecolor=AMBER,       label="Results & Analysis"),
    mpatches.Patch(facecolor=NEON_YELLOW, label="Report Writing"),
]
leg = ax.legend(
    handles=legend_items,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=4,
    frameon=False,
    fontsize=8.5,
    labelcolor="#222222",
    handlelength=1.2,
    handletextpad=0.5,
    columnspacing=1.2,
)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Project Timeline",
    fontsize=15, fontweight="bold", color="#111111", y=1.04,
)
ax.set_title(
    "BIUST Final Year Project  |  Aug 2025 – May 2026",
    fontsize=9, color="#555555", pad=4,
)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.12, 1, 1])

png_path = os.path.join(OUT, "fig_gantt_timeline.png")
pdf_path = os.path.join(OUT, "fig_gantt_timeline.pdf")
fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
