"""
neurowatch_animations.py  --  Manim scenes for the NeuroWatch FYP report

SCENES
------
  EEGWaveformScene       -- Normal / Pre-Seizure / Seizure EEG traces
  FeaturePipelineScene   -- 19-channel feature extraction block diagram
  EnsembleScene          -- Hybrid SVM + RF weighted blend (overview)
  SystemFlowScene        -- End-to-end pipeline → LCD / Dashboard / SMS
  ConfusionMatrixScene   -- Animated 3×3 confusion matrix
  RFScene                -- Random Forest deep-dive (247 dots → scout RF → trees → voting)
  SVMScene               -- SVM deep-dive (scatter → scaling → RBF → boundary → Platt)
  MLEnsembleDetailScene  -- Detailed ensemble blend, threshold correction, final flash
  ProjectGanttScene      -- Gantt chart project timeline Aug 2025 – May 2026

USAGE  (run from the REPORT/ folder or adjust --media-dir)
------
  manim -pql neurowatch_animations.py EEGWaveformScene
  manim -pql neurowatch_animations.py FeaturePipelineScene
  manim -pql neurowatch_animations.py EnsembleScene
  manim -pql neurowatch_animations.py SystemFlowScene
  manim -pql neurowatch_animations.py ConfusionMatrixScene
  manim -pql neurowatch_animations.py RFScene
  manim -pql neurowatch_animations.py SVMScene
  manim -pql neurowatch_animations.py MLEnsembleDetailScene
  manim -pql neurowatch_animations.py ProjectGanttScene

  # render the ML deep-dive trilogy:
  manim -qh neurowatch_animations.py RFScene SVMScene MLEnsembleDetailScene

  # render ALL at high quality:
  manim -qh neurowatch_animations.py EEGWaveformScene FeaturePipelineScene EnsembleScene SystemFlowScene ConfusionMatrixScene RFScene SVMScene MLEnsembleDetailScene ProjectGanttScene

Quality flags: -ql 480p (fast draft) | -qm 720p | -qh 1080p | -qk 4K
"""

from manim import *
import numpy as np

# ── NeuroWatch colour palette ────────────────────────────────────────────────
NEON_GREEN   = "#00CC66"
NEON_YELLOW  = "#FFD633"
NEON_RED     = "#FF3333"
DARK_BG      = "#1a1a2e"
MID_BG       = "#16213e"
BLUE_ACC     = "#4cc9f0"
PURPLE_ACC   = "#c77dff"
TEAL         = "#2a9d8f"
ORANGE_ACC   = "#e76f51"
GOLD_ACC     = "#e9c46a"
NAVY         = "#264653"
AMBER        = "#f4a261"


# ── shared helper ────────────────────────────────────────────────────────────
def make_box(label: str, fill: str,
             w: float = 2.9, h: float = 0.90,
             fsize: int = 18) -> VGroup:
    """Rounded rectangle with centred text."""
    rect = RoundedRectangle(
        corner_radius=0.15, width=w, height=h,
        fill_color=fill, fill_opacity=0.88,
        stroke_color=WHITE, stroke_width=1.5,
    )
    txt = Text(label, font_size=fsize, color=WHITE)
    txt.move_to(rect.get_center())
    return VGroup(rect, txt)


def _save_snapshot(scene: "Scene", stage_name: str) -> None:
    """Capture the current camera frame and write it to REPORT/stage_images/."""
    import os
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage_images")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{type(scene).__name__}_{stage_name}.png")
    scene.camera.get_image().save(path)
    print(f"  [snapshot] {path}")


def _save_report(scene: "Scene", name: str) -> None:
    """Save white-background report image to REPORT/stage_images/report/."""
    import os
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage_images", "report")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    scene.camera.get_image().save(path)
    print(f"  [report] {path}")


# ════════════════════════════════════════════════════════════════════════════
#  SCENE 1 – EEG waveforms for the three patient states
# ════════════════════════════════════════════════════════════════════════════
class EEGWaveformScene(Scene):
    """
    Draws three synthetic EEG traces — Normal, Pre-Seizure, Seizure —
    using the same colour scheme as the NeuroWatch dashboard.
    """

    def construct(self):
        self.camera.background_color = DARK_BG
        np.random.seed(42)

        # ── Title ─────────────────────────────────────────────────────────
        title = Text("EEG Signal States", font_size=38, color=WHITE, weight=BOLD)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title))

        # ── Synthetic signals ─────────────────────────────────────────────
        t = np.linspace(0, 4 * PI, 500)

        normal_y = (0.30 * np.sin(t)
                    + 0.20 * np.sin(2.3 * t + 0.5)
                    + 0.10 * np.sin(7.1 * t)
                    + 0.07 * np.random.randn(500))

        envelope = np.linspace(0.25, 1.0, 500)
        preseiz_y = envelope * (np.sin(2.0 * t)
                                + 0.40 * np.sin(4.8 * t)
                                + 0.08 * np.random.randn(500))

        seizure_y = (1.20 * np.sin(3.0 * t)
                     + 0.35 * np.sign(np.sin(6.0 * t))
                     + 0.05 * np.random.randn(500))

        configs = [
            ("Normal",       normal_y,   NEON_GREEN),
            ("Pre-Seizure",  preseiz_y,  NEON_YELLOW),
            ("Seizure",      seizure_y,  NEON_RED),
        ]
        y_positions = [1.85, 0.0, -1.85]

        for (label, sig, color), ypos in zip(configs, y_positions):
            ax = Axes(
                x_range=[0, 4 * PI, PI],
                y_range=[-1.7, 1.7, 0.8],
                x_length=9.5,
                y_length=1.50,
                axis_config={"color": GRAY_C, "stroke_width": 1.0},
                tips=False,
            )
            ax.move_to(RIGHT * 0.6 + UP * ypos)

            sig_norm = sig / (np.max(np.abs(sig)) + 1e-8) * 1.45
            graph = ax.plot_line_graph(
                x_values=t,
                y_values=sig_norm,
                line_color=color,
                stroke_width=2.2,
                add_vertex_dots=False,
            )

            dot = Dot(color=color, radius=0.11)
            lbl = Text(label, font_size=22, color=color, weight=BOLD)
            lbl_grp = VGroup(dot, lbl).arrange(RIGHT, buff=0.12)
            lbl_grp.next_to(ax, LEFT, buff=0.22)

            self.play(Create(ax), FadeIn(lbl_grp), run_time=0.45)
            self.play(Create(graph), run_time=1.10)

        self.wait(1.5)


# ════════════════════════════════════════════════════════════════════════════
#  SCENE 2 – Feature extraction pipeline
# ════════════════════════════════════════════════════════════════════════════
class FeaturePipelineScene(Scene):
    """
    Animated block diagram: Raw EEG → filter → resample → 13 features/ch
    → 247-dim vector → RF selector → 75 features.
    Laid out in two rows of three to fit the canvas.
    """

    def construct(self):
        self.camera.background_color = DARK_BG

        title = Text("Feature Extraction Pipeline", font_size=36,
                     color=WHITE, weight=BOLD)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title))

        # ── Two rows of 3 boxes ───────────────────────────────────────────
        top_specs = [
            ("Raw EEG\n19 ch  x  500 Hz",      NAVY),
            ("Bandpass Filter\n0.5 - 45 Hz",  TEAL),
            ("Resample\n500 -> 128 Hz",       TEAL),
        ]
        bot_specs = [
            ("13 Features\nper Channel",       ORANGE_ACC),
            ("Concatenate\n247-dim Vector",    GOLD_ACC),
            ("RF Selector\nTop 75 Kept",       AMBER),
        ]

        def build_row(specs):
            grp = VGroup(*[make_box(lbl, col, w=3.3, h=1.00, fsize=17)
                           for lbl, col in specs])
            grp.arrange(RIGHT, buff=0.55)
            return grp

        top_row = build_row(top_specs)
        bot_row = build_row(bot_specs)
        top_row.shift(UP * 1.20)
        bot_row.shift(DOWN * 1.20)

        # Horizontal arrows within rows
        def h_arrows(row):
            return VGroup(*[
                Arrow(row[i].get_right(), row[i + 1].get_left(),
                      color=WHITE, buff=0.05, stroke_width=2.0,
                      tip_length=0.18, max_tip_length_to_length_ratio=0.5)
                for i in range(len(row) - 1)
            ])

        top_arr = h_arrows(top_row)
        bot_arr = h_arrows(bot_row)

        # Elbow arrow: end of top row -> start of bot row
        mid_pt = (top_row[-1].get_bottom() + bot_row[0].get_top()) / 2
        elbow = VGroup(
            Line(top_row[-1].get_bottom(), mid_pt + DOWN * 0.05, color=YELLOW, stroke_width=2.5),
            Arrow(mid_pt + DOWN * 0.05, bot_row[0].get_top(),
                  color=YELLOW, buff=0.0, stroke_width=2.5, tip_length=0.18),
        )

        # Animate
        for i, box in enumerate(top_row):
            self.play(FadeIn(box, shift=UP * 0.15), run_time=0.35)
            if i < len(top_arr):
                self.play(GrowArrow(top_arr[i]), run_time=0.28)

        self.play(Create(elbow[0]), GrowArrow(elbow[1]), run_time=0.4)

        for i, box in enumerate(bot_row):
            self.play(FadeIn(box, shift=DOWN * 0.15), run_time=0.35)
            if i < len(bot_arr):
                self.play(GrowArrow(bot_arr[i]), run_time=0.28)

        # Equation summary (plain Text — no LaTeX required)
        eq_left  = Text("19 × 13 = 247",          font_size=34, color=WHITE)
        eq_arrow = Text("  ⟶  RF selector  ⟶  ", font_size=30, color=GRAY_A)
        eq_right = Text("75 features",                   font_size=34, color=YELLOW, weight=BOLD)
        eq = VGroup(eq_left, eq_arrow, eq_right).arrange(RIGHT, buff=0.1)
        eq.to_edge(DOWN, buff=0.35)
        self.play(Write(eq))
        self.wait(1.5)


# ════════════════════════════════════════════════════════════════════════════
#  SCENE 3 – Hybrid SVM + RF ensemble
# ════════════════════════════════════════════════════════════════════════════
class EnsembleScene(Scene):
    """
    Left-to-right flow: Feature Vector -> SVM (x0.40) and RF (x0.60)
    -> Weighted Blend -> Threshold Correction -> Final Prediction.
    Confidence bars appear beside the prediction box.
    """

    def construct(self):
        self.camera.background_color = DARK_BG

        title = Text("Hybrid SVM + RF Ensemble", font_size=36,
                     color=WHITE, weight=BOLD)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title))

        # ── Boxes ─────────────────────────────────────────────────────────
        inp  = make_box("Feature Vector\n(75 dims)",          NAVY,       w=2.7, h=1.00)
        svm  = make_box("SVM  x0.40\nRBF, C=2",             TEAL,       w=2.5, h=1.00)
        rf   = make_box("Random Forest  x0.60\n300 trees",   ORANGE_ACC, w=2.8, h=1.00)
        bld  = make_box("Weighted Blend\n0.40 * pSVM + 0.60 * pRF", GOLD_ACC,   w=3.3, h=1.00)
        pred = make_box("Threshold Correction\nθ = 0.39",    PURPLE_ACC, w=2.9, h=1.00)

        inp.move_to(LEFT * 4.8)
        svm.move_to(LEFT * 1.6 + UP  * 1.25)
        rf.move_to( LEFT * 1.6 + DOWN * 1.25)
        bld.move_to(RIGHT * 1.8)
        pred.move_to(RIGHT * 5.0)

        # ── Arrows ────────────────────────────────────────────────────────
        def arr(start, end, col=WHITE):
            return Arrow(start, end, color=col, buff=0.06,
                         stroke_width=2.0, tip_length=0.16,
                         max_tip_length_to_length_ratio=0.5)

        a_i_s = arr(inp.get_right(), svm.get_left())
        a_i_r = arr(inp.get_right(), rf.get_left())
        a_s_b = arr(svm.get_right(), bld.get_left(),  col=TEAL)
        a_r_b = arr(rf.get_right(),  bld.get_left(),  col=ORANGE_ACC)
        a_b_p = arr(bld.get_right(), pred.get_left())

        # ── Animate architecture ──────────────────────────────────────────
        self.play(FadeIn(inp))
        self.play(GrowArrow(a_i_s), GrowArrow(a_i_r))
        self.play(FadeIn(svm), FadeIn(rf))
        self.play(GrowArrow(a_s_b), GrowArrow(a_r_b))
        self.play(FadeIn(bld))
        self.play(GrowArrow(a_b_p))
        self.play(FadeIn(pred))

        # ── Probability bars beside pred ──────────────────────────────────
        probs  = [0.13, 0.30, 0.57]
        colors = [NEON_GREEN, NEON_YELLOW, NEON_RED]
        labels = ["Normal", "Pre-Seizure", "Seizure"]
        bar_w  = 2.2

        bars = VGroup()
        for p, col, lbl in zip(probs, colors, labels):
            bg  = Rectangle(width=bar_w, height=0.30,
                             fill_color="#222244", fill_opacity=1.0,
                             stroke_color=GRAY_C, stroke_width=1)
            fg  = Rectangle(width=bar_w * p, height=0.30,
                             fill_color=col, fill_opacity=0.90,
                             stroke_width=0)
            fg.align_to(bg, LEFT)
            lt  = Text(lbl, font_size=15, color=col).next_to(bg, LEFT, buff=0.08)
            pt  = Text(f"{p*100:.0f}%", font_size=15, color=WHITE).next_to(bg, RIGHT, buff=0.06)
            bars.add(VGroup(bg, fg, lt, pt))

        bars.arrange(DOWN, buff=0.18)
        bars.next_to(pred, DOWN, buff=0.40)

        self.play(FadeIn(bars))

        winner_box = SurroundingRectangle(bars[2], color=NEON_RED,
                                          buff=0.06, stroke_width=2.5)
        win_lbl = Text("SEIZURE DETECTED", font_size=19,
                       color=NEON_RED, weight=BOLD)
        win_lbl.next_to(bars[2], RIGHT, buff=0.18)
        self.play(Create(winner_box), Write(win_lbl))
        self.wait(1.5)


# ════════════════════════════════════════════════════════════════════════════
#  SCENE 4 – End-to-end system flow
# ════════════════════════════════════════════════════════════════════════════
class SystemFlowScene(Scene):
    """
    Vertical processing pipeline on the left, branching to three output
    channels on the right: LCD/LED/Buzzer, Remote Dashboard, SMS Alert.
    """

    def construct(self):
        self.camera.background_color = DARK_BG

        title = Text("NeuroWatch System Flow", font_size=36,
                     color=WHITE, weight=BOLD)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title))

        # ── Vertical pipeline ─────────────────────────────────────────────
        pipe_specs = [
            ("EEG Signal\n(Raspberry Pi 4)",           NAVY),
            ("Bandpass + Resample\n(0.5-45 Hz, 128 Hz)", TEAL),
            ("Feature Extraction\n(247 features)",      TEAL),
            ("RF Feature Selection\n(247 -> 75)",       AMBER),
            ("SVM + RF Ensemble\nPrediction",           ORANGE_ACC),
        ]

        pipe = VGroup(*[make_box(lbl, col, w=3.6, h=0.82, fsize=17)
                        for lbl, col in pipe_specs])
        pipe.arrange(DOWN, buff=0.28)
        pipe.shift(LEFT * 2.6 + DOWN * 0.25)

        pipe_arrows = VGroup(*[
            Arrow(pipe[i].get_bottom(), pipe[i + 1].get_top(),
                  color=WHITE, buff=0.04, stroke_width=2.0,
                  tip_length=0.15, max_tip_length_to_length_ratio=0.5)
            for i in range(len(pipe) - 1)
        ])

        # ── Output branches ───────────────────────────────────────────────
        out_specs = [
            ("LCD / LED / Buzzer\n(GPIO, Raspberry Pi)",  NEON_GREEN),
            ("Remote Dashboard\n(Streamlit, port 8501)",   BLUE_ACC),
            ("SMS Alert via Twilio\n(Seizure >= 30 s)",    NEON_RED),
        ]

        outs = VGroup(*[make_box(lbl, col, w=3.5, h=0.82, fsize=16)
                        for lbl, col in out_specs])
        outs.arrange(DOWN, buff=0.32)
        outs.shift(RIGHT * 3.1 + DOWN * 0.25)

        out_arrows = VGroup(*[
            Arrow(pipe[-1].get_right(), ob.get_left(),
                  color=YELLOW, buff=0.05, stroke_width=2.0,
                  tip_length=0.15, max_tip_length_to_length_ratio=0.5)
            for ob in outs
        ])

        # ── Animate pipeline ──────────────────────────────────────────────
        for i, box in enumerate(pipe):
            self.play(FadeIn(box, shift=DOWN * 0.10), run_time=0.38)
            if i < len(pipe_arrows):
                self.play(GrowArrow(pipe_arrows[i]), run_time=0.22)

        self.wait(0.2)

        # ── Animate outputs ───────────────────────────────────────────────
        for arrow, out_box in zip(out_arrows, outs):
            self.play(GrowArrow(arrow), FadeIn(out_box, shift=RIGHT * 0.10),
                      run_time=0.42)

        # Pulse the SMS box to emphasise the alert path
        self.play(Indicate(outs[2], color=NEON_RED, scale_factor=1.08))
        self.wait(1.2)


# ════════════════════════════════════════════════════════════════════════════
#  SCENE 5 – Confusion matrix fill-in
# ════════════════════════════════════════════════════════════════════════════
class ConfusionMatrixScene(Scene):
    """
    Animates the 3x3 confusion matrix for the final RF60+SVM40_C2 model.

    Matrix values derived from per-class precision / recall reported in
    the technical report (Acc = 77.1 %, n = 1169):

        True \\ Pred   Normal  Pre-Sz  Seizure
        Normal           280     87      22
        Pre-Seizure       83    283      24
        Seizure           16     36     338
    """

    # Confusion matrix: rows = true label, cols = predicted label
    CM = np.array([
        [280,  87,  22],
        [ 83, 283,  24],
        [ 16,  36, 338],
    ])
    CLASS_NAMES = ["Normal", "Pre-Sz", "Seizure"]

    def construct(self):
        self.camera.background_color = DARK_BG

        title = Text("Confusion Matrix  -  RF60+SVM40_C2", font_size=34,
                     color=WHITE, weight=BOLD)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title))

        n = 3
        cell = 1.65
        total = self.CM.sum()
        max_v = self.CM.max()

        # ── Axis labels ───────────────────────────────────────────────────
        y_title = Text("True Label", font_size=22, color=GRAY_A)
        y_title.rotate(PI / 2)
        y_title.move_to(LEFT * 4.3)

        x_title = Text("Predicted Label", font_size=22, color=GRAY_A)
        x_title.move_to(DOWN * 2.9)

        self.play(Write(y_title), Write(x_title))

        # Row and column headers
        for i, name in enumerate(self.CLASS_NAMES):
            col_h = Text(name, font_size=20, color=GRAY_A)
            col_h.move_to(RIGHT * (i - 1) * cell + UP * 2.4)
            row_h = Text(name, font_size=20, color=GRAY_A)
            row_h.move_to(LEFT * 2.9 + UP * (1 - i) * cell)
            self.play(Write(col_h), Write(row_h), run_time=0.22)

        # ── Cells ─────────────────────────────────────────────────────────
        class_totals = self.CM.sum(axis=1)

        for r in range(n):
            for c in range(n):
                val = self.CM[r, c]
                intensity = val / max_v

                if r == c:
                    fill = interpolate_color(
                        ManimColor(MID_BG), ManimColor(NEON_GREEN), intensity * 0.9 + 0.1
                    )
                else:
                    fill = interpolate_color(
                        ManimColor(MID_BG), ManimColor(NEON_RED), intensity * 0.85
                    )

                sq = Square(
                    side_length=cell,
                    fill_color=fill, fill_opacity=0.90,
                    stroke_color=GRAY_C, stroke_width=1.2,
                )
                sq.move_to(RIGHT * (c - 1) * cell + UP * (1 - r) * cell)

                pct = f"{100 * val / class_totals[r]:.1f}%"
                cnt = Text(str(val), font_size=24, color=WHITE, weight=BOLD)
                pct_t = Text(pct, font_size=15, color=GRAY_A)
                txt_grp = VGroup(cnt, pct_t).arrange(DOWN, buff=0.06)
                txt_grp.move_to(sq.get_center())

                self.play(FadeIn(sq), Write(txt_grp), run_time=0.22)

        # ── Overall accuracy footer ───────────────────────────────────────
        acc = int(np.trace(self.CM))
        acc_pct = 100 * acc / total
        acc_txt = Text(
            f"Overall Accuracy:  {acc}/{total}  =  {acc_pct:.1f}%",
            font_size=28, color=NEON_GREEN, weight=BOLD,
        )
        acc_txt.to_edge(DOWN, buff=0.30)
        self.play(Write(acc_txt))
        self.wait(1.8)


# ════════════════════════════════════════════════════════════════════════════
#  SCENE 6 – Random Forest deep-dive
# ════════════════════════════════════════════════════════════════════════════
class RFScene(Scene):
    """
    Full Random Forest lifecycle in NeuroWatch:
      1. 247 feature dots (3 classes) — remain visible in sidebar throughout
      2. Scout RF (300 trees) importance bar chart → top 75 selected / 172 greyed
      3. 3 representative trees (of 500), max_depth=5, sqrt(75)≈8 feats/split
         — sample seizure dot flows through centre tree to leaf
      4. 500-tree vote tally → RF output [Normal 0.18 | Pre-Sz 0.25 | Seizure 0.57]
    """

    def construct(self):
        self.camera.background_color = DARK_BG
        rng = np.random.default_rng(42)

        title = Text("NeuroWatch · Random Forest", font_size=36, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))
        self.wait(0.3)

        # ── Stage 1: 247 feature dots ──────────────────────────────────────
        sub = Text("247 EEG features per 4-second window", font_size=22, color=WHITE)
        sub.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(sub))

        def dot_cloud(n, color, x_lo, x_hi, y_lo, y_hi):
            g = VGroup()
            for _ in range(n):
                g.add(Dot(
                    point=[float(rng.uniform(x_lo, x_hi)),
                           float(rng.uniform(y_lo, y_hi)), 0],
                    radius=0.07, color=color,
                ))
            return g

        nd = dot_cloud(110, NEON_GREEN,  -5.5, -0.5, -2.0,  2.0)
        pd = dot_cloud( 80, NEON_YELLOW, -4.0,  0.8, -0.5,  2.8)
        sd = dot_cloud( 57, NEON_RED,    -5.0, -0.2, -3.0, -0.5)
        all_dots = VGroup(nd, pd, sd)

        legend = self._dot_legend(pos=[-5.9, -3.3, 0])

        self.play(
            LaggedStart(*[FadeIn(d, scale=0.3) for d in all_dots], lag_ratio=0.003),
            FadeIn(legend),
            run_time=2.0,
        )
        self.wait(0.8)
        _save_snapshot(self, "1_data_dots")

        # Shrink to left sidebar so they stay visible for subsequent stages
        self.play(
            all_dots.animate.scale(0.38).to_corner(UL, buff=0.55),
            legend.animate.scale(0.65).to_corner(UL, buff=0.08),
            FadeOut(sub),
            run_time=1.2,
        )
        self.wait(0.2)

        # ── Stage 2: Feature importance bar chart ──────────────────────────
        sub2 = Text(
            "Scout RF (300 trees) ranks all 247 features by Gini importance",
            font_size=20, color=WHITE,
        )
        sub2.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(sub2))

        n_bars = 30
        imp = np.exp(-np.linspace(0, 4.5, n_bars))
        imp += rng.uniform(0, 0.06, n_bars)
        imp /= imp.max()

        bw, bg, x0, y0, mh = 0.215, 0.04, 0.05, -2.9, 2.8

        target_bars = VGroup()
        for i, v in enumerate(imp):
            h = max(float(v) * mh, 0.02)
            r = Rectangle(width=bw, height=h,
                          fill_color=BLUE_ACC, fill_opacity=0.9, stroke_width=0)
            r.move_to([x0 + i * (bw + bg) + bw / 2, y0 + h / 2, 0])
            target_bars.add(r)

        start_bars = VGroup()
        for i in range(n_bars):
            r = Rectangle(width=bw, height=0.02,
                          fill_color=BLUE_ACC, fill_opacity=0.9, stroke_width=0)
            r.move_to([x0 + i * (bw + bg) + bw / 2, y0 + 0.01, 0])
            start_bars.add(r)

        ax_line = Line([x0, y0, 0], [x0 + n_bars * (bw + bg), y0, 0],
                       color=WHITE, stroke_width=1.5)
        ax_lbl = Text("Features (ranked by importance)", font_size=14, color=WHITE)
        ax_lbl.next_to(ax_line, DOWN, buff=0.14)

        self.play(Create(ax_line), FadeIn(ax_lbl))
        self.add(start_bars)
        self.play(
            LaggedStart(
                *[Transform(s, t) for s, t in zip(start_bars, target_bars)],
                lag_ratio=0.05,
            ),
            run_time=2.2,
        )

        # Top 9 bars ≈ top 75 of 247; rest greyed
        cutoff = 9
        self.play(
            *[start_bars[i].animate.set_fill(PURPLE_ACC) for i in range(cutoff)],
            *[start_bars[i].animate.set_fill(GRAY_C)     for i in range(cutoff, n_bars)],
            run_time=1.0,
        )

        sel_lbl = Text("Top 75 features kept   |   172 discarded", font_size=19, color=WHITE)
        sel_lbl.move_to([x0 + n_bars * (bw + bg) / 2, y0 - 0.62, 0])
        self.play(FadeIn(sel_lbl))

        # Pulse a sample of kept sidebar dots
        flash_dots = ([nd[i] for i in range(0, len(nd), 8)] +
                      [pd[i] for i in range(0, len(pd), 10)] +
                      [sd[i] for i in range(0, len(sd), 8)])
        self.play(
            *[Flash(d, color=PURPLE_ACC, flash_radius=0.06) for d in flash_dots],
            run_time=0.8,
        )
        self.wait(0.8)
        _save_snapshot(self, "2_feature_selection")

        self.play(
            FadeOut(start_bars), FadeOut(ax_line), FadeOut(ax_lbl),
            FadeOut(sel_lbl), FadeOut(sub2),
            run_time=0.7,
        )

        # ── Stage 3: Tree building ──────────────────────────────────────────
        sub3 = Text(
            "3 of 500 Trees  ·  max_depth=5  ·  sqrt(75) ≈ 8 features/split",
            font_size=20, color=WHITE,
        )
        sub3.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(sub3))

        t1 = self._build_tree(cx=-4.1, seed=0)
        t2 = self._build_tree(cx= 0.2, seed=1)
        t3 = self._build_tree(cx= 4.5, seed=2)

        self.play(
            LaggedStart(FadeIn(t1), FadeIn(t2), FadeIn(t3), lag_ratio=0.35),
            run_time=1.8,
        )

        # Seizure dot flows through centre tree (cx=0.2):
        #   root [0.2, 2.35] -> right child [1.15, 0.90] -> seizure leaf [1.75, -0.62]
        fdot = Dot(radius=0.10, color=NEON_RED).move_to([0.2, 2.35, 0])
        self.play(FadeIn(fdot))
        self.play(fdot.animate.move_to([1.15, 0.90, 0]), run_time=0.55)
        self.play(fdot.animate.move_to([1.75, -0.62, 0]), run_time=0.55)
        self.play(Flash(fdot, color=NEON_RED, flash_radius=0.24))
        self.play(FadeOut(fdot))
        self.wait(0.7)
        _save_snapshot(self, "3_tree_building")

        self.play(FadeOut(t1), FadeOut(t2), FadeOut(t3), FadeOut(sub3), run_time=0.7)

        # ── Stage 4: Vote tally ─────────────────────────────────────────────
        sub4 = Text("500 Trees Vote  —  Majority Class Wins", font_size=22, color=WHITE)
        sub4.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(sub4))

        vote_data = [
            ("Normal",      90,  NEON_GREEN),
            ("Pre-Seizure", 125, NEON_YELLOW),
            ("Seizure",     285, NEON_RED),
        ]
        max_v = 285
        bw2, bg2, x2, y2, mh2 = 1.25, 0.55, -2.25, -2.85, 3.2

        flat_v = VGroup()
        grow_v = VGroup()
        lbl_g  = VGroup()
        cnt_g  = VGroup()

        for i, (name, v, col) in enumerate(vote_data):
            x = x2 + i * (bw2 + bg2)
            h = (v / max_v) * mh2
            fr = Rectangle(width=bw2, height=0.02,
                           fill_color=col, fill_opacity=0.85, stroke_width=0)
            fr.move_to([x + bw2 / 2, y2 + 0.01, 0])
            flat_v.add(fr)

            gr = Rectangle(width=bw2, height=h,
                           fill_color=col, fill_opacity=0.85, stroke_width=0)
            gr.move_to([x + bw2 / 2, y2 + h / 2, 0])
            grow_v.add(gr)

            lbl_g.add(Text(name, font_size=17, color=col)
                      .move_to([x + bw2 / 2, y2 - 0.33, 0]))
            cnt_g.add(Text(str(v), font_size=17, color=col)
                      .move_to([x + bw2 / 2, y2 + h + 0.24, 0]))

        self.add(flat_v)
        self.play(FadeIn(lbl_g))
        self.play(
            LaggedStart(*[Transform(f, g) for f, g in zip(flat_v, grow_v)], lag_ratio=0.25),
            run_time=1.5,
        )
        self.play(FadeIn(cnt_g))

        # Highlight seizure winner
        self.play(flat_v[2].animate.set_stroke(WHITE, width=3.5), run_time=0.4)
        self.play(Flash(flat_v[2], color=NEON_RED, flash_radius=0.55), run_time=0.6)

        rf_prob = Text(
            "RF Output:  Normal 0.18  |  Pre-Seizure 0.25  |  Seizure 0.57",
            font_size=20, color=WHITE,
        )
        rf_prob.move_to([0.5, 0.55, 0])
        self.play(FadeIn(rf_prob))
        self.wait(1.5)
        _save_snapshot(self, "4_vote_tally")

        self.play(
            FadeOut(flat_v), FadeOut(lbl_g), FadeOut(cnt_g),
            FadeOut(rf_prob), FadeOut(sub4),
            FadeOut(all_dots), FadeOut(legend),
            run_time=0.8,
        )

    # ── helpers ──────────────────────────────────────────────────────────────
    def _dot_legend(self, pos):
        g = VGroup()
        for i, (col, lbl) in enumerate([
            (NEON_GREEN,  "Normal"),
            (NEON_YELLOW, "Pre-Seizure"),
            (NEON_RED,    "Seizure"),
        ]):
            row = VGroup(
                Dot(radius=0.09, color=col),
                Text(lbl, font_size=15, color=WHITE),
            ).arrange(RIGHT, buff=0.12)
            row.move_to(pos).shift(DOWN * i * 0.35)
            g.add(row)
        return g

    def _build_tree(self, cx, seed):
        """Return a VGroup: 3-level decision tree centred at x=cx."""
        feats = ["ch03-beta", "ch05-alpha", "ch01-delta", "ch04-theta",
                 "ch02-gamma", "ch06-activity", "ch03-entropy", "ch05-beta"]
        r = np.random.default_rng(seed * 37 + 11)

        def node(lbl, x, y, w=1.78):
            box = RoundedRectangle(
                corner_radius=0.08, width=w, height=0.40,
                fill_color=NAVY, fill_opacity=0.95,
                stroke_color=BLUE_ACC, stroke_width=1.2,
            ).move_to([x, y, 0])
            t = Text(lbl, font_size=10, color=WHITE)
            if t.width > w - 0.10:
                t.scale((w - 0.10) / t.width)
            t.move_to([x, y, 0])
            return VGroup(box, t)

        def leaf(x, y, col, lbl):
            c = Circle(
                radius=0.27, fill_color=col, fill_opacity=0.88,
                stroke_color=WHITE, stroke_width=1.0,
            ).move_to([x, y, 0])
            t = Text(lbl, font_size=10, color=WHITE).move_to([x, y, 0])
            return VGroup(c, t)

        def arr(x1, y1, x2, y2):
            return Arrow(
                [x1, y1, 0], [x2, y2, 0], buff=0,
                stroke_width=1.1, color=WHITE,
                max_tip_length_to_length_ratio=0.15,
            )

        f0 = str(r.choice(feats)); t0 = float(r.uniform(0.30, 0.80))
        f1 = str(r.choice(feats)); t1 = float(r.uniform(0.20, 0.90))
        f2 = str(r.choice(feats)); t2 = float(r.uniform(0.20, 0.90))

        root    = node(f"{f0} > {t0:.2f}?", cx,       2.35)
        child_l = node(f"{f1} > {t1:.2f}?", cx - 0.95, 0.90)
        child_r = node(f"{f2} > {t2:.2f}?", cx + 0.95, 0.90)

        yes_l = Text("YES", font_size=11, color=NEON_GREEN).move_to([cx - 0.57, 1.73, 0])
        no_r  = Text("NO",  font_size=11, color=NEON_RED  ).move_to([cx + 0.57, 1.73, 0])

        leaves = [
            leaf(cx - 1.55, -0.62, NEON_GREEN,  "Normal"),
            leaf(cx - 0.35, -0.62, NEON_YELLOW, "Pre-Sz"),
            leaf(cx + 0.35, -0.62, NEON_YELLOW, "Pre-Sz"),
            leaf(cx + 1.55, -0.62, NEON_RED,    "Seizure"),
        ]

        return VGroup(
            arr(cx, 2.15, cx - 0.95, 1.10),
            arr(cx, 2.15, cx + 0.95, 1.10),
            yes_l, no_r,
            root, child_l, child_r,
            arr(cx - 0.95, 0.70, cx - 1.55, -0.35),
            arr(cx - 0.95, 0.70, cx - 0.35, -0.35),
            arr(cx + 0.95, 0.70, cx + 0.35, -0.35),
            arr(cx + 0.95, 0.70, cx + 1.55, -0.35),
            *leaves,
        )


# ════════════════════════════════════════════════════════════════════════════
#  SCENE 7 – SVM deep-dive
# ════════════════════════════════════════════════════════════════════════════
class SVMScene(Scene):
    """
    Full SVM pipeline in NeuroWatch:
      1. 2D scatter: ch03-beta vs ch03-activity — all 3 classes visible
      2. StandardScaler: axes recolour blue, dots contract toward centre
      3. RBF kernel: dots "lift" apart to show kernel-space separability
      4. Support vectors: white rings on boundary-adjacent dots + margin dashes
      5. Decision boundary: animated RBF curve, 1.8x Pre-Seizure margin shown
      6. Platt scaling: sigmoid animation, SVM output [0.13, 0.30, 0.57]
    """

    def construct(self):
        self.camera.background_color = DARK_BG
        rng = np.random.default_rng(99)

        title = Text("NeuroWatch · Support Vector Machine", font_size=34, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))
        self.wait(0.3)

        # Build axes first so dots can be placed using coords_to_point
        ax = Axes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            x_length=6.2, y_length=5.0,
            axis_config={"color": WHITE, "stroke_width": 1.5, "include_tip": True},
        )
        ax.move_to([0.8, -0.55, 0])

        # Store raw feature coords so we can fit a real SVM for the boundary
        raw_X, raw_y = [], []

        def cluster(n, mu_x, mu_y, sig, color, cls_label):
            g = VGroup()
            for _ in range(n):
                x, y = float(rng.normal(mu_x, sig)), float(rng.normal(mu_y, sig))
                g.add(Dot(ax.coords_to_point(x, y), radius=0.07, color=color))
                raw_X.append([x, y])
                raw_y.append(cls_label)
            return g

        nd = cluster(42, -1.5, -0.5, 0.50, NEON_GREEN,  0)
        pd = cluster(30,  0.3,  1.4, 0.40, NEON_YELLOW, 1)
        sd = cluster(22,  1.7, -1.0, 0.38, NEON_RED,    2)
        all_dots = VGroup(nd, pd, sd)

        # Fit real SVM on the generated data (matching project hyperparameters)
        from sklearn.svm import SVC as _SVC
        _svm = _SVC(kernel='rbf', C=2, gamma='scale',
                    class_weight={0: 1.0, 1: 1.8, 2: 1.0}, random_state=42)
        _svm.fit(np.array(raw_X), np.array(raw_y))

        # Pre-compute real decision boundary contours via matplotlib (Agg, no display)
        import matplotlib as _mpl
        _mpl.use('Agg')
        import matplotlib.pyplot as _plt
        _lo, _hi = -3.2, 3.2
        _xx, _yy = np.meshgrid(np.linspace(_lo, _hi, 260), np.linspace(_lo, _hi, 260))
        _Z = _svm.predict(np.c_[_xx.ravel(), _yy.ravel()]).reshape(_xx.shape)
        _fig, _mpl_ax = _plt.subplots()
        _cs = _mpl_ax.contour(_xx, _yy, _Z, levels=[0.5, 1.5])
        _real_contours = []
        for _path in _cs.get_paths():
            if len(_path.vertices) > 8:
                _real_contours.append(_path.vertices.copy())
        _plt.close(_fig)

        xl = Text("ch03-beta (scaled)", font_size=15, color=WHITE)
        xl.next_to(ax.x_axis.get_right(), DOWN, buff=0.14)
        yl = Text("ch03-activity (scaled)", font_size=15, color=WHITE).rotate(PI / 2)
        yl.next_to(ax.y_axis.get_top(), LEFT, buff=0.12)

        legend = self._legend()
        sub1 = Text("2D Feature Space: ch03-beta vs ch03-activity", font_size=21, color=WHITE)
        sub1.next_to(title, DOWN, buff=0.15)

        # ── Stage 1: raw scatter ───────────────────────────────────────────
        self.play(FadeIn(sub1))
        self.play(Create(ax), Write(xl), Write(yl), run_time=0.9)
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in all_dots], lag_ratio=0.008),
            FadeIn(legend),
            run_time=1.4,
        )
        self.wait(0.8)
        _save_snapshot(self, "1_raw_scatter")

        # ── Stage 2: StandardScaler ────────────────────────────────────────
        sub2 = Text(
            "StandardScaler: centre μ=0, std σ=1  (fitted on training data only)",
            font_size=19, color=WHITE,
        )
        sub2.next_to(title, DOWN, buff=0.15)
        self.play(ReplacementTransform(sub1, sub2))

        # Axes turn blue to signal the transformation
        new_ax = Axes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1],
            x_length=6.2, y_length=5.0,
            axis_config={"color": BLUE_ACC, "stroke_width": 1.8, "include_tip": True},
        )
        new_ax.move_to([0.8, -0.55, 0])
        self.play(Transform(ax, new_ax), run_time=0.7)

        # Dots contract slightly toward the origin (centring effect)
        origin = np.array(new_ax.coords_to_point(0, 0))
        contract = []
        for grp in [nd, pd, sd]:
            for d in grp:
                old = d.get_center()
                contract.append(d.animate.move_to(old * 0.87 + origin * 0.13))
        self.play(*contract, run_time=0.9)

        mu_line = DashedLine(
            new_ax.coords_to_point(-2.8, 0), new_ax.coords_to_point(2.8, 0),
            dash_length=0.12, stroke_width=1.5, color=BLUE_ACC,
        )
        mu_lbl = Text("μ = 0", font_size=16, color=BLUE_ACC)
        mu_lbl.next_to(mu_line, RIGHT, buff=0.08)
        self.play(Create(mu_line), FadeIn(mu_lbl), run_time=0.6)
        self.wait(0.7)
        _save_snapshot(self, "2_scaler")
        self.play(FadeOut(mu_line), FadeOut(mu_lbl))

        # ── Stage 3: RBF kernel "lift" ─────────────────────────────────────
        sub3 = Text(
            "RBF Kernel (gamma='scale'): maps data to higher-dimensional space",
            font_size=19, color=WHITE,
        )
        sub3.next_to(title, DOWN, buff=0.15)
        self.play(ReplacementTransform(sub2, sub3))

        rng2 = np.random.default_rng(55)
        lift = []
        for d in nd:
            c = d.get_center()
            lift.append(d.animate.move_to(
                c + [float(rng2.uniform(-0.18, 0.18)), float(rng2.uniform(-0.38, -0.08)), 0]
            ))
        for d in pd:
            c = d.get_center()
            lift.append(d.animate.move_to(
                c + [float(rng2.uniform(-0.12, 0.12)), float(rng2.uniform(0.18, 0.45)), 0]
            ))
        for d in sd:
            c = d.get_center()
            lift.append(d.animate.move_to(
                c + [float(rng2.uniform(0.10, 0.28)), float(rng2.uniform(-0.38, -0.10)), 0]
            ))
        self.play(LaggedStart(*lift, lag_ratio=0.005), run_time=1.2)

        hp = new_ax.plot(lambda x: 0.28, x_range=[-2.8, 2.8],
                         color=PURPLE_ACC, stroke_width=2.5)
        hp_lbl = Text("Separating hyperplane (kernel space)", font_size=14, color=PURPLE_ACC)
        hp_lbl.next_to(hp, UP, buff=0.09)
        self.play(Create(hp), FadeIn(hp_lbl), run_time=0.8)
        self.wait(0.7)
        _save_snapshot(self, "3_rbf_kernel")
        self.play(FadeOut(hp), FadeOut(hp_lbl))

        # ── Stage 4: Support vectors ───────────────────────────────────────
        sub4 = Text("Support Vectors: points closest to the decision margin",
                    font_size=21, color=WHITE)
        sub4.next_to(title, DOWN, buff=0.15)
        self.play(ReplacementTransform(sub3, sub4))

        sv_rings = VGroup()
        for grp in [nd, pd, sd]:
            for d in list(grp)[:2]:
                ring = Circle(radius=0.13, color=WHITE, stroke_width=2.5)
                ring.move_to(d.get_center())
                sv_rings.add(ring)
        self.play(LaggedStart(*[Create(r) for r in sv_rings], lag_ratio=0.1), run_time=0.9)

        m_up = DashedLine(
            new_ax.coords_to_point(-2.8,  0.65), new_ax.coords_to_point(2.8,  0.65),
            dash_length=0.14, stroke_width=1.8, color=WHITE,
        )
        m_dn = DashedLine(
            new_ax.coords_to_point(-2.8, -0.10), new_ax.coords_to_point(2.8, -0.10),
            dash_length=0.14, stroke_width=1.8, color=WHITE,
        )
        m_lbl = Text("Margin", font_size=16, color=WHITE).next_to(m_up, RIGHT, buff=0.09)
        self.play(Create(m_up), Create(m_dn), FadeIn(m_lbl), run_time=0.8)
        self.wait(0.8)
        _save_snapshot(self, "4_support_vectors")
        self.play(FadeOut(sv_rings), FadeOut(m_up), FadeOut(m_dn), FadeOut(m_lbl))

        # ── Stage 5: Decision boundary ─────────────────────────────────────
        sub5 = Text(
            "RBF Decision Boundary  ·  C=2  ·  class_weight Pre-Sz = 1.8×",
            font_size=20, color=WHITE,
        )
        sub5.next_to(title, DOWN, buff=0.15)
        self.play(ReplacementTransform(sub4, sub5))

        # Draw real SVM decision boundary (computed from fitted SVC above)
        bdy_grp = VGroup()
        _bdy_cols = [WHITE, NEON_YELLOW]
        for _i, _verts in enumerate(_real_contours):
            _vm = VMobject(stroke_color=_bdy_cols[_i % 2], stroke_width=2.5)
            _pts = [new_ax.coords_to_point(float(_v[0]), float(_v[1])) for _v in _verts]
            if len(_pts) >= 2:
                _vm.set_points_as_corners(_pts)
            bdy_grp.add(_vm)

        w_lbl = Text(
            "Real SVM boundary  ·  class_weight Pre-Sz = 1.8×  (C=2, gamma='scale')",
            font_size=14, color=NEON_YELLOW,
        )
        w_lbl.move_to(new_ax.coords_to_point(0, -2.6))

        self.play(Create(bdy_grp), run_time=1.2)
        self.play(FadeIn(w_lbl), run_time=0.6)
        self.wait(1.0)
        _save_snapshot(self, "5_decision_boundary")
        self.play(FadeOut(bdy_grp), FadeOut(w_lbl))

        # ── Stage 6: Platt scaling ─────────────────────────────────────────
        sub6 = Text(
            "Platt Scaling (5-fold CV): raw SVM score → calibrated probability",
            font_size=19, color=WHITE,
        )
        sub6.next_to(title, DOWN, buff=0.15)
        self.play(
            ReplacementTransform(sub5, sub6),
            FadeOut(all_dots), FadeOut(ax), FadeOut(legend),
            FadeOut(xl), FadeOut(yl),
            run_time=0.8,
        )

        sig_ax = Axes(
            x_range=[-5, 5, 1], y_range=[0, 1, 0.25],
            x_length=7.2, y_length=3.6,
            axis_config={"color": WHITE, "stroke_width": 1.5, "include_tip": True},
        )
        sig_ax.move_to([0.4, -1.0, 0])
        sx_lbl = Text("Raw SVM decision score", font_size=16, color=WHITE)
        sx_lbl.next_to(sig_ax.x_axis.get_right(), DOWN, buff=0.12)
        sy_lbl = Text("Probability", font_size=16, color=WHITE).rotate(PI / 2)
        sy_lbl.next_to(sig_ax.y_axis.get_top(), LEFT, buff=0.12)

        sigmoid = sig_ax.plot(
            lambda x: 1 / (1 + np.exp(-x)),
            x_range=[-5, 5], color=PURPLE_ACC, stroke_width=3,
        )
        self.play(Create(sig_ax), Write(sx_lbl), Write(sy_lbl), run_time=0.8)
        self.play(Create(sigmoid), run_time=1.0)

        # Animate a score dot rising up the sigmoid to its probability
        raw_score = 2.1
        prob_val  = 1 / (1 + np.exp(-raw_score))
        s_pt = sig_ax.coords_to_point(raw_score, 0)
        p_pt = sig_ax.coords_to_point(raw_score, prob_val)

        s_dot = Dot(s_pt, color=NEON_RED, radius=0.12)
        p_dot = Dot(p_pt, color=NEON_RED, radius=0.12)
        s_lbl = Text("score = 2.1", font_size=16, color=NEON_RED)
        s_lbl.next_to(s_dot, DOWN, buff=0.10)
        p_lbl = Text("P(Seizure) ≈ 0.57", font_size=16, color=NEON_RED)
        p_lbl.next_to(p_dot, RIGHT, buff=0.10)

        self.play(FadeIn(s_dot), FadeIn(s_lbl))
        self.play(Transform(s_dot, p_dot), FadeOut(s_lbl), FadeIn(p_lbl), run_time=0.9)

        out_txt = Text(
            "SVM Output:  Normal 0.13  |  Pre-Seizure 0.30  |  Seizure 0.57",
            font_size=20, color=WHITE,
        )
        out_txt.move_to([0.4, 2.8, 0])
        self.play(FadeIn(out_txt))
        self.wait(1.8)
        _save_snapshot(self, "6_platt_scaling")

        self.play(
            FadeOut(sig_ax), FadeOut(sx_lbl), FadeOut(sy_lbl), FadeOut(sigmoid),
            FadeOut(s_dot), FadeOut(p_lbl), FadeOut(out_txt), FadeOut(sub6),
            run_time=0.7,
        )

    def _legend(self):
        g = VGroup()
        for i, (col, lbl) in enumerate([
            (NEON_GREEN,  "Normal"),
            (NEON_YELLOW, "Pre-Seizure"),
            (NEON_RED,    "Seizure"),
        ]):
            row = VGroup(
                Dot(radius=0.09, color=col),
                Text(lbl, font_size=15, color=WHITE),
            ).arrange(RIGHT, buff=0.12)
            row.to_corner(DR, buff=0.45).shift(UP * i * 0.40)
            g.add(row)
        return g


# ════════════════════════════════════════════════════════════════════════════
#  SCENE 8 – Ensemble decision detail
# ════════════════════════════════════════════════════════════════════════════
class MLEnsembleDetailScene(Scene):
    """
    Detailed ensemble decision pipeline:
      1. SVM [0.13, 0.30, 0.57] and RF [0.18, 0.25, 0.57] side-by-side bars
      2. Weighted blend: 0.40×SVM + 0.60×RF → [0.16, 0.27, 0.57]
      3. Normal threshold correction:
            argmax=2 (Seizure), Normal=0.16 < 0.39 → threshold NOT triggered
      4. Final prediction: SEIZURE at 57% confidence — red flash

    Run: manim -pql neurowatch_animations.py MLEnsembleDetailScene
    """

    SVM_P  = [0.13, 0.30, 0.57]
    RF_P   = [0.18, 0.25, 0.57]
    COLORS = [NEON_GREEN, NEON_YELLOW, NEON_RED]
    NAMES  = ["Normal", "Pre-Sz", "Seizure"]

    def construct(self):
        self.camera.background_color = DARK_BG

        title = Text("NeuroWatch · Ensemble Decision", font_size=36, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))
        self.wait(0.3)

        # ── Stage 1: side-by-side model outputs ────────────────────────────
        sub1 = Text("Model probability outputs before blending", font_size=21, color=WHITE)
        sub1.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(sub1))

        svm_g = self._prob_group("SVM  × 0.40", self.SVM_P, x=-3.5)
        rf_g  = self._prob_group("RF   × 0.60", self.RF_P,  x= 3.5)

        self.play(LaggedStart(FadeIn(svm_g), FadeIn(rf_g), lag_ratio=0.3), run_time=1.4)
        self.wait(1.0)

        # ── Stage 2: weighted blend ─────────────────────────────────────────
        sub2 = Text(
            "Weighted Blend:  combined = 0.40 × p_SVM + 0.60 × p_RF",
            font_size=20, color=WHITE,
        )
        sub2.next_to(title, DOWN, buff=0.15)
        self.play(ReplacementTransform(sub1, sub2))

        self.play(
            svm_g.animate.scale(0.70).move_to([-4.5, -0.7, 0]),
            rf_g.animate.scale(0.70).move_to([ 4.5, -0.7, 0]),
            run_time=0.8,
        )

        al = Arrow([-3.1, -0.5, 0], [-0.6, 0.1, 0], buff=0,
                   color=WHITE, stroke_width=2.0)
        ar = Arrow([ 3.1, -0.5, 0], [ 0.6, 0.1, 0], buff=0,
                   color=WHITE, stroke_width=2.0)
        wl = Text("0.40", font_size=16, color=TEAL).next_to(al, DOWN, buff=0.05)
        wr = Text("0.60", font_size=16, color=ORANGE_ACC).next_to(ar, DOWN, buff=0.05)

        self.play(GrowArrow(al), GrowArrow(ar), FadeIn(wl), FadeIn(wr), run_time=0.7)

        combined = [0.40 * s + 0.60 * r for s, r in zip(self.SVM_P, self.RF_P)]
        comb_g = self._prob_group("Combined", combined, x=0.0)
        self.play(FadeIn(comb_g), run_time=1.0)
        self.wait(1.0)

        # ── Stage 3: threshold correction logic ────────────────────────────
        sub3 = Text(
            "Normal Threshold Correction  (θ = 0.39)",
            font_size=21, color=WHITE,
        )
        sub3.next_to(title, DOWN, buff=0.15)
        self.play(
            ReplacementTransform(sub2, sub3),
            FadeOut(al), FadeOut(ar), FadeOut(wl), FadeOut(wr),
            FadeOut(svm_g), FadeOut(rf_g),
            run_time=0.7,
        )

        logic = [
            ("argmax([0.16, 0.27, 0.57]) = 2  →  Seizure predicted",       WHITE),
            ("Normal probability = 0.16",                                         WHITE),
            ("0.16  <  threshold 0.39  →  correction NOT triggered",         NEON_GREEN),
            ("Final prediction:  SEIZURE  (57% confidence)",                      NEON_RED),
        ]

        lg = VGroup()
        for i, (txt, col) in enumerate(logic):
            t = Text(txt, font_size=19, color=col)
            t.move_to([0.3, 2.10 - i * 0.72, 0])
            lg.add(t)

        self.play(
            LaggedStart(*[Write(t) for t in lg], lag_ratio=0.55),
            run_time=2.5,
        )
        self.wait(0.8)

        # ── Stage 4: final prediction flash ────────────────────────────────
        self.play(FadeOut(comb_g), FadeOut(lg), FadeOut(sub3), run_time=0.6)

        pred_box = RoundedRectangle(
            corner_radius=0.22, width=7.4, height=2.6,
            fill_color=NEON_RED, fill_opacity=0.12,
            stroke_color=NEON_RED, stroke_width=3.5,
        ).move_to([0.3, -0.3, 0])

        pred_txt = Text("SEIZURE DETECTED", font_size=52, color=NEON_RED, weight=BOLD)
        pred_txt.move_to([0.3, -0.05, 0])
        conf_txt = Text(
            "Confidence: 57%   |   Ensemble: RF × 0.60 + SVM × 0.40",
            font_size=22, color=WHITE,
        )
        conf_txt.move_to([0.3, -0.85, 0])

        self.play(FadeIn(pred_box), run_time=0.3)
        self.play(Write(pred_txt), Write(conf_txt), run_time=0.9)

        for _ in range(3):
            self.play(pred_box.animate.set_fill(NEON_RED, opacity=0.44), run_time=0.22)
            self.play(pred_box.animate.set_fill(NEON_RED, opacity=0.10), run_time=0.22)

        tagline = Text(
            "NeuroWatch · Real-Time EEG Seizure Monitoring",
            font_size=22, color=WHITE,
        )
        tagline.move_to([0.3, -2.55, 0])
        self.play(FadeIn(tagline))
        self.wait(2.0)

    def _prob_group(self, header, probs, x, bw=0.82, mh=2.2, y_base=-1.6):
        g = VGroup()
        hdr = Text(header, font_size=20, color=WHITE).move_to([x, 2.65, 0])
        g.add(hdr)

        x_offsets = [-0.92, 0.0, 0.92]
        for prob, col, name, xoff in zip(probs, self.COLORS, self.NAMES, x_offsets):
            h    = prob * mh
            rect = Rectangle(width=bw, height=h,
                             fill_color=col, fill_opacity=0.85, stroke_width=0)
            rect.move_to([x + xoff, y_base + h / 2, 0])
            plbl = Text(f"{prob:.2f}", font_size=15, color=col)
            plbl.move_to([x + xoff, y_base + h + 0.23, 0])
            clbl = Text(name, font_size=13, color=WHITE)
            clbl.move_to([x + xoff, y_base - 0.30, 0])
            g.add(VGroup(rect, plbl, clbl))
        return g


# ════════════════════════════════════════════════════════════════════════════
#  SCENE 9 – RF Report Images  (white background, publication quality)
# ════════════════════════════════════════════════════════════════════════════
class RFReportImages(Scene):
    """
    4 static white-background images for the RF pipeline.
    Saves to REPORT/stage_images/report/:
      RF_1_data_dots.png  RF_2_feature_selection.png
      RF_3_trees.png      RF_4_vote_tally.png

    Run: python -m manim -pql neurowatch_animations.py RFReportImages
    """

    # Report-friendly colours (good contrast on white)
    CN = "#006633"   # Normal       dark green
    CP = "#996600"   # Pre-Seizure  dark amber
    CS = "#CC1111"   # Seizure      dark red
    CV = "#5500BB"   # Violet – selected features

    def construct(self):
        self.camera.background_color = WHITE
        rng = np.random.default_rng(42)
        self._s1_dots(rng)
        self._s2_importance(rng)
        self._s3_trees(rng)
        self._s4_votes()

    # ── shared helpers ────────────────────────────────────────────────────────
    def _wipe(self):
        self.remove(*self.mobjects)

    def _head(self, title_str, sub_str):
        t = Text(title_str, font_size=30, color=BLACK, weight=BOLD)
        t.to_edge(UP, buff=0.28)
        s = Text(sub_str, font_size=15, color="#555555")
        s.next_to(t, DOWN, buff=0.08)
        return t, s

    def _legend_row(self):
        items = [
            (self.CN, "Normal  (class 0)"),
            (self.CP, "Pre-Seizure  (class 1)"),
            (self.CS, "Seizure  (class 2)"),
        ]
        g = VGroup(*[
            VGroup(Dot(radius=0.10, color=c),
                   Text(lbl, font_size=16, color=BLACK)).arrange(RIGHT, buff=0.12)
            for c, lbl in items
        ])
        g.arrange(RIGHT, buff=0.55)
        return g

    def _code_footer(self, txt):
        return Text(txt, font_size=13, color="#444444").to_edge(DOWN, buff=0.28)

    # ── Stage 1: 247 dots ─────────────────────────────────────────────────────
    def _s1_dots(self, rng):
        t, s = self._head(
            "247 EEG Features per 4-Second Window",
            "19 channels × 13 features  ·  Each dot = one feature value  ·  Colour = patient state",
        )

        def cloud(n, col, x0, x1, y0, y1):
            g = VGroup()
            for _ in range(n):
                g.add(Dot([float(rng.uniform(x0, x1)),
                           float(rng.uniform(y0, y1)), 0],
                          radius=0.065, color=col))
            return g

        nd = cloud(110, self.CN,  -5.5, -0.5, -1.8,  1.8)
        pd = cloud( 80, self.CP,  -4.0,  0.8, -0.3,  2.5)
        sd = cloud( 57, self.CS,  -5.0, -0.2, -2.8, -0.5)
        dots = VGroup(nd, pd, sd).move_to(ORIGIN + DOWN * 0.25)

        legend = self._legend_row().to_edge(DOWN, buff=0.55)

        counts = VGroup(
            Text("n = 110", font_size=14, color=self.CN),
            Text("n = 80",  font_size=14, color=self.CP),
            Text("n = 57",  font_size=14, color=self.CS),
        ).arrange(RIGHT, buff=2.0).next_to(legend, UP, buff=0.14)

        code = self._code_footer(
            "Feature set:  19 channels  ×  13 per channel  =  247 dimensions  |"
            "  Unscaled, used directly by RF"
        )

        self.add(t, s, dots, legend, counts, code)
        self.wait(0.1)
        _save_report(self, "RF_1_data_dots")
        self._wipe()

    # ── Stage 2: Feature importance ───────────────────────────────────────────
    def _s2_importance(self, rng):
        t, s = self._head(
            "Scout RF (300 Trees): Gini Feature Importance Ranking",
            "Mean decrease in Gini impurity  ·  Top 75 of 247 selected  ·  172 discarded",
        )

        n = 30
        imp = np.exp(-np.linspace(0, 4.5, n)) + rng.uniform(0, 0.06, n)
        imp /= imp.max()
        bw, bg_, x0, y0, mh = 0.215, 0.04, -3.1, -2.8, 3.2
        cutoff = 9

        bars = VGroup()
        for i, v in enumerate(imp):
            h = max(float(v) * mh, 0.03)
            col = self.CV if i < cutoff else "#CCCCCC"
            r = Rectangle(width=bw, height=h,
                          fill_color=col, fill_opacity=1.0, stroke_width=0)
            r.move_to([x0 + i * (bw + bg_) + bw / 2, y0 + h / 2, 0])
            bars.add(r)

        ax_line = Line([x0, y0, 0], [x0 + n * (bw + bg_), y0, 0],
                       color=BLACK, stroke_width=1.5)
        ax_lbl = Text("All 247 features (sorted by importance)", font_size=14, color=BLACK)
        ax_lbl.next_to(ax_line, DOWN, buff=0.14)
        y_lbl = Text("Gini\nImportance", font_size=13, color=BLACK)
        y_lbl.move_to([x0 - 0.65, y0 + mh / 2, 0])

        cut_x = x0 + cutoff * (bw + bg_)
        vline = DashedLine([cut_x, y0 - 0.12, 0], [cut_x, y0 + mh + 0.25, 0],
                           dash_length=0.10, stroke_width=1.8, color=self.CV)
        kept_lbl = Text("Top 75 kept", font_size=14, color=self.CV, weight=BOLD)
        kept_lbl.move_to([x0 + (cutoff - 1) * (bw + bg_) / 2 + bw, y0 + mh + 0.48, 0])
        disc_lbl = Text("172 discarded", font_size=14, color="#888888")
        disc_lbl.move_to([cut_x + (n - cutoff) * (bw + bg_) / 2, y0 + 1.6, 0])

        code = self._code_footer(
            "RFImportanceSelector(k=75)  |  n_estimators=300, max_depth=10  |"
            "  selector.top_indices_ passed to both RF and SVM"
        )

        self.add(t, s, bars, ax_line, ax_lbl, y_lbl, vline, kept_lbl, disc_lbl, code)
        self.wait(0.1)
        _save_report(self, "RF_2_feature_selection")
        self._wipe()

    # ── Stage 3: Decision trees ───────────────────────────────────────────────
    def _s3_trees(self, rng):
        t, s = self._head(
            "Random Forest: 3 of 500 Decision Trees  (max_depth = 5)",
            "Bootstrap sample per tree  ·  Each node splits on sqrt(75) ≈ 8 randomly chosen features",
        )

        trees = VGroup(
            self._rpt_tree(-4.1, 0, rng),
            self._rpt_tree( 0.2, 1, rng),
            self._rpt_tree( 4.5, 2, rng),
        ).move_to(ORIGIN + DOWN * 0.1)

        code = self._code_footer(
            "RandomForestClassifier(n_estimators=500, max_depth=5, max_features='sqrt',"
            " class_weight='balanced', min_samples_leaf=10, random_state=42)"
        )

        self.add(t, s, trees, code)
        self.wait(0.1)
        _save_report(self, "RF_3_trees")
        self._wipe()

    def _rpt_tree(self, cx, seed, rng):
        feats = ["ch03-beta", "ch05-alpha", "ch01-delta", "ch04-theta",
                 "ch02-gamma", "ch06-activity", "ch03-entropy", "ch05-beta"]
        r = np.random.default_rng(seed * 37 + 11)

        def node(lbl, x, y, w=1.82):
            box = RoundedRectangle(corner_radius=0.08, width=w, height=0.42,
                                   fill_color="#E8EEFF", fill_opacity=1.0,
                                   stroke_color="#1155CC", stroke_width=1.5)
            box.move_to([x, y, 0])
            t_ = Text(lbl, font_size=10, color=BLACK)
            if t_.width > w - 0.10:
                t_.scale((w - 0.10) / t_.width)
            t_.move_to([x, y, 0])
            return VGroup(box, t_)

        def leaf(x, y, col, lbl, bg):
            c = Circle(radius=0.27, fill_color=bg, fill_opacity=1.0,
                       stroke_color=col, stroke_width=2.0).move_to([x, y, 0])
            t_ = Text(lbl, font_size=10, color=col, weight=BOLD).move_to([x, y, 0])
            return VGroup(c, t_)

        def arr_(x1, y1, x2, y2):
            return Arrow([x1, y1, 0], [x2, y2, 0], buff=0, stroke_width=1.0,
                         color="#444444", max_tip_length_to_length_ratio=0.12)

        f0, t0 = str(r.choice(feats)), float(r.uniform(0.30, 0.80))
        f1, t1 = str(r.choice(feats)), float(r.uniform(0.20, 0.90))
        f2, t2 = str(r.choice(feats)), float(r.uniform(0.20, 0.90))

        root    = node(f"{f0} > {t0:.2f}?", cx,        2.10)
        child_l = node(f"{f1} > {t1:.2f}?", cx - 0.92, 0.65)
        child_r = node(f"{f2} > {t2:.2f}?", cx + 0.92, 0.65)
        yes_l   = Text("YES", font_size=11, color="#006633", weight=BOLD).move_to([cx - 0.55, 1.48, 0])
        no_r    = Text("NO",  font_size=11, color="#CC1111", weight=BOLD).move_to([cx + 0.55, 1.48, 0])
        leaves  = [
            leaf(cx - 1.52, -0.65, self.CN, "Normal",  "#EDFFF5"),
            leaf(cx - 0.32, -0.65, self.CP, "Pre-Sz",  "#FFF9E8"),
            leaf(cx + 0.32, -0.65, self.CP, "Pre-Sz",  "#FFF9E8"),
            leaf(cx + 1.52, -0.65, self.CS, "Seizure", "#FFECEC"),
        ]
        return VGroup(
            arr_(cx, 1.90, cx - 0.92, 0.85), arr_(cx, 1.90, cx + 0.92, 0.85),
            yes_l, no_r, root, child_l, child_r,
            arr_(cx - 0.92, 0.45, cx - 1.52, -0.38),
            arr_(cx - 0.92, 0.45, cx - 0.32, -0.38),
            arr_(cx + 0.92, 0.45, cx + 0.32, -0.38),
            arr_(cx + 0.92, 0.45, cx + 1.52, -0.38),
            *leaves,
        )

    # ── Stage 4: Vote tally ───────────────────────────────────────────────────
    def _s4_votes(self):
        t, s = self._head(
            "500-Tree Majority Vote → RF Posterior Probabilities",
            "Vote counts are proportional to the RF class posterior  p(y | x)",
        )

        data = [
            ("Normal",      90,  self.CN, "0.18"),
            ("Pre-Seizure", 125, self.CP, "0.25"),
            ("Seizure",     285, self.CS, "0.57"),
        ]
        max_v, bw, gap, x0, y0, mh = 285, 1.4, 0.6, -2.8, -2.85, 3.5

        bars  = VGroup()
        lbls  = VGroup()
        cnts  = VGroup()
        probs = VGroup()

        for i, (name, v, col, prob) in enumerate(data):
            x = x0 + i * (bw + gap)
            h = (v / max_v) * mh
            r = Rectangle(width=bw, height=h,
                          fill_color=col, fill_opacity=0.80, stroke_width=0)
            r.move_to([x + bw / 2, y0 + h / 2, 0])
            bars.add(r)
            lbls.add(Text(name, font_size=16, color=col, weight=BOLD)
                     .move_to([x + bw / 2, y0 - 0.40, 0]))
            cnts.add(Text(f"{v} votes", font_size=14, color=col)
                     .move_to([x + bw / 2, y0 + h + 0.28, 0]))
            probs.add(Text(f"p = {prob}", font_size=20, color=col, weight=BOLD)
                      .move_to([x + bw / 2, y0 + h + 0.68, 0]))

        winner_box = SurroundingRectangle(bars[2], color=self.CS, buff=0.10, stroke_width=2.5)
        win_lbl    = Text("WINNER", font_size=15, color=self.CS, weight=BOLD)
        win_lbl.next_to(winner_box, RIGHT, buff=0.14)

        out_box = RoundedRectangle(corner_radius=0.12, width=8.8, height=0.55,
                                   fill_color="#F4F4F4", fill_opacity=1.0,
                                   stroke_color=BLACK, stroke_width=1.0)
        out_box.to_edge(DOWN, buff=0.58)
        out_txt = Text(
            "rf.predict_proba(x_rf)  →  [Normal: 0.18,  Pre-Seizure: 0.25,  Seizure: 0.57]",
            font_size=16, color=BLACK,
        ).move_to(out_box.get_center())

        code = self._code_footer(
            "x_rf = selector.transform(x_raw)  — unscaled, top-75 features fed directly to RF"
        )

        self.add(t, s, bars, lbls, cnts, probs, winner_box, win_lbl, out_box, out_txt, code)
        self.wait(0.1)
        _save_report(self, "RF_4_vote_tally")
        self._wipe()


# ════════════════════════════════════════════════════════════════════════════
#  SCENE 10 – SVM Report Images  (white background, REAL SVM boundary)
# ════════════════════════════════════════════════════════════════════════════
class SVMReportImages(Scene):
    """
    6 static white-background images for the SVM pipeline.
    The decision boundary and support vectors are computed from a REAL
    sklearn SVC fitted on the same data used for the scatter plots.

    Saves to REPORT/stage_images/report/:
      SVM_1_raw_scatter.png      SVM_2_scaler.png
      SVM_3_rbf_kernel.png       SVM_4_support_vectors.png
      SVM_5_decision_boundary.png  SVM_6_platt_scaling.png

    Run: python -m manim -pql neurowatch_animations.py SVMReportImages
    """

    CN = "#006633"
    CP = "#996600"
    CS = "#CC1111"
    CB = "#1155CC"
    CV = "#6611AA"
    # Soft fills for decision regions
    RN = "#D4F5E5"
    RP = "#FFF4CC"
    RS = "#FFD8D8"

    def construct(self):
        self.camera.background_color = WHITE

        # ── Generate same data as SVMScene (seed=99) ──────────────────────
        rng = np.random.default_rng(99)
        mu_list = [
            (42, -1.5, -0.5, 0.50, 0),
            (30,  0.3,  1.4, 0.40, 1),
            (22,  1.7, -1.0, 0.38, 2),
        ]
        raw   = {0: [], 1: [], 2: []}
        X_lst, y_lst = [], []
        for n, mx, my, sig, cls in mu_list:
            for _ in range(n):
                x, y = float(rng.normal(mx, sig)), float(rng.normal(my, sig))
                raw[cls].append((x, y))
                X_lst.append([x, y])
                y_lst.append(cls)
        X_data = np.array(X_lst)
        y_data = np.array(y_lst)

        # ── Fit REAL SVM (project hyperparameters) ─────────────────────────
        from sklearn.svm import SVC as _SVC2
        svm = _SVC2(kernel='rbf', C=2, gamma='scale',
                    class_weight={0: 1.0, 1: 1.8, 2: 1.0},
                    probability=True, random_state=42)
        svm.fit(X_data, y_data)

        # ── Compute decision grid + real boundary contours ─────────────────
        lo, hi = -3.5, 3.5
        xx, yy = np.meshgrid(np.linspace(lo, hi, 300), np.linspace(lo, hi, 300))
        Z_pred = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        import matplotlib as _mpl2
        _mpl2.use('Agg')
        import matplotlib.pyplot as _plt2
        _fig2, _mpl_ax2 = _plt2.subplots()
        _cs2 = _mpl_ax2.contour(xx, yy, Z_pred, levels=[0.5, 1.5])
        contours = []
        for _path in _cs2.get_paths():
            if len(_path.vertices) > 8:
                contours.append(_path.vertices.copy())
        _plt2.close(_fig2)

        sv_pts = svm.support_vectors_
        sv_cls = y_data[svm.support_]
        cls_colors = [self.CN, self.CP, self.CS]

        # ── Shared Manim axes ──────────────────────────────────────────────
        def make_ax():
            a = Axes(
                x_range=[lo, hi, 1], y_range=[lo, hi, 1],
                x_length=6.0, y_length=5.8,
                axis_config={"color": "#333333", "stroke_width": 1.5,
                             "include_tip": True},
            )
            a.move_to([0.3, -0.35, 0])
            return a

        def ax_labels(a):
            xl = Text("ch03-beta (standardised)", font_size=14, color=BLACK)
            xl.next_to(a.x_axis.get_right(), DOWN, buff=0.14)
            yl = Text("ch03-activity (standardised)", font_size=14, color=BLACK)
            yl.rotate(PI / 2).next_to(a.y_axis.get_top(), LEFT, buff=0.12)
            return xl, yl

        def scatter_dots(a):
            nd_ = VGroup(*[Dot(a.coords_to_point(x, y), radius=0.07, color=self.CN)
                           for x, y in raw[0]])
            pd_ = VGroup(*[Dot(a.coords_to_point(x, y), radius=0.07, color=self.CP)
                           for x, y in raw[1]])
            sd_ = VGroup(*[Dot(a.coords_to_point(x, y), radius=0.07, color=self.CS)
                           for x, y in raw[2]])
            return nd_, pd_, sd_

        def legend_col():
            items = [(self.CN, "Normal (0)"),
                     (self.CP, "Pre-Seizure (1)"),
                     (self.CS, "Seizure (2)")]
            g = VGroup(*[
                VGroup(Dot(radius=0.09, color=c),
                       Text(lbl, font_size=14, color=BLACK)).arrange(RIGHT, buff=0.10)
                for c, lbl in items
            ])
            return g.arrange(DOWN, buff=0.22)

        def boundary_curves(a):
            grp = VGroup()
            cols = [BLACK, "#888888"]
            for i, verts in enumerate(contours):
                vm = VMobject(stroke_color=cols[i % 2], stroke_width=2.5)
                pts = [a.coords_to_point(float(v[0]), float(v[1])) for v in verts]
                if len(pts) >= 2:
                    vm.set_points_as_corners(pts)
                grp.add(vm)
            return grp

        def region_fill(a):
            step = 0.22
            xs = np.arange(lo + 0.05, hi, step)
            ys = np.arange(lo + 0.05, hi, step)
            Xg = np.array([[x, y] for x in xs for y in ys])
            Zg = svm.predict(Xg)
            fills = [self.RN, self.RP, self.RS]
            g = VGroup()
            for (x, y), c in zip(Xg, Zg):
                sq = Square(side_length=step * 0.9,
                            fill_color=fills[c], fill_opacity=0.55, stroke_width=0)
                sq.move_to(a.coords_to_point(x, y))
                g.add(sq)
            return g

        def head(title_str, sub_str):
            t = Text(title_str, font_size=26, color=BLACK, weight=BOLD)
            t.to_edge(UP, buff=0.28)
            s = Text(sub_str, font_size=14, color="#555555")
            s.next_to(t, DOWN, buff=0.08)
            return t, s

        def code_footer(txt):
            return Text(txt, font_size=12, color="#444444").to_edge(DOWN, buff=0.26)

        def wipe():
            self.remove(*self.mobjects)

        # ── Stage 1: raw scatter ───────────────────────────────────────────
        t, s = head(
            "SVM Input: 2D Feature Slice  (ch03-beta vs ch03-activity)",
            "94 samples shown  ·  Data unscaled here; SVC trained on all 75 standardised features",
        )
        a = make_ax()
        xl, yl = ax_labels(a)
        nd_, pd_, sd_ = scatter_dots(a)
        leg = legend_col().next_to(a, RIGHT, buff=0.35)
        note = Text(
            "Note: 2D slice for visualisation only.\n"
            "Actual SVC operates in 75-dimensional space.",
            font_size=13, color="#777777",
        ).next_to(leg, DOWN, buff=0.4)
        cf = code_footer(
            "x_svm = selector.transform(scaler.transform(x_raw))  — scaled + top-75 features"
        )
        self.add(t, s, a, xl, yl, nd_, pd_, sd_, leg, note, cf)
        self.wait(0.1)
        _save_report(self, "SVM_1_raw_scatter")
        wipe()

        # ── Stage 2: StandardScaler ────────────────────────────────────────
        t, s = head(
            "StandardScaler: Zero Mean, Unit Variance",
            "Fitted on X_train only  ·  Applied to both train and inference data",
        )
        a = make_ax()
        xl, yl = ax_labels(a)
        nd_, pd_, sd_ = scatter_dots(a)
        leg = legend_col().next_to(a, RIGHT, buff=0.35)

        mu_line = DashedLine(a.coords_to_point(lo + 0.1, 0), a.coords_to_point(hi - 0.1, 0),
                             dash_length=0.12, stroke_width=2.0, color=self.CB)
        sd_p = DashedLine(a.coords_to_point(lo + 0.1, 1), a.coords_to_point(hi - 0.1, 1),
                          dash_length=0.08, stroke_width=1.2, color="#AAAAAA")
        sd_n = DashedLine(a.coords_to_point(lo + 0.1, -1), a.coords_to_point(hi - 0.1, -1),
                          dash_length=0.08, stroke_width=1.2, color="#AAAAAA")
        mu_lbl  = Text("μ = 0", font_size=15, color=self.CB, weight=BOLD).next_to(mu_line, RIGHT, buff=0.08)
        sdp_lbl = Text("+1σ", font_size=13, color="#888888").next_to(sd_p, RIGHT, buff=0.06)
        sdn_lbl = Text("−1σ", font_size=13, color="#888888").next_to(sd_n, RIGHT, buff=0.06)

        cf = code_footer(
            "scaler = StandardScaler().fit(X_train)  |  x_scaled = scaler.transform(x_raw)"
        )
        self.add(t, s, a, xl, yl, nd_, pd_, sd_, leg,
                 mu_line, sd_p, sd_n, mu_lbl, sdp_lbl, sdn_lbl, cf)
        self.wait(0.1)
        _save_report(self, "SVM_2_scaler")
        wipe()

        # ── Stage 3: RBF kernel concept ────────────────────────────────────
        t, s = head(
            "RBF Kernel: Implicit High-Dimensional Mapping",
            "K(xᵢ, xⱼ) = exp(−γ ||xᵢ − xⱼ||²)   where γ = 1 / (n_features × Var(X))",
        )
        a = make_ax()
        xl, yl = ax_labels(a)
        # Slightly spread dots to hint at kernel expansion
        rng_l = np.random.default_rng(55)
        nd_l = VGroup(*[Dot(a.coords_to_point(x + float(rng_l.uniform(-0.18, 0.18)),
                                               y + float(rng_l.uniform(-0.38, -0.08))),
                            radius=0.07, color=self.CN) for x, y in raw[0]])
        pd_l = VGroup(*[Dot(a.coords_to_point(x + float(rng_l.uniform(-0.12, 0.12)),
                                               y + float(rng_l.uniform(0.18, 0.45))),
                            radius=0.07, color=self.CP) for x, y in raw[1]])
        sd_l = VGroup(*[Dot(a.coords_to_point(x + float(rng_l.uniform(0.10, 0.28)),
                                               y + float(rng_l.uniform(-0.38, -0.10))),
                            radius=0.07, color=self.CS) for x, y in raw[2]])
        leg = legend_col().next_to(a, RIGHT, buff=0.35)

        hp = a.plot(lambda x: 0.28, x_range=[lo + 0.2, hi - 0.2],
                    color=self.CV, stroke_width=2.5)
        hp_lbl = Text("Separating hyperplane in kernel space", font_size=14, color=self.CV)
        hp_lbl.next_to(hp, UP, buff=0.10)

        cf = code_footer(
            "SVC(kernel='rbf', C=2, gamma='scale')  — gamma auto = 1/(75 × Var(X_train))"
        )
        self.add(t, s, a, xl, yl, nd_l, pd_l, sd_l, leg, hp, hp_lbl, cf)
        self.wait(0.1)
        _save_report(self, "SVM_3_rbf_kernel")
        wipe()

        # ── Stage 4: Support vectors ───────────────────────────────────────
        t, s = head(
            f"Support Vectors  (C = 2)  —  {len(sv_pts)} total",
            "Points closest to the margin boundary  ·  Ringed in their class colour",
        )
        a = make_ax()
        xl, yl = ax_labels(a)
        nd_, pd_, sd_ = scatter_dots(a)
        leg = legend_col().next_to(a, RIGHT, buff=0.35)

        sv_rings = VGroup()
        for pt, cls in zip(sv_pts, sv_cls):
            ring = Circle(radius=0.14, color=cls_colors[cls], stroke_width=2.8)
            ring.move_to(a.coords_to_point(float(pt[0]), float(pt[1])))
            sv_rings.add(ring)

        bdy = boundary_curves(a)

        ann = VGroup(
            Text(f"n_support = {len(sv_pts)}", font_size=14, color=BLACK),
            Text("class_weight = {0:1.0, 1:1.8, 2:1.0}", font_size=13, color=self.CP),
        ).arrange(DOWN, buff=0.14).next_to(leg, DOWN, buff=0.3)

        cf = code_footer(
            "svm.support_vectors_  |  svm.support_  |  Rings drawn at real SV locations"
        )
        self.add(t, s, a, xl, yl, nd_, pd_, sd_, leg, sv_rings, bdy, ann, cf)
        self.wait(0.1)
        _save_report(self, "SVM_4_support_vectors")
        wipe()

        # ── Stage 5: Real decision boundary ───────────────────────────────
        t, s = head(
            "RBF SVM Decision Boundary  —  Computed from Fitted Model",
            "Boundary from svm.predict() on 300×300 grid  ·  Shaded = predicted class region",
        )
        a = make_ax()
        xl, yl = ax_labels(a)
        nd_, pd_, sd_ = scatter_dots(a)
        leg = legend_col().next_to(a, RIGHT, buff=0.35)

        regions = region_fill(a)
        bdy     = boundary_curves(a)

        weight_ann = Text(
            "Pre-Seizure weight = 1.8×\n→ boundary shifted\nto reduce missed detections",
            font_size=13, color=self.CP,
        ).next_to(leg, DOWN, buff=0.32)

        cf = code_footer(
            "SVC(kernel='rbf', C=2, gamma='scale', class_weight={0:1.0, 1:1.8, 2:1.0},"
            " probability=True, random_state=42)"
        )
        self.add(t, s, a, xl, yl, regions, nd_, pd_, sd_, bdy, leg, weight_ann, cf)
        self.wait(0.1)
        _save_report(self, "SVM_5_decision_boundary")
        wipe()

        # ── Stage 6: Platt scaling ─────────────────────────────────────────
        t, s = head(
            "Platt Scaling: SVM Decision Score → Calibrated Probability",
            "5-fold cross-validated sigmoid fit  ·  Enabled by  probability=True  in sklearn SVC",
        )

        sig_ax = Axes(
            x_range=[-5, 5, 1], y_range=[0, 1, 0.25],
            x_length=7.5, y_length=4.0,
            axis_config={"color": "#333333", "stroke_width": 1.5, "include_tip": True},
        )
        sig_ax.move_to([0.0, -0.85, 0])
        sx = Text("Raw SVM decision function score  f(x)", font_size=15, color=BLACK)
        sx.next_to(sig_ax.x_axis.get_right(), DOWN, buff=0.13)
        sy = Text("P(y | x)", font_size=15, color=BLACK).rotate(PI / 2)
        sy.next_to(sig_ax.y_axis.get_top(), LEFT, buff=0.12)

        sig_curve = sig_ax.plot(lambda x: 1 / (1 + np.exp(-x)),
                                x_range=[-5, 5], color=self.CB, stroke_width=3)

        score_map = [
            (-1.62, self.CN, "Normal",      0.13),
            ( 0.78, self.CP, "Pre-Seizure", 0.30),
            ( 2.10, self.CS, "Seizure",     0.57),
        ]
        dots_s = VGroup()
        dots_p = VGroup()
        lines  = VGroup()
        anns   = VGroup()
        for raw_s, col, lbl, prob in score_map:
            pv = 1 / (1 + np.exp(-raw_s))
            sp = Dot(sig_ax.coords_to_point(raw_s, 0),    color=col, radius=0.11)
            pp = Dot(sig_ax.coords_to_point(raw_s, pv),   color=col, radius=0.11)
            vl = DashedLine(sig_ax.coords_to_point(raw_s, 0),
                            sig_ax.coords_to_point(raw_s, pv),
                            dash_length=0.08, stroke_width=1.2, color=col)
            hl = DashedLine(sig_ax.coords_to_point(-5, pv),
                            sig_ax.coords_to_point(raw_s, pv),
                            dash_length=0.08, stroke_width=1.2, color=col)
            an = Text(f"{lbl}: p = {prob}", font_size=13, color=col, weight=BOLD)
            an.next_to(pp, RIGHT, buff=0.10)
            dots_s.add(sp); dots_p.add(pp)
            lines.add(vl, hl); anns.add(an)

        out_box = RoundedRectangle(corner_radius=0.10, width=9.0, height=0.52,
                                   fill_color="#F4F4F4", fill_opacity=1.0,
                                   stroke_color=BLACK, stroke_width=1.0)
        out_box.to_edge(UP, buff=2.52)
        out_txt = Text(
            "svm.predict_proba(x_svm)  →  [Normal: 0.13,  Pre-Seizure: 0.30,  Seizure: 0.57]",
            font_size=15, color=BLACK,
        ).move_to(out_box.get_center())

        cf = code_footer(
            "Platt scaling fitted internally during svm.fit()  |"
            "  x_svm = selector.transform(scaler.transform(x_raw))"
        )

        self.add(t, s, sig_ax, sx, sy, sig_curve, lines, dots_s, dots_p, anns,
                 out_box, out_txt, cf)
        self.wait(0.1)
        _save_report(self, "SVM_6_platt_scaling")
        self.remove(*self.mobjects)


# ══════════════════════════════════════════════════════════════════════════════
#  ProjectGanttScene  —  Gantt chart project timeline Aug 2025 – May 2026
#
#  Render:
#    manim -pql neurowatch_animations.py ProjectGanttScene   (draft)
#    manim -qh  neurowatch_animations.py ProjectGanttScene   (high quality)
# ══════════════════════════════════════════════════════════════════════════════

class ProjectGanttScene(Scene):
    def construct(self):
        self.camera.background_color = DARK_BG

        # ── Timeline data ──────────────────────────────────────────────────
        MONTHS = ["Aug", "Sep", "Oct", "Nov", "Jan", "Feb", "Mar", "Apr", "May"]
        YEARS  = ["2025", "",   "",    "",    "2026","",    "",    "",    ""   ]

        # (label, start_col, end_col_exclusive, hex_colour)
        # Columns 0-3 = Semester 1 (Aug–Nov 2025)
        # Columns 4-8 = Semester 2 (Jan–May 2026)
        TASKS = [
            ("Literature Review",        0, 2, BLUE_ACC),
            ("Methodology Design",       1, 4, PURPLE_ACC),
            ("ML Model Development",     2, 4, TEAL),
            ("Circuit Simulation",       3, 4, ORANGE_ACC),
            ("Hardware Implementation",  4, 6, NEON_GREEN),
            ("Monitoring Dashboard",     5, 7, "#4cc9f0"),
            ("Patient User Interface",   6, 8, GOLD_ACC),
            ("Results & Analysis",       7, 9, AMBER),
            ("Report Writing",           7, 9, NEON_YELLOW),
        ]

        N_MONTHS  = len(MONTHS)
        N_TASKS   = len(TASKS)
        SEM1_COLS = 4
        SEM2_COLS = 5

        # ── Layout constants ───────────────────────────────────────────────
        LABEL_CX   = -5.05
        CHART_L    = -3.0
        CHART_R    =  6.8
        COL_W      = (CHART_R - CHART_L) / N_MONTHS
        TITLE_Y    =  3.65
        HEADER_Y   =  3.05
        ROW_TOP_Y  =  2.60
        ROW_H      =  0.54
        BAR_H      =  0.36
        BOTTOM_Y   =  ROW_TOP_Y - N_TASKS * ROW_H   # ≈ -2.26

        # ── Title ──────────────────────────────────────────────────────────
        title = Text("NeuroWatch — Project Timeline", font_size=32,
                     weight=BOLD, color=WHITE)
        subtitle = Text("BIUST Final Year Project  |  Aug 2025 – May 2026",
                        font_size=15, color=GOLD_ACC)
        title.move_to([0, TITLE_Y, 0])
        subtitle.next_to(title, DOWN, buff=0.10)
        self.play(Write(title), FadeIn(subtitle), run_time=1.0)

        # ── Semester background blocks ──────────────────────────────────────
        blk_h  = N_TASKS * ROW_H + 0.10
        blk_cy = ROW_TOP_Y - blk_h / 2 + 0.05

        sem1_w  = SEM1_COLS * COL_W
        sem2_w  = SEM2_COLS * COL_W
        sem1_cx = CHART_L + sem1_w / 2
        sem2_cx = CHART_L + SEM1_COLS * COL_W + sem2_w / 2

        sem1_bg = Rectangle(
            width=sem1_w, height=blk_h,
            fill_color=BLUE_ACC, fill_opacity=0.08,
            stroke_color=BLUE_ACC, stroke_width=0.8, stroke_opacity=0.45,
        ).move_to([sem1_cx, blk_cy, 0])

        sem2_bg = Rectangle(
            width=sem2_w, height=blk_h,
            fill_color=NEON_GREEN, fill_opacity=0.07,
            stroke_color=NEON_GREEN, stroke_width=0.8, stroke_opacity=0.45,
        ).move_to([sem2_cx, blk_cy, 0])

        self.play(FadeIn(sem1_bg), FadeIn(sem2_bg), run_time=0.5)

        # ── Month header row ───────────────────────────────────────────────
        month_grp = VGroup()
        for i, (m, y) in enumerate(zip(MONTHS, YEARS)):
            cx = CHART_L + (i + 0.5) * COL_W
            top = Text(m, font_size=13, color=WHITE)
            top.move_to([cx, HEADER_Y + 0.10, 0])
            month_grp.add(top)
            if y:
                yr = Text(y, font_size=10, color=GRAY_A)
                yr.next_to(top, DOWN, buff=0.03)
                month_grp.add(yr)

        header_line = Line(
            [CHART_L, HEADER_Y - 0.18, 0],
            [CHART_R, HEADER_Y - 0.18, 0],
            color=GRAY, stroke_width=0.9,
        )
        self.play(Write(month_grp), Create(header_line), run_time=0.7)

        # ── Vertical grid + semester divider ───────────────────────────────
        grid = VGroup(*[
            Line(
                [CHART_L + i * COL_W, HEADER_Y - 0.18, 0],
                [CHART_L + i * COL_W, BOTTOM_Y,        0],
                color=GRAY, stroke_width=0.4, stroke_opacity=0.25,
            )
            for i in range(N_MONTHS + 1)
        ])
        div_x = CHART_L + SEM1_COLS * COL_W
        sem_div = DashedLine(
            [div_x, HEADER_Y - 0.18, 0],
            [div_x, BOTTOM_Y,        0],
            color=GOLD_ACC, stroke_width=1.6, stroke_opacity=0.75,
            dash_length=0.13,
        )
        self.play(Create(grid), Create(sem_div), run_time=0.45)

        # ── Task bars ──────────────────────────────────────────────────────
        for j, (name, s, e, clr) in enumerate(TASKS):
            y_c = ROW_TOP_Y - (j + 0.5) * ROW_H

            lbl = Text(name, font_size=12, color=WHITE)
            lbl.move_to([LABEL_CX, y_c, 0])

            bx_l = CHART_L + s * COL_W + 0.05
            bx_r = CHART_L + e * COL_W - 0.05
            bar = RoundedRectangle(
                width=bx_r - bx_l, height=BAR_H,
                corner_radius=0.07,
                fill_color=clr, fill_opacity=0.88,
                stroke_width=0,
            ).move_to([(bx_l + bx_r) / 2, y_c, 0])

            row_sep = Line(
                [LABEL_CX - 1.6, y_c - ROW_H / 2, 0],
                [CHART_R,        y_c - ROW_H / 2, 0],
                color=GRAY, stroke_width=0.3, stroke_opacity=0.20,
            )

            self.play(
                Write(lbl),
                GrowFromEdge(bar, LEFT),
                Create(row_sep),
                run_time=0.42,
            )

        # ── Semester labels at bottom ───────────────────────────────────────
        sem1_lbl = Text("SEMESTER 1", font_size=11, weight=BOLD, color=BLUE_ACC)
        sem1_lbl.move_to([sem1_cx, BOTTOM_Y - 0.28, 0])
        sem2_lbl = Text("SEMESTER 2", font_size=11, weight=BOLD, color=NEON_GREEN)
        sem2_lbl.move_to([sem2_cx, BOTTOM_Y - 0.28, 0])

        # ── Milestone checkmarks ────────────────────────────────────────────
        ck1 = Text("✓  Circuit Simulation done", font_size=12, color=NEON_GREEN)
        ck1.move_to([sem1_cx, BOTTOM_Y - 0.55, 0])
        ck2 = Text("✓  Hardware & Report done", font_size=12, color=NEON_GREEN)
        ck2.move_to([sem2_cx, BOTTOM_Y - 0.55, 0])

        self.play(
            Write(sem1_lbl), Write(sem2_lbl),
            run_time=0.5,
        )
        self.play(Write(ck1), Write(ck2), run_time=0.6)
        self.wait(3.0)
