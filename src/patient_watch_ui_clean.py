"""
NeuroWatch --- Patient Home Monitor
A wearable-style, patient-friendly UI inspired by smartwatch health dashboards.
Shows: status ring, brain activity level, recent history, and emergency contact.
No medical jargon --- plain English only.
"""

import json
import os
import math
import tkinter as tk
from datetime import datetime

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ------ Paths (same as main_pi.py / desktop_dashboard.py) ---------------------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATIENTS_JSON  = os.path.join(BASE_DIR, "json", "neurowatch_patients.json")
STATUS_JSON    = os.path.join(BASE_DIR, "json", "neurowatch_status.json")
HISTORY_PREFIX = "neurowatch_history_"
PHOTO_DIR      = os.path.join(BASE_DIR, "patient_photos")
HOME_PATIENT_ID = "P001"

# ------ Palette ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
BG        = "#0d0f1a"
PANEL     = "#13162a"
CARD      = "#181c30"
BORDER    = "#252a45"

GREEN     = "#00e676"
AMBER     = "#ffca28"
RED       = "#ff5252"
BLUE      = "#40c4ff"
MUTED     = "#5a6080"
TEXT      = "#e8ecf4"
SUBTEXT   = "#8a93b0"

STATE_COLORS = {
    "Seizure":    RED,
    "Pre-Seizure": AMBER,
    "Normal":     GREEN,
}
STATE_LABELS = {
    "Seizure":    "---  Alert --- Please stay calm",
    "Pre-Seizure": "---  Stay relaxed",
    "Normal":     "---  You're doing great",
}
STATE_ADVICE = {
    "Seizure":    "Your caregiver has been notified.",
    "Pre-Seizure": "Find a safe, comfortable place to sit or lie down and take medication.",
    "Normal":     "Everything looks normal. Keep resting.",
}

DEFAULT_EMERGENCY_PHONE = "74390351"

RING_SIZE   = 220   # diameter of the status ring canvas
RING_W      = 18    # ring stroke width
PHOTO_SIZE  = (86, 86)
HISTORY_DOTS = 30


class PatientUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NeuroWatch --- My Health Monitor")
        self.root.geometry("480x780")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)

        self._photo_ref = None
        self._state     = "Normal"
        self._name      = "---"
        self._history   = []
        self._last_ts   = "---"
        self._emergency = "---"
        self._streak    = 0     # consecutive Normal readings

        self._build_ui()
        self._refresh()

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # UI CONSTRUCTION
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _build_ui(self):
        root = self.root

        # ------ Top bar ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        top = tk.Frame(root, bg=BG)
        top.pack(fill="x", padx=20, pady=(18, 0))

        self.lbl_greeting = tk.Label(
            top, text="Hello ----", bg=BG, fg=SUBTEXT,
            font=("Segoe UI", 11),
        )
        self.lbl_greeting.pack(side="left")

        self.lbl_time = tk.Label(
            top, text="---", bg=BG, fg=MUTED,
            font=("Segoe UI", 11),
        )
        self.lbl_time.pack(side="right")

        self.lbl_name = tk.Label(
            root, text="---", bg=BG, fg=TEXT,
            font=("Segoe UI", 18, "bold"),
        )
        self.lbl_name.pack(pady=(4, 0))

        # ------ Status ring ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ring_frame = tk.Frame(root, bg=BG)
        ring_frame.pack(pady=(10, 0))

        self.ring_canvas = tk.Canvas(
            ring_frame, width=RING_SIZE, height=RING_SIZE,
            bg=BG, highlightthickness=0,
        )
        self.ring_canvas.pack()

        # ------ Status label under ring ---------------------------------------------------------------------------------------------------------------------------------
        self.lbl_status_main = tk.Label(
            root, text="---  You're doing great", bg=BG, fg=GREEN,
            font=("Segoe UI", 14, "bold"),
        )
        self.lbl_status_main.pack(pady=(6, 0))

        self.lbl_advice = tk.Label(
            root, text="Everything looks normal. Keep resting.",
            bg=BG, fg=SUBTEXT, font=("Segoe UI", 10),
            wraplength=380,
        )
        self.lbl_advice.pack(pady=(2, 0))

        # ------ Cards row ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        cards = tk.Frame(root, bg=BG)
        cards.pack(fill="x", padx=20, pady=(20, 0))
        cards.columnconfigure(0, weight=1)
        cards.columnconfigure(1, weight=1)

        self.card_streak = self._make_card(cards, "---- Good Streak", "0 readings", 0, 0)
        self.card_updated = self._make_card(cards, "---- Last Checked", "---", 0, 1)

        # ------ History strip ---------------------------------------------------------------------------------------------------------------------------------------------------------------
        hist_label = tk.Label(
            root, text="Recent Activity", bg=BG, fg=SUBTEXT,
            font=("Segoe UI", 10, "bold"),
        )
        hist_label.pack(anchor="w", padx=24, pady=(18, 4))

        self.hist_canvas = tk.Canvas(
            root, height=36, bg=BG, highlightthickness=0,
        )
        self.hist_canvas.pack(fill="x", padx=24)

        # Legend
        legend_frame = tk.Frame(root, bg=BG)
        legend_frame.pack(anchor="w", padx=24, pady=(4, 0))
        for color, label in [(GREEN, "Normal"), (AMBER, "Heads-up"), (RED, "Alert")]:
            dot = tk.Canvas(legend_frame, width=10, height=10, bg=BG, highlightthickness=0)
            dot.pack(side="left")
            dot.create_oval(1, 1, 9, 9, fill=color, outline="")
            tk.Label(legend_frame, text=label, bg=BG, fg=SUBTEXT,
                     font=("Segoe UI", 9)).pack(side="left", padx=(2, 12))

        # ------ Emergency contact card ------------------------------------------------------------------------------------------------------------------------------------
        emer_frame = tk.Frame(root, bg=CARD, bd=0, relief="flat")
        emer_frame.pack(fill="x", padx=20, pady=(20, 0))
        emer_frame.configure(highlightbackground=BORDER, highlightthickness=1)

        tk.Label(
            emer_frame, text="----  Emergency Contact",
            bg=CARD, fg=SUBTEXT, font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", padx=14, pady=(10, 2))

        self.lbl_emergency = tk.Label(
            emer_frame, text="---", bg=CARD, fg=TEXT,
            font=("Segoe UI", 12, "bold"),
        )
        self.lbl_emergency.pack(anchor="w", padx=14, pady=(0, 12))

        # ------ Live indicator dot ------------------------------------------------------------------------------------------------------------------------------------------------
        bot = tk.Frame(root, bg=BG)
        bot.pack(fill="x", padx=24, pady=(16, 0))

        self._blink_dot = tk.Canvas(bot, width=10, height=10, bg=BG, highlightthickness=0)
        self._blink_dot.pack(side="left")
        self._blink_dot.create_oval(1, 1, 9, 9, fill=GREEN, outline="", tags="dot")

        tk.Label(
            bot, text="Live monitoring active",
            bg=BG, fg=MUTED, font=("Segoe UI", 9),
        ).pack(side="left", padx=6)

        self._blink_state = True
        self._blink()

    def _make_card(self, parent, title, value, row, col):
        """Returns a dict with references to the card's value label."""
        f = tk.Frame(parent, bg=CARD, bd=0)
        f.configure(highlightbackground=BORDER, highlightthickness=1)
        f.grid(row=row, column=col, sticky="nsew",
               padx=(0, 6) if col == 0 else (6, 0), pady=4)

        tk.Label(f, text=title, bg=CARD, fg=SUBTEXT,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=12, pady=(8, 0))
        lbl = tk.Label(f, text=value, bg=CARD, fg=TEXT,
                       font=("Segoe UI", 13, "bold"))
        lbl.pack(anchor="w", padx=12, pady=(2, 10))
        return lbl

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # RING DRAWING
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _draw_ring(self, state, confidence):
        c     = self.ring_canvas
        c.delete("all")
        cx    = RING_SIZE // 2
        cy    = RING_SIZE // 2
        r     = (RING_SIZE - RING_W * 2) // 2
        x0, y0 = cx - r, cy - r
        x1, y1 = cx + r, cy + r

        color = STATE_COLORS.get(state, GREEN)

        # Background ring
        c.create_arc(x0, y0, x1, y1,
                     start=0, extent=359.9,
                     style="arc", outline=BORDER,
                     width=RING_W)

        # Filled arc based on confidence (0---1 --- 0---360--)
        extent = max(10, confidence * 360)
        c.create_arc(x0, y0, x1, y1,
                     start=90, extent=-extent,
                     style="arc", outline=color,
                     width=RING_W)

        # Glow dot at tip
        tip_angle = math.radians(90 - extent)
        tx = cx + r * math.cos(tip_angle)
        ty = cy - r * math.sin(tip_angle)
        glow = RING_W // 2 + 3
        c.create_oval(tx - glow, ty - glow, tx + glow, ty + glow,
                      fill=color, outline="")

        # Centre content
        # Confidence percentage
        pct = int(confidence * 100)
        c.create_text(cx, cy - 22,
                      text=f"{pct}%",
                      fill=color, font=("Segoe UI", 30, "bold"))
        c.create_text(cx, cy + 14,
                      text="brain activity", fill=SUBTEXT,
                      font=("Segoe UI", 10))

        # State pill
        pill_text = {"Seizure": "ALERT", "Pre-Seizure": "HEADS UP", "Normal": "NORMAL"}.get(state, state)
        pill_w, pill_h = 90, 26
        px0 = cx - pill_w // 2
        py0 = cy + 36
        px1 = cx + pill_w // 2
        py1 = py0 + pill_h
        c.create_rounded_rect = lambda *a, **kw: self._rounded_rect(c, *a, **kw)
        self._rounded_rect(c, px0, py0, px1, py1, r=13, fill=self._dim(color), outline="")
        c.create_text(cx, (py0 + py1) // 2,
                      text=pill_text, fill=color,
                      font=("Segoe UI", 9, "bold"))

    @staticmethod
    def _dim(hex_color, factor=0.20):
        """Return a much-darker version of the colour for pill backgrounds."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r, g, b = int(r * factor), int(g * factor), int(b * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _rounded_rect(canvas, x0, y0, x1, y1, r=10, **kwargs):
        canvas.create_polygon(
            x0 + r, y0,
            x1 - r, y0,
            x1, y0,
            x1, y0 + r,
            x1, y1 - r,
            x1, y1,
            x1 - r, y1,
            x0 + r, y1,
            x0, y1,
            x0, y1 - r,
            x0, y0 + r,
            x0, y0,
            smooth=True, **kwargs,
        )

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # HISTORY STRIP
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _draw_history(self, states):
        c = self.hist_canvas
        c.delete("all")
        c.update_idletasks()
        width  = c.winfo_width() or 432
        height = 36
        n      = HISTORY_DOTS
        recent = states[-n:]
        if not recent:
            c.create_text(10, height // 2, anchor="w", fill=MUTED, text="No data yet")
            return

        dot_d  = 18
        gap    = (width - n * dot_d) / max(n - 1, 1)
        for i, s in enumerate(recent):
            color = STATE_COLORS.get(s, MUTED)
            x = i * (dot_d + gap)
            y = (height - dot_d) / 2
            # Glow for latest
            if i == len(recent) - 1:
                self._rounded_rect(c, x - 3, y - 3, x + dot_d + 3, y + dot_d + 3,
                                   r=11, fill=self._dim(color, 0.3), outline="")
            self._rounded_rect(c, x, y, x + dot_d, y + dot_d,
                               r=9, fill=color, outline="")

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # BLINKING LIVE DOT
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _blink(self):
        self._blink_state = not self._blink_state
        self._blink_dot.itemconfig("dot", fill=GREEN if self._blink_state else BG)
        self.root.after(900, self._blink)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _read_json(self, path, default):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _load_patient(self):
        raw = self._read_json(PATIENTS_JSON, {})
        patients = list(raw.values()) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
        # Prefer HOME_PATIENT_ID
        match = next((p for p in patients if p.get("patient_id") == HOME_PATIENT_ID), None)
        return match or (patients[0] if patients else None)

    def _load_history(self, patient_id):
        path = os.path.join(BASE_DIR, f"{HISTORY_PREFIX}{patient_id}.json")
        raw = self._read_json(path, [])
        return raw[-HISTORY_DOTS * 2:] if isinstance(raw, list) else []

    def _format_time_short(self, iso):
        if not iso:
            return "---"
        try:
            dt = datetime.fromisoformat(iso) if "T" in iso else datetime.strptime(iso, "%Y-%m-%d %H:%M:%S")
            return dt.strftime("%H:%M")
        except Exception:
            return "---"

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # REFRESH LOOP
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _refresh(self):
        patient = self._load_patient()

        if patient:
            pid     = patient.get("patient_id", "")
            name    = patient.get("patient_name", "Patient")
            state   = patient.get("state", "Normal")
            conf    = float(patient.get("confidence", 0.75))
            ts      = patient.get("timestamp", "")
            emer    = patient.get("emergency_contact", "---")
            emer_no = str(patient.get("emergency_phone", DEFAULT_EMERGENCY_PHONE))
            sms_sent = bool(patient.get("sms_sent", False))
            history = self._load_history(pid)

            self._state   = state
            self._history = [r.get("state", "Normal") for r in history]

            # Count consecutive normals for streak
            streak = 0
            for s in reversed(self._history):
                if s == "Normal":
                    streak += 1
                else:
                    break
            self._streak = streak

            # Update UI
            first_name = name.split()[0]
            self.lbl_greeting.config(text=f"Hello, {first_name} ----")
            self.lbl_name.config(text=name)
            status_text = STATE_LABELS.get(state, state)
            if state == "Seizure" and sms_sent:
                status_text = "---  Alert --- Doctor contacted"
            self.lbl_status_main.config(
                text=status_text,
                fg=STATE_COLORS.get(state, GREEN),
            )
            advice_text = STATE_ADVICE.get(state, "")
            if state == "Seizure":
                advice_text = (
                    "Doctor has been contacted. Please stay calm and wait for assistance."
                    if sms_sent
                    else "CONTACT DOCTOR/EMERGENCY"
                )
            self.lbl_advice.config(text=advice_text)
            self.lbl_emergency.config(text=f"{emer}\n{emer_no}")

            # Cards
            self.card_streak.config(
                text=f"{self._streak} reading{'s' if self._streak != 1 else ''}",
                fg=GREEN if self._streak >= 5 else TEXT,
            )
            self.card_updated.config(text=self._format_time_short(ts))

            self._draw_ring(state, conf)
            self._draw_history(self._history)
        else:
            # No data --- show placeholders
            self.lbl_name.config(text="Waiting for data---")
            self._draw_ring("Normal", 0.0)
            self._draw_history([])

        # Clock
        self.lbl_time.config(text=datetime.now().strftime("%H:%M"))

        # Schedule next refresh (2 s)
        self.root.after(2000, self._refresh)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def main():
    root = tk.Tk()
    app = PatientUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
