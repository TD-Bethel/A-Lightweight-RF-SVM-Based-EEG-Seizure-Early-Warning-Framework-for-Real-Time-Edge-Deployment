# ---------------------------------------------------------------
# Raspberry Pi - LCD Controller for NeuroWatch
# 20x4 HD44780 LCD, matches main_pi_bios_v15.py display output
#
# PURPOSE: Runs on the Raspberry Pi (or alongside the main system on
#   Windows for testing). Reads opcodes from proteus_bridge.py via a
#   shared virtual-serial file and drives the physical 20x4 LCD,
#   LEDs, and buzzer accordingly. Display format mirrors exactly
#   what main_pi_bios_v15.py would show if it controlled the LCD
#   directly.
# ---------------------------------------------------------------

# Standard Python libraries for GPIO, timing, threads, filesystem, JSON, and timestamps
import RPi.GPIO as GPIO
from time import sleep, time
import threading
import os
import json
from datetime import datetime

# ============================================================
# SECTION 1 — GPIO PIN NUMBERS (BCM numbering)
# These are the Raspberry Pi GPIO pins wired to the LCD and
# peripheral hardware. If you re-wire the hardware, update
# the numbers here. BCM numbering refers to the chip's logical
# pin numbers, NOT the physical header position.
# ============================================================
LCD_RS = 26   # Register Select: HIGH = data, LOW = command
LCD_E  = 19   # Enable pulse triggers LCD to read the data bus
LCD_D4 = 13   # Data bus bit 4 (4-bit mode uses D4–D7 only)
LCD_D5 = 6    # Data bus bit 5
LCD_D6 = 5    # Data bus bit 6
LCD_D7 = 20   # Data bus bit 7
GREEN_LED = 27  # Green LED (heartbeat blink — system alive)
RED_LED   = 22  # Red LED (alarm — flashes during pre-seizure / seizure)
BUZZER    = 4   # Buzzer (alarm — same timing as Red LED)

# ============================================================
# SECTION 2 — LCD PROTOCOL CONSTANTS
# The HD44780 controller understands 1-bit flags.
#   LCD_WIDTH : how many characters fit on one row (20-column display)
#   LCD_CHR   : True  → we are sending a printable character
#   LCD_CMD   : False → we are sending a command (clear, set cursor, etc.)
#   LCD_LINE_x: DDRAM address for the start of each row
#               These are fixed by the HD44780 spec for a 20x4 display.
# ============================================================
LCD_WIDTH = 20           # 20-column display
LCD_CHR = True           # Mode flag: character data
LCD_CMD = False          # Mode flag: command data
LCD_LINE_1 = 0x80        # Row 1 DDRAM start address
LCD_LINE_2 = 0xC0        # Row 2 DDRAM start address
LCD_LINE_3 = 0x94        # Row 3 DDRAM start address (20-col offset)
LCD_LINE_4 = 0xD4        # Row 4 DDRAM start address (20-col offset)

# ============================================================
# SECTION 3 — HD44780 TIMING CONSTANTS (seconds)
# The HD44780 needs small delays between signal changes.
#   E_PULSE    : how long the Enable pin stays HIGH (strobe width)
#   E_DELAY    : setup/hold time around the Enable pulse
#   INIT_DELAY : longer pause during power-on initialisation sequence
#   CMD_DELAY  : pause after sending a command byte
#   CHAR_DELAY : pause after sending each character byte
# Making these too short causes garbled characters; too long
# makes the display noticeably slow.
# ============================================================
E_PULSE    = 0.0005   # 500 µs Enable pulse width
E_DELAY    = 0.0005   # 500 µs setup/hold delay around Enable
INIT_DELAY = 0.005    # 5 ms pause during power-on init
CMD_DELAY  = 0.002    # 2 ms pause after a command
CHAR_DELAY = 0.0001   # 100 µs pause after each character

# ============================================================
# SECTION 4 — VIRTUAL SERIAL FILE PATHS
# proteus_bridge.py (running on Windows) writes an opcode to
# VIRTUAL_SERIAL_RX.  This controller reads from that file.
# This controller writes "ACK\n" to VIRTUAL_SERIAL_TX so the
# bridge knows the LCD is alive (handshake).
#
# IMPORTANT: Both paths must be IDENTICAL in proteus_bridge.py
# and this file. If you move them, update both files.
# ============================================================
VIRTUAL_SERIAL_RX = r"C:\Users\thebe\AppData\Local\Temp\vsm_serial_rx.txt"
VIRTUAL_SERIAL_TX = r"C:\Users\thebe\AppData\Local\Temp\vsm_serial_tx.txt"

# How often the display loop refreshes (seconds)
LCD_REFRESH_RATE  = 0.3

# How fast the green heartbeat LED blinks (seconds per half-cycle)
GREEN_BLINK_RATE  = 0.5

# ============================================================
# SECTION 5 — JSON FILE PATHS
# main_pi_bios_v15.py continuously writes two JSON files:
#   neurowatch_status.json  → current confidence score
#   neurowatch_patients.json→ patient name, state, SMS status
# This controller reads them so the LCD can show live conf %
# and the correct patient name without re-implementing the
# full detection pipeline.
# ============================================================
_HERE        = os.path.dirname(os.path.abspath(__file__))
STATUS_JSON   = os.path.join(_HERE, "json", "neurowatch_status.json")
PATIENTS_JSON = os.path.join(_HERE, "json", "neurowatch_patients.json")

# ============================================================
# SECTION 6 — ESCALATION THRESHOLDS
# Must match the values used in main_pi_bios_v15.py so that
# the LCD escalation messages appear at the same time as the
# main engine would act on them.
#   PRESEIZURE_LONG_SECONDS: if Pre-Seizure lasts >= 12 s,
#     row 4 changes to "TAKE MEDICATION"
#   SEIZURE_LONG_SECONDS   : if Seizure lasts >= 30 s,
#     row 4 changes to "DR NOTIFIED" (if SMS sent) or "CONTACT DR"
# ============================================================
PRESEIZURE_LONG_SECONDS = 12
SEIZURE_LONG_SECONDS    = 30

# How long to show the mode-switch banner (seconds) before
# returning to the last detection state
MODE_BANNER_DURATION = 2.0

# ============================================================
# SECTION 7 — STATE MACHINE CONSTANTS
# Every possible system state maps to an integer constant so
# the rest of the code compares integers, not strings.
# The states are:
#   BOOT/INIT/LOADING/PROCESSING/TRAINING/READY
#     → startup and model-load phases (before live monitoring)
#   NORMAL/PRE_SEIZURE/SEIZURE
#     → live EEG monitoring states (main operating mode)
#   RESTARTING / WAITING_MODEL
#     → transient states between operating modes
#   DATASET_MODE / SIGGEN_MODE
#     → alternative input modes (shown briefly then return to last state)
#   HEARTBEAT
#     → keepalive opcode from bridge (no display change)
# ============================================================
class SystemState:
    BOOT          = 0
    INIT          = 1
    LOADING       = 2
    PROCESSING    = 3
    TRAINING      = 4
    READY         = 5
    NORMAL        = 6
    PRE_SEIZURE   = 7
    SEIZURE       = 8
    RESTARTING    = 9
    WAITING_MODEL = 10
    DATASET_MODE  = 11
    SIGGEN_MODE   = 12
    HEARTBEAT     = 13

# ============================================================
# SECTION 8 — LOW-LEVEL LCD DRIVER FUNCTIONS
# These three functions form the HD44780 4-bit bus driver.
# They are called by lcd_string() and lcd_clear().
# You should not need to change anything here unless you
# change the physical wiring.
# ============================================================

def lcd_toggle_enable():
    """Pulse the Enable pin HIGH → LOW to latch data into the LCD."""
    sleep(E_DELAY)
    GPIO.output(LCD_E, True)
    sleep(E_PULSE)
    GPIO.output(LCD_E, False)
    sleep(E_DELAY)

def lcd_send_byte(bits, mode):
    """
    Send one full byte to the LCD in two 4-bit nibbles.
    mode=LCD_CHR → data register (printing a character)
    mode=LCD_CMD → instruction register (sending a command)
    The HD44780 in 4-bit mode requires the high nibble first,
    then the low nibble, each followed by an Enable pulse.
    """
    GPIO.output(LCD_RS, mode)          # Tell LCD: data or command?
    sleep(E_DELAY)

    # High nibble (bits 4–7 of the byte)
    GPIO.output(LCD_D4, bool(bits & 0x10))
    GPIO.output(LCD_D5, bool(bits & 0x20))
    GPIO.output(LCD_D6, bool(bits & 0x40))
    GPIO.output(LCD_D7, bool(bits & 0x80))
    lcd_toggle_enable()

    # Low nibble (bits 0–3 of the byte)
    GPIO.output(LCD_D4, bool(bits & 0x01))
    GPIO.output(LCD_D5, bool(bits & 0x02))
    GPIO.output(LCD_D6, bool(bits & 0x04))
    GPIO.output(LCD_D7, bool(bits & 0x08))
    lcd_toggle_enable()

def lcd_init():
    """
    Power-on initialisation sequence for the HD44780 in 4-bit mode.
    The datasheet requires this exact sequence of nibble sends with
    specific delays. If you power-cycle the LCD, call this again.
    After init:
      0x28 → 4-bit mode, 2-line (works for 4-row too), 5x8 font
      0x0C → display ON, cursor OFF, blink OFF
      0x06 → cursor moves right after each character, no display shift
      0x01 → clear display, cursor home
    """
    sleep(0.050)                              # Wait for LCD power-on
    GPIO.output(LCD_RS, False)
    # Three-pulse reset sequence (HD44780 datasheet requirement)
    GPIO.output(LCD_D4, True);  GPIO.output(LCD_D5, True)
    GPIO.output(LCD_D6, False); GPIO.output(LCD_D7, False)
    lcd_toggle_enable(); sleep(INIT_DELAY)
    lcd_toggle_enable(); sleep(INIT_DELAY)
    lcd_toggle_enable(); sleep(INIT_DELAY)
    # Switch to 4-bit mode
    GPIO.output(LCD_D4, False); GPIO.output(LCD_D5, True)
    GPIO.output(LCD_D6, False); GPIO.output(LCD_D7, False)
    lcd_toggle_enable(); sleep(INIT_DELAY)
    # Function set, display control, entry mode, clear
    lcd_send_byte(0x28, LCD_CMD); sleep(CMD_DELAY)  # 4-bit, 2-line, 5x8
    lcd_send_byte(0x0C, LCD_CMD); sleep(CMD_DELAY)  # Display ON
    lcd_send_byte(0x06, LCD_CMD); sleep(CMD_DELAY)  # Cursor increment
    lcd_send_byte(0x01, LCD_CMD); sleep(CMD_DELAY)  # Clear
    print("LCD initialized successfully")

def lcd_string(message, line):
    """
    Write text to one row of the LCD.
    message : the string to display (truncated/padded to LCD_WIDTH=20)
    line    : one of LCD_LINE_1/2/3/4 — sets cursor to start of that row
    Padding with spaces ensures old characters are overwritten when
    the new text is shorter than the previous text on that row.
    """
    message = message.ljust(LCD_WIDTH, " ")[:LCD_WIDTH]   # Pad/truncate to 20
    lcd_send_byte(line, LCD_CMD)      # Move cursor to start of target row
    sleep(CMD_DELAY)
    for ch in message:
        lcd_send_byte(ord(ch), LCD_CHR)
        sleep(CHAR_DELAY)

def lcd_clear():
    """Send the HD44780 "clear display" command (0x01)."""
    lcd_send_byte(0x01, LCD_CMD)
    sleep(CMD_DELAY)

# ============================================================
# SECTION 9 — GPIO SETUP
# Sets every connected pin to OUTPUT and pulls it LOW.
# This runs once at import time (module level), before any
# threads or classes are created, so the hardware is ready
# before anything else tries to use it.
# ============================================================
GPIO.setwarnings(False)   # Suppress "already in use" warnings on restart
GPIO.setmode(GPIO.BCM)    # Use BCM chip-logical pin numbers (not physical)

# Configure all hardware pins as outputs, starting LOW
for p in [LCD_RS, LCD_E, LCD_D4, LCD_D5, LCD_D6, LCD_D7, GREEN_LED, RED_LED, BUZZER]:
    GPIO.setup(p, GPIO.OUT)
    GPIO.output(p, False)

sleep(0.1)      # Brief pause to let GPIO settle
lcd_init()      # Run HD44780 power-on sequence

# ============================================================
# SECTION 10 — GREEN LED HEARTBEAT CONTROLLER
# The green LED blinks continuously whenever the system is
# running, acting as a visible "I'm alive" indicator.
# It runs in its own daemon thread so it never blocks the
# main display loop.
# To stop blinking (e.g., during shutdown): call .stop()
# ============================================================
class GreenLEDController:
    def __init__(self):
        self.running = True    # Set to False to exit the thread
        self.enabled = True    # Set to False to pause blinking without stopping

    def worker(self):
        """Thread body: toggle green LED on/off at GREEN_BLINK_RATE."""
        while self.running:
            if self.enabled:
                GPIO.output(GREEN_LED, True)
                sleep(GREEN_BLINK_RATE)
                GPIO.output(GREEN_LED, False)
                sleep(GREEN_BLINK_RATE)
            else:
                sleep(0.1)   # Idle when disabled

    def start(self):
        """Launch the blink thread as a daemon (dies when main exits)."""
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop blinking and turn the LED off immediately."""
        self.running = False
        GPIO.output(GREEN_LED, False)

# ============================================================
# SECTION 11 — VIRTUAL SERIAL RECEIVER
# Reads opcodes written by proteus_bridge.py from the shared
# file at VIRTUAL_SERIAL_RX. Runs in a background thread so
# the main display loop is never blocked waiting for file I/O.
#
# Startup handshake:
#   init_files() writes "ACK\n" to VIRTUAL_SERIAL_TX so that
#   proteus_bridge.py's handshake() function knows the LCD
#   controller is ready.
#
# Opcode → state mapping (decode_message / worker):
#   BOOT → BOOT, INIT → INIT, LOAD → LOADING, PROC → PROCESSING
#   TRAIN → TRAINING, RDY → READY, NORM → NORMAL,
#   WARN → PRE_SEIZURE, SEIZ → SEIZURE, REST → RESTARTING, HB → HEARTBEAT
#
# JSON refresh:
#   Every 10 ticks (~1 s) _read_json() pulls the current
#   confidence score and patient name from the JSON files that
#   main_pi_bios_v15.py writes, so the LCD shows live data.
# ============================================================
class VirtualSerialReceiver:
    def __init__(self):
        self.current_state  = SystemState.WAITING_MODEL
        self.last_message   = ""
        self.running        = False
        self.model_started  = False    # True once first valid opcode arrives
        self.message_count  = 0        # Total opcodes received (for debug log)
        self.last_read_time = time()
        # Live values populated from JSON files
        self.current_conf   = 0.0      # Confidence score 0.0–1.0
        self.patient_name   = "John M."
        self.sms_sent       = False    # Whether SMS alert was sent
        self._json_tick     = 0        # Counter to throttle JSON reads
        self.init_files()

    def init_files(self):
        """
        Clear the RX file and write "ACK\n" to the TX file.
        proteus_bridge.py's handshake() polls TX looking for "ACK",
        so this must run before the bridge starts sending opcodes.
        """
        try:
            with open(VIRTUAL_SERIAL_RX, 'w') as f:
                f.write("")          # Clear any stale opcode
            with open(VIRTUAL_SERIAL_TX, 'w') as f:
                f.write("ACK\n")    # Tell bridge: LCD controller is ready
        except Exception as e:
            print(f"File I/O error: {e}")

    def _read_json(self):
        """
        Pull live data from the two JSON files main_pi writes.
        Updates self.current_conf (confidence 0.0–1.0),
        self.patient_name (e.g. "John M."), and self.sms_sent.
        Silently ignores missing or malformed JSON — the last
        known values are retained.
        """
        try:
            with open(STATUS_JSON, 'r') as f:
                status = json.load(f)
            self.current_conf = float(status.get("conf", 0.0))
        except Exception:
            pass
        try:
            with open(PATIENTS_JSON, 'r') as f:
                patients = json.load(f)
            for pdata in patients.values():
                if pdata.get("patient_name"):
                    parts = pdata["patient_name"].split()
                    # Abbreviate to "First L." to fit the 20-char row
                    self.patient_name = (
                        f"{parts[0]} {parts[1][0]}." if len(parts) >= 2 else parts[0]
                    )
                self.sms_sent = bool(pdata.get("sms_sent", False))
                break   # Only the first patient entry is used
        except Exception:
            pass

    def decode_message(self, msg):
        """
        Translate a short opcode (e.g. "NORM") to its full state name
        (e.g. "NORMAL"). The opcode map must match what
        proteus_bridge.py's VirtualSerialHandler.send_raw() compresses.
        If the opcode is not in the map, the raw message is returned
        unchanged (fallback for forward-compatibility).
        """
        opcode_map = {
            "BOOT":  "SYSTEM BOOTING",
            "INIT":  "INITIALIZING",
            "LOAD":  "LOADING",
            "PROC":  "PROCESSING",
            "TRAIN": "TRAINING",
            "RDY":   "READY",
            "NORM":  "NORMAL",
            "WARN":  "PRE-SEIZURE",
            "SEIZ":  "SEIZURE",
            "REST":  "RESTARTING",
            "HB":    "HEARTBEAT",
            "DSM":   "DATASET MODE",
            "SGM":   "SIGGEN MODE",
            "STOP":  "SYSTEM STOPPED",
        }
        return opcode_map.get(msg.strip(), msg)

    def worker(self):
        """
        Background thread: polls VIRTUAL_SERIAL_RX every 100 ms.
        - Skips HEARTBEAT opcodes (no display change needed).
        - Maps decoded state names to SystemState integer constants.
        - Falls back to WAITING_MODEL if no data arrives and the
          model has not yet connected (no_data_counter > 50 = ~5 s).
        """
        self.running    = True
        last_content    = ""      # Previous file content (detect changes)
        no_data_counter = 0       # Counts ticks with no new content

        while self.running:
            try:
                current_time = time()

                # Refresh JSON data approximately once per second (10 ticks × 100 ms)
                self._json_tick += 1
                if self._json_tick >= 10:
                    self._json_tick = 0
                    self._read_json()

                if os.path.exists(VIRTUAL_SERIAL_RX):
                    with open(VIRTUAL_SERIAL_RX, 'r') as f:
                        content = f.read().strip()

                    if content and content != last_content:
                        # New opcode arrived — process it
                        last_content    = content
                        no_data_counter = 0

                        lines = content.split('\n')
                        if lines:
                            msg = self.decode_message(lines[-1])   # Use most recent line

                            if msg == "HEARTBEAT":
                                # Heartbeat only means "bridge is alive"; don't change state
                                sleep(0.1)
                                continue

                            if not self.model_started:
                                self.model_started = True
                                print("Model connected")

                            self.last_message   = msg
                            self.message_count += 1
                            self.last_read_time  = current_time
                            print(f"[{self.message_count}] {msg}")

                            # Map decoded state name → SystemState integer constant
                            state_map = {
                                "SYSTEM BOOTING": SystemState.BOOT,
                                "INITIALIZING":   SystemState.INIT,
                                "LOADING":        SystemState.LOADING,
                                "PROCESSING":     SystemState.PROCESSING,
                                "TRAINING":       SystemState.TRAINING,
                                "READY":          SystemState.READY,
                                "NORMAL":         SystemState.NORMAL,
                                "PRE-SEIZURE":    SystemState.PRE_SEIZURE,
                                "SEIZURE":        SystemState.SEIZURE,
                                "RESTARTING":     SystemState.RESTARTING,
                                "DATASET MODE":   SystemState.DATASET_MODE,
                                "SIGGEN MODE":    SystemState.SIGGEN_MODE,
                                "SYSTEM STOPPED": SystemState.WAITING_MODEL,
                            }

                            if msg in state_map:
                                self.current_state = state_map[msg]
                    else:
                        # No new content
                        no_data_counter += 1
                        # After ~5 s with no data and model not yet seen, stay in WAITING_MODEL
                        if no_data_counter > 50 and not self.model_started:
                            self.current_state = SystemState.WAITING_MODEL

                sleep(0.1)   # Poll interval

            except Exception as e:
                print(f"Read error: {e}")
                sleep(1)     # Back off on repeated errors

    def start(self):
        """Launch the file-polling thread as a daemon."""
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

# ============================================================
# SECTION 12 — DISPLAY CONTROLLER
# Decides which of the render_state() branches to call and
# how often. Key responsibilities:
#
#   Refresh rate:  Only redraws every LCD_REFRESH_RATE seconds
#     (0.3 s) to avoid flickering.
#
#   Animated states: LOADING, PROCESSING, TRAINING, SEIZURE,
#     PRE_SEIZURE, NORMAL are redrawn every tick (for animated
#     dots and blinking).  Static states (BOOT, INIT, READY …)
#     are only redrawn when the state changes.
#
#   Escalation timing: Tracks how long we have been in
#     PRE_SEIZURE or SEIZURE (self._detection_state_since) to
#     know when to change row 4 to the escalation message.
#
#   Mode banner: DATASET_MODE and SIGGEN_MODE show a brief
#     full-screen banner for MODE_BANNER_DURATION seconds,
#     then revert to the last detection state (NORMAL / etc.).
# ============================================================
class DisplayController:
    def __init__(self, serial_rx):
        self.serial_rx         = serial_rx
        self.last_update       = 0          # Timestamp of last LCD write
        self.last_state        = None       # State last rendered (for change detection)
        self.blink_state       = False      # Toggle for LED/buzzer and row-1 blink
        self.loading_dots      = 0          # Animated dot counter for LOADING etc.

        self._last_detection_state   = SystemState.WAITING_MODEL
        self._detection_cur_state    = None
        self._detection_state_since  = 0.0   # Epoch when current detection state started

        self._mode_banner_shown_at   = 0
        self._mode_banner_state      = None

    def _is_detection_state(self, state):
        """Return True if state is one of the three live-monitoring states."""
        return state in (SystemState.NORMAL, SystemState.PRE_SEIZURE, SystemState.SEIZURE)

    def _state_duration(self):
        """Return how many seconds we have been in the current detection state."""
        return time() - self._detection_state_since

    def update(self):
        """
        Called in the main loop every 50 ms.
        Applies the refresh-rate gate (0.3 s), tracks state transitions,
        handles the mode-banner logic, then calls render_state().
        """
        current_time = time()
        if current_time - self.last_update < LCD_REFRESH_RATE:
            return   # Not time yet — skip this tick

        state = self.serial_rx.current_state

        # --- Escalation timer ---
        # Track entry time for NORMAL / PRE_SEIZURE / SEIZURE so we know
        # when to show the escalation message on row 4.
        if self._is_detection_state(state):
            if state != self._detection_cur_state:
                self._detection_cur_state   = state
                self._detection_state_since = current_time   # Reset timer on state change
            self._last_detection_state = state   # Remember last live state

        # --- Mode-switch banner ---
        # When DATASET_MODE or SIGGEN_MODE arrives, show a banner briefly
        # then silently fall back to the last detection state.
        if state in (SystemState.DATASET_MODE, SystemState.SIGGEN_MODE):
            if state != self._mode_banner_state:
                self._mode_banner_state    = state
                self._mode_banner_shown_at = current_time
            if current_time - self._mode_banner_shown_at < MODE_BANNER_DURATION:
                self.render_state(state)
                self.last_update = current_time
                return
            else:
                self._mode_banner_state = None
                state = self._last_detection_state   # Revert to last detection state

        # --- Render decision ---
        # Animated / alarm states are redrawn every tick.
        # Static states are only redrawn when they change.
        if state in (SystemState.LOADING, SystemState.PROCESSING,
                     SystemState.TRAINING, SystemState.SEIZURE,
                     SystemState.PRE_SEIZURE, SystemState.NORMAL):
            self.render_state(state)
        elif state != self.last_state:
            self.last_state = state
            self.render_state(state)

        self.last_update = current_time

    def render_state(self, state):
        """
        Write the four LCD rows and set GPIO outputs (LED, buzzer)
        for the given state. Mirrors the LCD output of
        main_pi_bios_v15.py exactly.

        Row layout for live-monitoring states (NORMAL / PRE_SEIZURE / SEIZURE):
          Row 1: "LIVE: <patient name>"
          Row 2: "STATE: <Normal|Pre-Seizure|Seizure>"
          Row 3: "CONF: <XX.X%>"
          Row 4: timestamp OR escalation message

        Escalation messages:
          PRE_SEIZURE lasting >= 12 s → row 4 = "TAKE MEDICATION"
          SEIZURE lasting >= 30 s     → row 4 = "DR NOTIFIED" (if SMS sent)
                                               or "CONTACT DR"
        """
        rx   = self.serial_rx
        conf = f"{rx.current_conf * 100:.1f}%"    # Format 0.83 → "83.0%"
        name = rx.patient_name                     # e.g. "John M."
        ts   = datetime.now().strftime('T:%H:%M:%S')  # e.g. "T:14:32:07"
        dur  = self._state_duration()              # Seconds in current detection state

        # ---- WAITING_MODEL ----
        # Shown before proteus_bridge.py sends its first opcode
        if state == SystemState.WAITING_MODEL:
            lcd_clear()
            lcd_string("LCD READY", LCD_LINE_1)
            lcd_string("Waiting Model..", LCD_LINE_2)
            lcd_string("", LCD_LINE_3)
            lcd_string("", LCD_LINE_4)
            GPIO.output(RED_LED, False)
            GPIO.output(BUZZER, False)

        # ---- SYSTEM BOOTING ----
        # Bridge sends "BOOT" for the first ~2 s
        elif state == SystemState.BOOT:
            lcd_clear()
            lcd_string("NeuroWatch v1.3", LCD_LINE_1)
            lcd_string("System Booting..", LCD_LINE_2)
            lcd_string("", LCD_LINE_3)
            lcd_string("Wait...", LCD_LINE_4)
            GPIO.output(RED_LED, False)
            GPIO.output(BUZZER, False)

        # ---- INITIALIZING ----
        # Bridge sends "INIT" for the next ~3 s
        elif state == SystemState.INIT:
            lcd_clear()
            lcd_string("INITIALIZING", LCD_LINE_1)
            lcd_string("Setting Up...", LCD_LINE_2)
            lcd_string("", LCD_LINE_3)
            lcd_string("", LCD_LINE_4)
            GPIO.output(RED_LED, False)
            GPIO.output(BUZZER, False)

        # ---- LOADING ----
        # Bridge holds "LOAD" until main_pi JSON appears.
        # Animated dots (1 to 4) cycle on the first row.
        elif state == SystemState.LOADING:
            dots = "." * ((self.loading_dots % 4) + 1)
            lcd_string("LOADING" + dots.ljust(13), LCD_LINE_1)
            lcd_string("Reading Data...", LCD_LINE_2)
            lcd_string("", LCD_LINE_3)
            lcd_string("", LCD_LINE_4)
            self.loading_dots += 1
            GPIO.output(RED_LED, False)
            GPIO.output(BUZZER, False)

        # ---- PROCESSING ----
        # Animated processing indicator
        elif state == SystemState.PROCESSING:
            dots = "." * ((self.loading_dots % 4) + 1)
            lcd_string("PROCESSING" + dots.ljust(10), LCD_LINE_1)
            lcd_string("Analyzing...", LCD_LINE_2)
            lcd_string("", LCD_LINE_3)
            lcd_string("", LCD_LINE_4)
            self.loading_dots += 1
            GPIO.output(RED_LED, False)
            GPIO.output(BUZZER, False)

        # ---- TRAINING ----
        # Animated training indicator (used if model is re-training on device)
        elif state == SystemState.TRAINING:
            dots = "." * ((self.loading_dots % 4) + 1)
            lcd_string("TRAINING" + dots.ljust(12), LCD_LINE_1)
            lcd_string("Learning Model", LCD_LINE_2)
            lcd_string("", LCD_LINE_3)
            lcd_string("", LCD_LINE_4)
            self.loading_dots += 1
            GPIO.output(RED_LED, False)
            GPIO.output(BUZZER, False)

        # ---- READY ----
        # Model loaded, about to start monitoring
        elif state == SystemState.READY:
            lcd_clear()
            lcd_string("SYSTEM READY", LCD_LINE_1)
            lcd_string("Starting Scan", LCD_LINE_2)
            lcd_string("", LCD_LINE_3)
            lcd_string("", LCD_LINE_4)
            GPIO.output(RED_LED, False)
            GPIO.output(BUZZER, False)

        # ---- NORMAL ----
        # Live monitoring, no seizure activity detected
        elif state == SystemState.NORMAL:
            lcd_string(f"LIVE: {name}", LCD_LINE_1)
            lcd_string("STATE: Normal", LCD_LINE_2)
            lcd_string(f"CONF: {conf}", LCD_LINE_3)
            lcd_string(ts, LCD_LINE_4)              # Timestamp on row 4
            GPIO.output(RED_LED, False)
            GPIO.output(BUZZER, False)

        # ---- PRE-SEIZURE ----
        # Warning state: LED and buzzer blink; after 12 s row 4 escalates
        elif state == SystemState.PRE_SEIZURE:
            row4 = "TAKE MEDICATION" if dur >= PRESEIZURE_LONG_SECONDS else ts
            lcd_string(f"LIVE: {name}", LCD_LINE_1)
            lcd_string("STATE: Pre-Seizure", LCD_LINE_2)
            lcd_string(f"CONF: {conf}", LCD_LINE_3)
            lcd_string(row4, LCD_LINE_4)
            GPIO.output(RED_LED, self.blink_state)     # Blink on each render tick
            GPIO.output(BUZZER, self.blink_state)
            self.blink_state = not self.blink_state    # Toggle for next tick

        # ---- SEIZURE ----
        # Alarm state: row 1 blinks on/off; LED+buzzer blink.
        # After 30 s row 4 shows "DR NOTIFIED" or "CONTACT DR".
        elif state == SystemState.SEIZURE:
            if dur >= SEIZURE_LONG_SECONDS:
                row4 = "DR NOTIFIED" if rx.sms_sent else "CONTACT DR"
            else:
                row4 = ts
            if self.blink_state:
                lcd_string(f"LIVE: {name}", LCD_LINE_1)
            else:
                lcd_string("", LCD_LINE_1)             # Blank row 1 on alternate ticks
            lcd_string("STATE: Seizure", LCD_LINE_2)
            lcd_string(f"CONF: {conf}", LCD_LINE_3)
            lcd_string(row4, LCD_LINE_4)
            self.blink_state = not self.blink_state
            GPIO.output(RED_LED, self.blink_state)
            GPIO.output(BUZZER, self.blink_state)

        # ---- RESTARTING ----
        # main_pi has stopped or crashed; bridge detected stale JSON
        elif state == SystemState.RESTARTING:
            lcd_clear()
            lcd_string("RESTARTING", LCD_LINE_1)
            lcd_string("Please Wait...", LCD_LINE_2)
            lcd_string("", LCD_LINE_3)
            lcd_string("", LCD_LINE_4)
            GPIO.output(RED_LED, False)
            GPIO.output(BUZZER, False)

        # ---- DATASET MODE ----
        # Brief banner when an alternative input mode is activated
        elif state == SystemState.DATASET_MODE:
            lcd_clear()
            lcd_string(">MODE SWITCHED<", LCD_LINE_1)
            lcd_string("  DATASET MODE  ", LCD_LINE_2)
            lcd_string("", LCD_LINE_3)
            lcd_string("", LCD_LINE_4)
            GPIO.output(RED_LED, False)
            GPIO.output(BUZZER, False)

        # ---- SIGGEN MODE ----
        # Brief banner when physical signal generator mode is activated
        elif state == SystemState.SIGGEN_MODE:
            lcd_clear()
            lcd_string(">MODE SWITCHED<", LCD_LINE_1)
            lcd_string(" PHYSICAL MODE  ", LCD_LINE_2)
            lcd_string("", LCD_LINE_3)
            lcd_string("", LCD_LINE_4)
            GPIO.output(RED_LED, False)
            GPIO.output(BUZZER, False)

# ============================================================
# SECTION 13 — MAIN LOOP
# Initialises all subsystems, shows a startup splash, then
# enters the tight 50 ms loop that drives the display.
#
# The loop calls display.update() which internally rate-limits
# LCD writes to LCD_REFRESH_RATE (0.3 s).  The 50 ms sleep
# gives the CPU breathing room without missing state changes.
#
# Ctrl+C shuts everything down cleanly:
#   - stops threads, turns off LEDs/buzzer, runs GPIO.cleanup()
# ============================================================
try:
    # Start heartbeat LED — blinks green throughout
    green_led = GreenLEDController()
    green_led.start()

    # Show splash screen while the rest of the system boots
    lcd_clear()
    lcd_string("NeuroWatch v1.3", LCD_LINE_1)
    lcd_string("LCD Controller", LCD_LINE_2)
    lcd_string("", LCD_LINE_3)
    lcd_string("Starting...", LCD_LINE_4)
    sleep(2)

    # Build the serial receiver (writes "ACK" handshake) and display controller
    serial_rx = VirtualSerialReceiver()
    display   = DisplayController(serial_rx)

    lcd_clear()
    lcd_string("Initialized!", LCD_LINE_1)
    sleep(1)

    # Start background file-polling thread
    serial_rx.start()

    # Show "waiting for model" screen until bridge sends first opcode
    lcd_clear()
    lcd_string("LCD READY", LCD_LINE_1)
    lcd_string("Waiting Model..", LCD_LINE_2)

    print("\n" + "=" * 60)
    print("LCD Controller Active")
    print("=" * 60)
    print("Green LED : continuous blink (system alive)")
    print("Red LED   : alarm during PRE-SEIZURE / SEIZURE")
    print("Buzzer    : alarm during PRE-SEIZURE / SEIZURE")
    print("=" * 60 + "\n")

    # Main loop — runs at ~20 Hz; display.update() internally rate-limits to ~3 Hz
    while True:
        display.update()
        sleep(0.05)

except KeyboardInterrupt:
    # Graceful shutdown on Ctrl+C
    print("\nSystem halted by user")
    lcd_clear()
    lcd_string("System Halted", LCD_LINE_1)
    sleep(1)
    green_led.stop()
    GPIO.output(RED_LED, False)
    GPIO.output(BUZZER, False)
    serial_rx.stop()
    GPIO.cleanup()   # Release all GPIO pins

except Exception as e:
    # Catch-all for unexpected errors — always clean up GPIO
    print(f"\nCritical error: {e}")
    import traceback
    traceback.print_exc()
    green_led.stop()
    GPIO.output(RED_LED, False)
    GPIO.output(BUZZER, False)
    serial_rx.stop()
    GPIO.cleanup()
