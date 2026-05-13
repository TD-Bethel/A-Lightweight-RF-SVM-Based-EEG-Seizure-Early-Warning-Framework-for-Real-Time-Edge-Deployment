# ---------------------------------------------------------------
# NeuroWatch — Proteus VSM Bridge
#
# PURPOSE: Runs on Windows alongside main_pi_bios_v15.py.
# Reads neurowatch_patients.json (written by main_pi) and
# translates the live patient state into short opcodes that
# the Proteus VSM LCD component can understand.  Communication
# is done through two plain text files that act as a virtual
# serial port (no real COM port hardware needed).
#
# How it fits in the system:
#   main_pi_bios_v15.py  →  writes neurowatch_patients.json
#   proteus_bridge.py    →  reads JSON, writes opcode to vsm_serial_rx.txt
#   Proteus VSM          →  reads vsm_serial_rx.txt, displays on LCD
#
# Run this ALONGSIDE main_pi_bios_v15.py on Windows.
# Proteus reads: C:\Users\thebe\AppData\Local\Temp\vsm_serial_rx.txt
# ---------------------------------------------------------------

# Standard Python libraries — no third-party dependencies
import os, time, json, threading, queue
import warnings
warnings.filterwarnings("ignore")   # Suppress non-critical deprecation notices

# ============================================================
# SECTION 1 — CONFIGURATION
# All file paths and timing knobs are in one place so you
# never have to search through the code to change them.
# ============================================================

# Virtual serial file paths — must match what Proteus VSM is configured to read/write
VIRTUAL_SERIAL_RX = r"C:\Users\thebe\AppData\Local\Temp\vsm_serial_rx.txt"  # Bridge → Proteus
VIRTUAL_SERIAL_TX = r"C:\Users\thebe\AppData\Local\Temp\vsm_serial_tx.txt"  # Proteus → Bridge (ACK)

# Locate json/ folder relative to this script (works from any working directory)
_HERE         = os.path.dirname(os.path.abspath(__file__))
PATIENTS_JSON = os.path.join(_HERE, "json", "neurowatch_patients.json")   # Live patient state
STATUS_JSON   = os.path.join(_HERE, "json", "neurowatch_status.json")     # Extra status (optional)

UPDATE_RATE      = 0.1    # Minimum seconds between consecutive sends (rate limiter)
POLL_RATE        = 0.3    # How often the bridge checks the JSON file (seconds)
HB_INTERVAL      = 1.0    # How often to send HEARTBEAT when nothing has changed (seconds)
JSON_STALE_SECS  = 5.0    # If JSON has not been updated in this many seconds, treat as offline
watchdog_timeout = 2.0    # If no send in this many seconds, re-initialise the serial files

# ============================================================
# SECTION 2 — VIRTUAL SERIAL HANDLER
# Encapsulates all file I/O for the virtual serial connection.
# The Proteus VSM component reads VIRTUAL_SERIAL_RX (RX from
# its perspective); this bridge writes to that file.
# The VSM writes its replies to VIRTUAL_SERIAL_TX; this bridge
# reads that file to detect the ACK handshake.
#
# A background worker thread drains an outgoing queue at
# UPDATE_RATE pace so we never flood Proteus with rapid writes.
# If the queue is full, send_raw() is called directly (bypasses queue).
# ============================================================
class VirtualSerialHandler:
    def __init__(self):
        self.tx_queue       = queue.Queue(maxsize=10)  # Outgoing message buffer
        self.last_heartbeat = time.time()               # Timestamp of last successful send
        self.running        = False                     # Worker thread control flag
        self.init_files()

    def init_files(self):
        """
        Clear both virtual serial files and run the handshake.
        Called at startup and by the watchdog if the connection goes stale.
        """
        try:
            with open(VIRTUAL_SERIAL_RX, 'w') as f:
                f.write("")    # Clear stale opcode
            with open(VIRTUAL_SERIAL_TX, 'w') as f:
                f.write("")    # Clear stale ACK
            print("Virtual serial initialized")
            self.handshake()
        except Exception as e:
            print(f"File init error: {e}")

    def handshake(self):
        """
        Send "READY" and wait up to 5 s for the Proteus LCD component
        to reply with "ACK" in vsm_serial_tx.txt.
        lcd_controller_pi.py writes "ACK" during its own init_files().
        If no ACK arrives in time, the bridge continues anyway (Proteus
        may not be connected, which is fine for JSON-only monitoring).
        Returns True on success, False on timeout.
        """
        self.send_raw("READY")
        timeout = time.time() + 5
        while time.time() < timeout:
            try:
                with open(VIRTUAL_SERIAL_TX, 'r') as f:
                    response = f.read().strip()
                if "ACK" in response:
                    print("Handshake established")
                    self.last_heartbeat = time.time()
                    return True
            except Exception:
                pass
            time.sleep(0.2)
        print("Handshake timeout, continuing anyway...")
        return False

    def send_raw(self, msg):
        """
        Compress msg to its short opcode and overwrite vsm_serial_rx.txt.
        The opcode_map here must mirror the decode map in
        lcd_controller_pi.py's VirtualSerialReceiver.decode_message().
        If msg is not in the map, the first 16 characters of msg are used
        as a fallback (forward-compatibility for future states).
        """
        try:
            # Short opcode map — Proteus has a small serial buffer,
            # so we send compressed 4-char codes instead of full state names
            opcode_map = {
                "SYSTEM BOOTING": "BOOT",
                "INITIALIZING":   "INIT",
                "LOADING":        "LOAD",
                "PROCESSING":     "PROC",
                "TRAINING":       "TRAIN",
                "READY":          "RDY",
                "NORMAL":         "NORM",
                "PRE-SEIZURE":    "WARN",
                "SEIZURE":        "SEIZ",
                "RESTARTING":     "REST",
                "HEARTBEAT":      "HB",
                "SYSTEM STOPPED": "STOP",
            }
            compressed = opcode_map.get(msg, msg[:16])    # Use short code or truncate
            with open(VIRTUAL_SERIAL_RX, 'w') as f:
                f.write(compressed + "\n")
            self.last_heartbeat = time.time()
            print(f"Sent: {msg} ({compressed})")
        except Exception as e:
            print(f"Send error: {e}")

    def send(self, msg):
        """
        Queue a message for rate-limited delivery by the worker thread.
        If the queue is full (10 pending messages), send immediately via
        send_raw() to avoid blocking the caller.
        Use this instead of send_raw() when you don't need guaranteed timing.
        """
        if not self.tx_queue.full():
            self.tx_queue.put(msg)
        else:
            self.send_raw(msg)    # Bypass queue when full

    def worker(self):
        """
        Background thread: drains the outgoing queue at UPDATE_RATE pace.
        Also sends periodic HEARTBEAT messages when the queue is empty and
        the connection has been silent for longer than HB_INTERVAL.
        This keeps the Proteus LCD alive during long quiet periods.
        """
        self.running = True
        while self.running:
            try:
                msg = self.tx_queue.get(timeout=0.5)   # Wait up to 500 ms for a message
                self.send_raw(msg)
                time.sleep(UPDATE_RATE)                 # Rate limiter between sends
            except queue.Empty:
                # No messages — send heartbeat if overdue
                if time.time() - self.last_heartbeat > HB_INTERVAL:
                    self.send_raw("HEARTBEAT")

    def check_watchdog(self):
        """
        Called in the main loop every POLL_RATE seconds.
        If no successful send has happened in watchdog_timeout seconds,
        re-initialise the serial files and attempt handshake again.
        This recovers from Proteus VSM restarts or file system errors.
        """
        if time.time() - self.last_heartbeat > watchdog_timeout:
            print("Watchdog timeout — reinitializing...")
            self.init_files()

    def start(self):
        """Launch the worker thread as a daemon (auto-stops when main exits)."""
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False


# ============================================================
# SECTION 3 — PROTEUS BRIDGE
# The core logic class. Polls neurowatch_patients.json and
# decides which opcode to send to Proteus based on the current
# phase of the system:
#
#  Phase 1 — Startup (main_pi not yet running / JSON missing):
#    Cycles through a startup animation using _STARTUP_PHASES:
#      0–2 s   → "SYSTEM BOOTING"
#      2–5 s   → "INITIALIZING"
#      5 s+    → "LOADING" (held until JSON appears)
#
#  Phase 2 — Live monitoring (JSON fresh):
#    Reads patient state and sends NORMAL / PRE-SEIZURE / SEIZURE.
#
#  Phase 3 — Recovery (JSON goes stale after being fresh):
#    Sends "RESTARTING" to tell the LCD that main_pi stopped.
#
# Heartbeat logic (in tick()):
#    Even if the state hasn't changed, we resend every HB_INTERVAL
#    seconds so the Proteus LCD doesn't think the bridge crashed.
# ============================================================
class ProteusBridge:

    # Startup phase sequence while waiting for main_pi to produce JSON.
    # Each tuple is (duration_seconds, opcode_to_send).
    # The last entry has a large duration so "LOADING" is held
    # indefinitely until the JSON file appears.
    _STARTUP_PHASES = [
        (2.0,  "SYSTEM BOOTING"),
        (3.0,  "INITIALIZING"),
        (99.0, "LOADING"),        # Held until JSON appears — effectively infinite
    ]

    def __init__(self, serial):
        self.serial        = serial
        self._last_opcode  = None          # Last opcode actually sent (for change detection)
        self._last_send    = 0.0           # Timestamp of last send (for heartbeat gate)
        self._start_time   = time.time()   # Bridge launch timestamp
        self._model_seen   = False         # True once we have had at least one fresh JSON read

    # ---- Internal helpers ----

    def _elapsed(self):
        """Return seconds since the bridge started."""
        return time.time() - self._start_time

    def _startup_opcode(self):
        """
        Walk through _STARTUP_PHASES using elapsed time to determine
        which startup phase we are currently in.
        Returns the opcode for the matching phase.
        Example: at 2.5 s elapsed → "INITIALIZING"
        """
        elapsed    = self._elapsed()
        cumulative = 0
        for duration, opcode in self._STARTUP_PHASES:
            cumulative += duration
            if elapsed < cumulative:
                return opcode
        return "LOADING"    # Safety fallback (should not be reached)

    def _read_patient_state(self):
        """
        Read neurowatch_patients.json and return the state string for the
        first patient entry, e.g. "Normal", "Pre-Seizure", or "Seizure".

        Returns None if:
          - The file does not exist yet (main_pi not started)
          - The file has not been modified in JSON_STALE_SECS seconds
            (main_pi crashed or was stopped)
          - The file cannot be parsed (write-in-progress race condition)

        The staleness check (mtime comparison) is what distinguishes
        "startup" (never had data) from "restarting" (had data but lost it).
        """
        try:
            mtime = os.path.getmtime(PATIENTS_JSON)
            if time.time() - mtime > JSON_STALE_SECS:
                return None    # File is stale — main_pi not writing
            with open(PATIENTS_JSON, 'r') as f:
                patients = json.load(f)
            for pdata in patients.values():
                return pdata.get("state", "Normal")    # Return first patient's state
        except Exception:
            return None    # File missing or unreadable

    # ---- Main tick ----

    def tick(self):
        """
        Called every POLL_RATE seconds by run().
        Determines the correct opcode for the current situation and
        sends it if either the state has changed OR a heartbeat is due.

        Decision tree:
          state is None AND never saw model → startup animation opcode
          state is None AND previously saw model → "RESTARTING"
          state is valid → map patient state to opcode (NORMAL/PRE-SEIZURE/SEIZURE)
        """
        state = self._read_patient_state()

        if state is None:
            if self._model_seen:
                # We had a good JSON read before, but now it is gone/stale
                # → main_pi stopped or is restarting
                opcode = "RESTARTING"
            else:
                # Still in startup — cycle through boot animation
                opcode = self._startup_opcode()
        else:
            self._model_seen = True    # Mark that we have successfully read at least once
            # Map the patient state string to the opcode the LCD understands
            opcode = {
                "Normal":      "NORMAL",
                "Pre-Seizure": "PRE-SEIZURE",
                "Seizure":     "SEIZURE",
            }.get(state, "NORMAL")    # Fallback to NORMAL for unknown state strings

        now = time.time()
        state_changed = opcode != self._last_opcode           # True if state is new
        heartbeat_due = now - self._last_send >= HB_INTERVAL  # True if time to resend

        # Only send when something changed OR heartbeat is overdue
        if state_changed or heartbeat_due:
            self.serial.send_raw(opcode)
            self._last_opcode = opcode
            self._last_send   = now

    def run(self):
        """
        Entry point for the bridge.  Prints a startup banner then
        enters the infinite poll loop: check watchdog → tick → sleep.
        Ctrl+C is caught in the top-level try/except below.
        """
        print("\n" + "=" * 55)
        print("NeuroWatch — Proteus Bridge  (polling JSON @ %.1fs)" % POLL_RATE)
        print("Watching:", PATIENTS_JSON)
        print("Writing :", VIRTUAL_SERIAL_RX)
        print("=" * 55 + "\n")

        while True:
            self.serial.check_watchdog()   # Re-init serial files if connection went stale
            self.tick()                    # Read JSON and send opcode if needed
            time.sleep(POLL_RATE)          # Wait before next poll


# ============================================================
# SECTION 4 — ENTRY POINT
# Instantiates and starts the serial handler (worker thread)
# then runs the bridge loop. Handles Ctrl+C cleanly by sending
# "SYSTEM STOPPED" so the Proteus LCD shows a shutdown state.
# Any other unexpected exception is caught, traceback printed,
# and the serial thread stopped before exit.
# ============================================================
try:
    serial = VirtualSerialHandler()
    serial.start()    # Start background queue-drain worker thread

    bridge = ProteusBridge(serial)
    bridge.run()      # Blocks here until Ctrl+C or crash

except KeyboardInterrupt:
    # User pressed Ctrl+C — inform Proteus and exit gracefully
    serial.send_raw("SYSTEM STOPPED")
    serial.stop()
    print("\nBridge stopped by user")

except Exception as e:
    # Unexpected crash — print details and clean up
    print(f"\nCritical error: {e}")
    import traceback
    traceback.print_exc()
    serial.stop()
