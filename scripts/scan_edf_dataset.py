# =============================================================================
# scan_edf_dataset.py  —  NeuroWatch EDF Dataset Scanner
#
# Scans your Raw_EDF_Files folder and prints a full breakdown:
#   - Total samples per patient
#   - Number of channels
#   - Recording duration
#   - Sampling frequency
#   - Class/label distribution (if annotations exist)
#
# Install dependencies first:
#   pip install mne pyedflib numpy
#
# Run:
#   python scan_edf_dataset.py
# =============================================================================

import os
import sys
import numpy as np
from pathlib import Path

# =============================================================================
# !! SET THIS TO YOUR Raw_EDF_Files FOLDER PATH !!
# =============================================================================
RAW_EDF_PATH = r"C:\Users\YourUsername\Documents\EEG_Seizure_Detection - To edit\NeuroWatch_Project\data\Test dataset\Raw_EDF_Files"   # <-- change this

# =============================================================================
# Try importing MNE (best for EDF), fallback to pyedflib
# =============================================================================
try:
    import mne
    mne.set_log_level("WARNING")
    USE_MNE = True
    print("✅ Using MNE to read EDF files\n")
except ImportError:
    USE_MNE = False
    try:
        import pyedflib
        print("✅ Using pyedflib to read EDF files\n")
    except ImportError:
        print("❌ Please install MNE or pyedflib:")
        print("   pip install mne   OR   pip install pyedflib")
        sys.exit(1)

# =============================================================================
# SCAN
# =============================================================================

def scan_edf_mne(fpath):
    """Read EDF with MNE and return metadata dict."""
    try:
        raw = mne.io.read_raw_edf(fpath, preload=False, verbose=False)
        info = raw.info
        n_channels   = len(raw.ch_names)
        sfreq        = info["sfreq"]
        n_times      = raw.n_times
        duration_sec = n_times / sfreq
        total_samples = n_channels * n_times

        # Check for annotations (seizure markers)
        annots = raw.annotations
        events = [(a["onset"], a["duration"], a["description"])
                  for a in annots] if len(annots) > 0 else []

        return {
            "file":           os.path.basename(fpath),
            "n_channels":     n_channels,
            "channel_names":  raw.ch_names,
            "sfreq_hz":       sfreq,
            "n_timepoints":   n_times,
            "duration_sec":   round(duration_sec, 2),
            "duration_min":   round(duration_sec / 60, 2),
            "total_samples":  total_samples,
            "annotations":    events,
            "n_seizure_events": len([e for e in events if "seiz" in e[2].lower()]),
            "error":          None,
        }
    except Exception as e:
        return {"file": os.path.basename(fpath), "error": str(e)}


def scan_edf_pyedf(fpath):
    """Read EDF with pyedflib and return metadata dict."""
    try:
        import pyedflib
        f = pyedflib.EdfReader(fpath)
        n_channels   = f.signals_in_file
        sfreq        = f.getSampleFrequency(0)
        n_times      = f.getNSamples()[0]
        duration_sec = n_times / sfreq
        ch_names     = f.getSignalLabels()
        f._close()
        return {
            "file":           os.path.basename(fpath),
            "n_channels":     n_channels,
            "channel_names":  ch_names,
            "sfreq_hz":       sfreq,
            "n_timepoints":   n_times,
            "duration_sec":   round(duration_sec, 2),
            "duration_min":   round(duration_sec / 60, 2),
            "total_samples":  n_channels * n_times,
            "annotations":    [],
            "n_seizure_events": 0,
            "error":          None,
        }
    except Exception as e:
        return {"file": os.path.basename(fpath), "error": str(e)}


def scan_folder(root_path):
    root = Path(root_path)
    if not root.exists():
        print(f"❌ Path not found: {root_path}")
        print("   Please update RAW_EDF_PATH at the top of this script.")
        sys.exit(1)

    # Find all EDF files recursively
    edf_files = sorted(list(root.rglob("*.edf")) + list(root.rglob("*.EDF")))
    if not edf_files:
        print(f"❌ No .edf files found in: {root_path}")
        sys.exit(1)

    print(f"📂 Found {len(edf_files)} EDF file(s) in: {root_path}")
    print("=" * 65)

    results     = []
    total_samp  = 0
    total_dur   = 0
    errors      = []

    for i, fpath in enumerate(edf_files):
        print(f"  [{i+1:02d}/{len(edf_files)}] Reading: {fpath.name} ...", end=" ", flush=True)
        r = scan_edf_mne(str(fpath)) if USE_MNE else scan_edf_pyedf(str(fpath))

        if r["error"]:
            print(f"❌ ERROR: {r['error']}")
            errors.append(r)
            continue

        print(f"✅  {r['n_channels']}ch  "
              f"{r['sfreq_hz']:.0f}Hz  "
              f"{r['duration_min']:.1f}min  "
              f"{r['total_samples']:,} samples")
        results.append(r)
        total_samp += r["total_samples"]
        total_dur  += r["duration_sec"]

    return results, errors, total_samp, total_dur


# =============================================================================
# REPORT
# =============================================================================

def print_report(results, errors, total_samp, total_dur):
    print("\n" + "=" * 65)
    print("📊  DATASET SUMMARY REPORT")
    print("=" * 65)

    if not results:
        print("No files read successfully.")
        return

    # Per-file table
    print(f"\n{'FILE':<35} {'CH':>4} {'Hz':>6} {'MIN':>7} {'SAMPLES':>14} {'SEIZ':>5}")
    print("-" * 72)
    for r in results:
        sz = r.get("n_seizure_events", 0)
        print(f"{r['file']:<35} {r['n_channels']:>4} "
              f"{r['sfreq_hz']:>6.0f} {r['duration_min']:>7.1f} "
              f"{r['total_samples']:>14,} {sz:>5}")

    # Totals
    print("-" * 72)
    print(f"\n  Total EDF files read     : {len(results)}")
    print(f"  Total recording duration : {total_dur/60:.1f} minutes  "
          f"({total_dur/3600:.2f} hours)")
    print(f"  Total samples (all ch)   : {total_samp:,}")
    print(f"  Total timepoints         : "
          f"{sum(r['n_timepoints'] for r in results):,}")

    # Channel info (from first file)
    r0 = results[0]
    print(f"\n  Channels ({r0['n_channels']}): {', '.join(r0['channel_names'][:10])}"
          + (" ..." if r0['n_channels'] > 10 else ""))
    print(f"  Sampling freq            : {r0['sfreq_hz']:.0f} Hz")

    # Seizure annotations
    total_sz = sum(r.get("n_seizure_events", 0) for r in results)
    if total_sz > 0:
        print(f"\n  Seizure events found     : {total_sz}")
        print("  Seizure annotation details:")
        for r in results:
            if r.get("n_seizure_events", 0) > 0:
                print(f"    {r['file']}: {r['n_seizure_events']} event(s)")
                for onset, dur, desc in r["annotations"]:
                    print(f"      onset={onset:.1f}s  dur={dur:.1f}s  label='{desc}'")
    else:
        print(f"\n  ⚠️  No seizure annotations found in EDF files.")
        print("     Annotations may be in a separate .txt or .csv file.")
        print("     Check your Documentation folder for label files.")

    # Segmentation estimate
    print("\n" + "=" * 65)
    print("📐  SEGMENTATION ESTIMATES (how many epochs you'll get)")
    print("=" * 65)
    total_tp = sum(r["n_timepoints"] for r in results)
    avg_sfreq = np.mean([r["sfreq_hz"] for r in results])
    for win_sec in [1, 2, 4, 8]:
        win_samples = int(win_sec * avg_sfreq)
        n_epochs    = total_tp // win_samples
        print(f"  {win_sec}s window  ({win_samples} samples/epoch)  →  "
              f"~{n_epochs:,} epochs total")

    # Errors
    if errors:
        print(f"\n⚠️  {len(errors)} file(s) failed to read:")
        for e in errors:
            print(f"   {e['file']}: {e['error']}")

    print("\n✅  Scan complete.")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Allow path override from command line:  python scan_edf_dataset.py "C:\my\path"
    if len(sys.argv) > 1:
        RAW_EDF_PATH = sys.argv[1]

    results, errors, total_samp, total_dur = scan_folder(RAW_EDF_PATH)
    print_report(results, errors, total_samp, total_dur)
