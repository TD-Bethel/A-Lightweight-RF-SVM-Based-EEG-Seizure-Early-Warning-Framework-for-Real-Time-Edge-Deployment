# NeuroWatch

> Remote EEG seizure monitoring system — Raspberry Pi edge inference + clinician dashboard + patient mobile UI.

NeuroWatch detects brain states (Normal, Pre-Seizure, Seizure) from multi-channel EEG data using a hybrid SVM + Random Forest ensemble. The system runs on a Raspberry Pi at the patient's location, writes predictions to JSON, and makes them visible to a remote clinician through a Streamlit dashboard. An optional patient-facing mobile UI shows a simplified status view.

---

## Project Structure

```
NeuroWatch_Project/
├── src/
│   ├── main_pi_bios_v15.py       # Live monitoring engine (Raspberry Pi / desktop)
│   ├── dashboard_v2.py           # Clinician dashboard (Streamlit, port 8501)
│   ├── patient_mobile_ui.py      # Patient mobile UI (Streamlit, port 8502)
│   ├── patient_watch_ui_clean.py # Patient desktop UI (Tkinter)
│   ├── patients.py               # Patient registry (P001 live, P002-P007 demo)
│   ├── sms_notifier.py           # Twilio SMS alerts
│   └── neurowatch_metrics.py     # Metrics pipeline
│
├── scripts/
│   ├── training/
│   │   ├── train_mendeley.py     # Train on Mendeley 19-channel EEG dataset
│   │   ├── train_bonn.py         # Train on Bonn University dataset
│   │   └── train_and_export.py   # Bonn training + export
│   ├── inference/
│   │   ├── quick_viz.py          # Lightweight offline model visualizer (no Streamlit)
│   │   └── run_prediction.py     # Single-sample prediction helper
│   ├── evaluation/
│   │   ├── eval_preictal_backup.py  # Full test-set evaluation with confusion matrix
│   │   └── cross_dataset_eval.py   # Cross-dataset evaluation
│   ├── preprocessing/
│   │   └── preprocess_preictal.py  # Extract pre-ictal windows from Mendeley data
│   ├── plotting/
│   │   ├── plot_results.py       # Training result plots from saved JSON
│   │   ├── plot_test_samples.py  # Visualise Bonn test-split samples
│   │   ├── plot_npy_eeg.py       # Plot raw Mendeley .npy EEG
│   │   └── plot_bonn_eeg.py      # Plot raw Bonn .edf EEG
│   └── utils/
│       ├── generate_readme.py    # Auto-generate README
│       └── scan_edf_dataset.py   # Scan and summarise EDF dataset
│
├── models/
│   ├── MODELS_FS75/              # Current model (75 RF-selected features) ← active
│   └── MODELS_PREICTAL_BACKUP/   # Backup (418 features, older pipeline)
│
├── data/
│   ├── Mendelay dataset/
│   │   ├── Npy_files_preictal/   # Pre-processed .npy files used for training
│   │   └── Raw_EDF_Files/        # Raw EDF (live inference only, no labels)
│   └── Bonn University Dataset/
│
├── json/                         # Runtime outputs (patients, metrics, history)
├── assets/                       # Brain 3D model (.glb) and other UI assets
├── neurowatch_selector.py        # Shared RFImportanceSelector class (required for pickle)
├── dashboard_v2.py               # Root-level dashboard launcher
├── patients.py                   # Root-level patient registry
└── requirements.txt
```

---

## Model — MODELS_FS75 (Active)

| Property | Value |
|---|---|
| Dataset | Mendeley Epileptic EEG (Wassim Nasreddine, AUB) |
| Channels | 19 |
| Raw features | 247 (19 channels × 13: 5 band powers + 8 stats) |
| Feature selection | RF importance — top 75 of 247 |
| Classes | Normal · Pre-Seizure · Seizure |
| Model | SVM (95%) + RF (5%) ensemble, auto-tuned weights |
| SVM class weights | `{Normal: 1.0, Pre-Seizure: 1.8, Seizure: 1.0}` |
| Normal threshold | 0.38 (tuned on validation set) |
| Test accuracy | **76.1%** |
| Seizure recall | **85.1%** |
| Overfitting gap | 5.9% (train 82.3% → val 76.5%) |

### Feature Extraction (per EEG window)

For each of the 19 channels:
- Resample 500 Hz → 128 Hz
- Bandpass filter 0.5–45 Hz (Butterworth order 4)
- **Band powers** (Welch PSD): delta 1–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz, gamma 30–45 Hz — normalised as relative proportions
- **Hjorth parameters**: activity, mobility, complexity
- **Statistical**: skewness, kurtosis, RMS, zero-crossing rate, spectral entropy

247 features → RF importance bootstrap (300 trees) selects top 75 → StandardScaler → SVM + RF ensemble → Normal threshold override.

---

## Quick Start

### 1. Install dependencies

```bash
pip install imbalanced-learn joblib matplotlib mne numpy pandas plotly pyedflib scikit-learn scipy seaborn streamlit twilio
```

### 2. Train the model

```bash
# Standard training (no GUI)
python scripts/training/train_mendeley.py \
  --data "data/Mendelay dataset/Npy_files_preictal" \
  --out  "models/MODELS_FS75"

# Training with live visualization (6-panel dashboard + saves PNG images)
python scripts/training/train_mendeley_visual.py \
  --data "data/Mendelay dataset/Npy_files_preictal" \
  --out  "models/MODELS_FS75"
```

### 2b. Train with PyTorch CNN + TensorBoard

```bash
# Terminal 1 — start training (saves logs to runs/)
python scripts/training/train_cnn_tensorboard.py

# Terminal 2 — open TensorBoard in browser while training runs
tensorboard --logdir runs/
```
Then visit http://localhost:6006 to see live loss/accuracy curves, confusion matrix, weight histograms, and the model graph.

### 3. Run everything

Open **three terminals** from the project root:

```bash
# Terminal 1 — live monitoring engine
python src/main_pi_bios_v15.py \
  --models "models/RF60+SVM40_C2" \
  --data   "data/Mendelay dataset/Npy_files_preictal"

# Terminal 2 — clinician dashboard
streamlit run src/dashboard_v2.py --server.port 8505

# Terminal 3 — patient mobile UI
streamlit run src/patient_mobile_ui.py --server.port 8502
```

| UI | URL |
|---|---|
| Clinician dashboard | http://localhost:8505
| Patient mobile UI | http://localhost:8502 |

---

## Useful Commands

### Evaluate the model (full metrics + confusion matrix)
```bash
python scripts/evaluation/eval_preictal_backup.py
```

### Lightweight offline visualizer (no Streamlit needed)
```bash
python scripts/inference/quick_viz.py
# Shows 4-panel matplotlib figure: prediction timeline, ground truth, confidence, per-class accuracy
```

### View training result plots
```bash
python scripts/plotting/plot_results.py
```

### Browse preprocessed EEG windows (after pre-ictal extraction)
```bash
# All classes — slider browses all ~11,000 samples, background colour shows class
python scripts/plotting/plot_preprocessed_eeg.py

# Start at a specific sample index
python scripts/plotting/plot_preprocessed_eeg.py --start 500

# One split only
python scripts/plotting/plot_preprocessed_eeg.py --split train
python scripts/plotting/plot_preprocessed_eeg.py --split test
```
All 19 channels on one single graph with vertical offsets. Background: green = Normal, orange = Pre-ictal, red = Ictal. Use ◀ ▶ or slider to browse. Save PNG button saves current sample.

---

### Visualise raw EDF files before preprocessing
```bash
# List all EDF files and seizure counts (shows which are found/missing)
python scripts/plotting/plot_raw_edf.py --list

# Interactive picker — prompts for patient and record number
python scripts/plotting/plot_raw_edf.py

# Direct — e.g. Patient 10, Record 1
python scripts/plotting/plot_raw_edf.py --patient 10 --record 1
```
Shows all 19 channels stacked with coloured bands: green = Normal, orange = Pre-ictal, red = Ictal.

### Patient desktop UI (Tkinter window)
```bash
python src/patient_watch_ui_clean.py
```

### Remote access via ngrok (open patient UI on phone)
```bash
streamlit run src/patient_mobile_ui.py --server.address 0.0.0.0 --server.port 8502
ngrok http 8502
```

---

## Runtime JSON Outputs

The monitoring engine writes to `json/` every prediction cycle:

| File | Contents |
|---|---|
| `neurowatch_patients.json` | Current state, confidence, GPS, ward/bed per patient |
| `neurowatch_metrics.json` | Rolling accuracy, seizure count, uptime |
| `neurowatch_status.json` | Engine phase (DETECTING / LOADING / ERROR) |

The dashboard reads these files and refreshes every 30 seconds by default. Use the **⏸ Pause** button in the sidebar to freeze the display while observing.

---

## Hardware (Raspberry Pi)

The monitoring engine (`main_pi_bios_v15.py`) auto-detects Raspberry Pi GPIO and supports:
- LCD display (RPLCD)
- LED / buzzer alerts via GPIO
- Twilio SMS on seizure detection (configure in `.env`)

Set `RASPBERRY_PI_SETUP.txt` for wiring details.

---

## Notes

- `neurowatch_selector.py` must stay in the project root — all scripts import it so joblib can resolve the `RFImportanceSelector` class when loading `selector.pkl`.
- If you retrain, always point `--out` to the same model directory to overwrite the old `selector.pkl`.
- For live Raspberry Pi deployment, copy the full project folder including `neurowatch_selector.py` and `models/MODELS_FS75/`.
- SMS credentials should be stored in a `.env` file — never commit them to git.
