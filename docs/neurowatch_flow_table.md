# NeuroWatch Pipeline Table

| Stage | File / Component | Input | What Happens | Output | Connects To |
|---|---|---|---|---|---|
| 1 | `scripts/training/train_mendeley.py` | Mendeley `.npy` EEG files | Loads `x_train`, `y_train`, `x_test`, and `y_test` files | Combined EEG dataset | Stage 2 |
| 2 | `load_and_extract()` | Combined EEG dataset | Applies stratified split into train, validation, and test sets | Raw train/val/test EEG samples | Stage 3 |
| 3 | `extract_features()` | One EEG sample with 19 channels | Resamples EEG, filters it, extracts band powers and statistical features | 247-feature vector per sample | Stage 4 |
| 4 | `RFImportanceSelector` | 247 extracted features | Ranks features using Random Forest importance and keeps the top 75 | Selected feature matrix | Stage 5 |
| 5 | `train()` | Selected training features | Trains SVM and Random Forest models, then blends their probabilities | Hybrid SVM + RF ensemble | Stage 6 |
| 6 | `evaluate_split()` | Train, validation, and test features | Calculates accuracy, precision, recall, F1, specificity, seizure recall, and confusion matrix | Model performance metrics | Stage 7 |
| 7 | `save_models()` | Trained scaler, selector, SVM, RF, metrics | Saves all deployable model files and reports | `.pkl` files and `training_report.json` | Stage 8 |
| 8 | `models/...` | Saved model artifacts | Stores `scaler.pkl`, `selector.pkl`, `svm_model.pkl`, `rf_model.pkl`, and `ensemble_weights.pkl` | Reusable trained model package | Stage 9 |
| 9 | `src/main_pi_bios_v15.py` | Live/test EEG windows + saved model files | Loads model files, extracts the same features, scales/selects them, and predicts brain state | Normal, Pre-Seizure, or Seizure prediction | Stage 10 and Stage 11 |
| 10 | Raspberry Pi hardware | Predicted brain state | Updates LEDs, buzzer, LCD, and optional SMS alert | Physical patient/doctor alerts | End-user alerting |
| 11 | `json/` runtime files | Predicted state, confidence, patient info, history | Writes current monitoring results to JSON files | `neurowatch_status.json`, `neurowatch_patients.json`, `neurowatch_metrics.json`, history JSON | Stage 12 |
| 12 | `dashboard_v2.py` | Runtime JSON files | Displays clinician dashboard with ward overview, patient detail, map, logs, and metrics | Doctor-facing monitoring dashboard | Clinician |
| 13 | `patient_mobile_ui.py` | Patient and history JSON files | Displays simplified patient status, confidence, advice, and alert state | Patient-facing mobile UI | Patient |

## Simplified Flow

| From | To | Meaning |
|---|---|---|
| Training script | Model artifacts | Training creates reusable `.pkl` model files |
| Model artifacts | Pi monitoring engine | The live system loads the trained model |
| Pi monitoring engine | Hardware | Predictions trigger LEDs, buzzer, LCD, and SMS |
| Pi monitoring engine | Runtime JSON | Predictions are saved for apps to read |
| Runtime JSON | Clinician dashboard | Doctor sees live patient status and metrics |
| Runtime JSON | Patient mobile UI | Patient sees simplified current state |
