import os
import numpy as np
import joblib
import mne
import matplotlib.pyplot as plt
from scipy.signal import resample, butter, filtfilt, welch

# ======================== CONFIGURATION ========================
TARGET_FS = 128
WINDOW_SEC = 2
MODEL_PATH = "models"
CLASSES = ["Normal", "Pre-Seizure", "Seizure"]
COLORS = ["#2ecc71", "#f1c40f", "#e74c3c"]
BONN_FOLDERS = {"O", "N", "S", "F", "Z"}


class UniversalPredictor:
    def __init__(self):
        self.scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
        self.svm = joblib.load(os.path.join(MODEL_PATH, "svm_model.pkl"))
        self.rf = joblib.load(os.path.join(MODEL_PATH, "rf_model.pkl"))

    def predict(self, chunk, fs):
        try:
            if fs != TARGET_FS:
                chunk = resample(chunk, int(len(chunk) * TARGET_FS / fs))

            nyq = 0.5 * TARGET_FS
            b, a = butter(4, [0.5 / nyq, 45 / nyq], btype="band")
            filt = filtfilt(b, a, chunk)

            f, psd = welch(filt, fs=TARGET_FS, nperseg=128)
            feats = [
                np.mean(psd[(f >= 1) & (f <= 4)]),
                np.mean(psd[(f >= 4) & (f <= 8)]),
                np.mean(psd[(f >= 8) & (f <= 13)]),
                np.mean(psd[(f >= 13) & (f <= 30)]),
            ]
            feats = np.array(feats).reshape(1, -1)

            xs = self.scaler.transform(feats)
            conf = (0.4 * self.svm.predict_proba(xs)[0]) + (0.6 * self.rf.predict_proba(feats)[0])
            return int(np.argmax(conf)), conf, filt
        except Exception:
            return None, None, None


def find_eeg_files(dir_path):
    """Recursively collect valid EEG files from a dataset root or subfolder."""
    eeg_files = []
    for root, _, files in os.walk(dir_path):
        folder_name = os.path.basename(root)
        for filename in sorted(files):
            lower = filename.lower()
            file_path = os.path.join(root, filename)

            if lower.endswith(".edf"):
                eeg_files.append(file_path)
            elif lower.endswith(".txt") and folder_name in BONN_FOLDERS:
                eeg_files.append(file_path)

    return eeg_files


def load_signal(file_path):
    """Load a single EEG signal from EDF or TXT, returning (data, fs)."""
    if file_path.lower().endswith(".edf"):
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        return raw.get_data()[0], raw.info["sfreq"]

    data = np.asarray(np.loadtxt(file_path)).squeeze()
    if data.ndim != 1 or data.size < TARGET_FS:
        raise ValueError(f"invalid EEG shape {data.shape}")
    return data, 128.0


def infer_expected_label(file_path):
    """Infer the expected Bonn label from the parent folder when available."""
    folder_name = os.path.basename(os.path.dirname(file_path))
    if folder_name in BONN_FOLDERS:
        return folder_name
    return "Unknown"


# ======================== VISUALIZATION SETUP ========================
def setup_dashboard():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.canvas.manager.set_window_title("EEG Seizure Detection Live Monitor")

    line, = ax1.plot([], [], lw=1.5, color="#3498db")
    ax1.set_title("Current EEG Chunk (Filtered)")
    ax1.set_ylim(-150, 150)
    ax1.set_xlim(0, WINDOW_SEC)
    ax1.grid(True, alpha=0.3)

    bars = ax2.bar(CLASSES, [0, 0, 0], color=COLORS)
    ax2.set_ylim(0, 1)
    ax2.set_title("Model Confidence (%)")

    plt.tight_layout()
    return fig, ax1, ax2, line, bars


# ======================== MAIN PROCESSING ========================
def process_with_viz(dir_path):
    predictor = UniversalPredictor()
    fig, ax1, ax2, line, bars = setup_dashboard()

    files = find_eeg_files(dir_path)
    if not files:
        print("No valid EEG files found.")
        return

    print(f"Found {len(files)} EEG file(s).")

    for file_path in files:
        filename = os.path.relpath(file_path, dir_path)
        expected = infer_expected_label(file_path)
        print(f"Processing: {filename}")
        print(f"Expected source folder: {expected}")

        try:
            data, fs = load_signal(file_path)
            samples_per_win = int(WINDOW_SEC * fs)

            for i in range(len(data) // samples_per_win):
                chunk = data[i * samples_per_win:(i + 1) * samples_per_win]
                idx, conf, filt_chunk = predictor.predict(chunk, fs)

                if idx is not None:
                    predicted = CLASSES[idx]
                    confidence = float(np.max(conf))
                    print(
                        f"  Window {i + 1}: predicted={predicted} "
                        f"confidence={confidence * 100:.1f}%"
                    )

                    t = np.linspace(0, WINDOW_SEC, len(filt_chunk))
                    line.set_data(t, filt_chunk)
                    ax1.set_ylim(np.min(filt_chunk) - 10, np.max(filt_chunk) + 10)

                    for bar, val in zip(bars, conf):
                        bar.set_height(val)

                    ax2.set_title(f"Prediction: {predicted} ({confidence * 100:.1f}%)")

                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.1)

        except Exception as e:
            print(f"Skipping {filename}: {e}")


if __name__ == "__main__":
    folder_path = input("Enter path to your EEG folder: ").strip('"')
    if os.path.isdir(folder_path):
        process_with_viz(folder_path)
    else:
        print("Invalid directory.")
