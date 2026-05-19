"""
log_model_to_tensorboard.py  --  Push an already-trained model's metrics
                                 into a TensorBoard logdir.

Reads training_report.json + the diagnostic PNG figures saved inside a model
directory (e.g. models/RF60+SVM40_C2) and writes them as TensorBoard scalar
and image summaries so you can browse them in the standard tensorboard UI.

Usage:
    python scripts/utils/log_model_to_tensorboard.py
    python scripts/utils/log_model_to_tensorboard.py --model-dir models/RF60+SVM40_C2
    python scripts/utils/log_model_to_tensorboard.py --all

View later with:
    tensorboard --logdir runs/
    (then open http://localhost:6006)
"""

import argparse
import json
import os
import sys

import numpy as np
from tensorboardX import SummaryWriter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models")
RUNS_DIR   = os.path.join(ROOT, "runs")


def _load_png_as_chw(path: str):
    """Read a PNG into a (3, H, W) numpy array for SummaryWriter.add_image()."""
    try:
        from PIL import Image
    except ImportError:
        return None
    img = Image.open(path).convert("RGB")
    arr = np.array(img)               # (H, W, 3)
    return np.transpose(arr, (2, 0, 1))  # (3, H, W)


def log_one(model_dir: str) -> bool:
    report_path = os.path.join(model_dir, "training_report.json")
    if not os.path.exists(report_path):
        print(f"  SKIP  no training_report.json in {model_dir}")
        return False

    with open(report_path, "r", encoding="utf-8") as f:
        r = json.load(f)

    run_name = os.path.basename(os.path.normpath(model_dir))
    log_dir  = os.path.join(RUNS_DIR, run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(logdir=log_dir)
    print(f"  -> writing TensorBoard logs to {log_dir}")

    # Hyperparameters as text (TB doesn't have native param tracking like MLflow)
    params_md = "\n".join([
        f"- **model**: {r.get('model')}",
        f"- **n_features**: {r.get('n_features')}",
        f"- **n_selected**: {r.get('n_selected')}",
        f"- **n_channels**: {r.get('n_channels')}",
        f"- **feature_selector**: {r.get('feature_selector')}",
        f"- **normalisation**: {r.get('normalisation')}",
        f"- **svm_class_weight**: {r.get('svm_class_weight')}",
        f"- **normal_threshold**: {r.get('normal_threshold')}",
        f"- **split**: {r.get('split')}",
        f"- **trained_on**: {r.get('trained_on')}",
    ])
    writer.add_text("hyperparameters", params_md)

    # Scalar metrics — one point per split, plotted on a 3-step x-axis so
    # train / val / test appear as a connected curve.
    split_order = [("training", 0, "train"),
                   ("validation", 1, "val"),
                   ("test", 2, "test")]
    for split_name, step, _short in split_order:
        m = r.get(split_name, {})
        if not m:
            continue
        if "accuracy" in m:
            writer.add_scalar("split_curve/accuracy",       float(m["accuracy"]),       step)
        if "f1" in m:
            writer.add_scalar("split_curve/f1",             float(m["f1"]),             step)
        if "seizure_recall" in m:
            writer.add_scalar("split_curve/seizure_recall", float(m["seizure_recall"]), step)

    # Same metrics as single named tags (easier to compare across runs)
    for split_name, _, short in split_order:
        m = r.get(split_name, {})
        if "accuracy" in m:       writer.add_scalar(f"{short}/accuracy",       float(m["accuracy"]),       0)
        if "f1" in m:             writer.add_scalar(f"{short}/f1",             float(m["f1"]),             0)
        if "seizure_recall" in m: writer.add_scalar(f"{short}/seizure_recall", float(m["seizure_recall"]), 0)
        for cls_name, cls_m in (m.get("per_class") or {}).items():
            tag = cls_name.lower().replace("-", "_").replace(" ", "_")
            for k, v in cls_m.items():
                writer.add_scalar(f"{short}_per_class/{tag}_{k}", float(v), 0)

    # Derived
    try:
        gap = float(r["training"]["accuracy"]) - float(r["validation"]["accuracy"])
        writer.add_scalar("overfitting_gap", gap, 0)
    except KeyError:
        pass

    # Diagnostic figures as images
    for fname in os.listdir(model_dir):
        if not fname.endswith(".png"):
            continue
        img = _load_png_as_chw(os.path.join(model_dir, fname))
        if img is not None:
            writer.add_image(f"figures/{os.path.splitext(fname)[0]}", img, 0)

    writer.close()
    print(f"     OK")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(MODELS_DIR, "RF60+SVM40_C2"),
                    help="Model directory to log (default: RF60+SVM40_C2)")
    ap.add_argument("--all", action="store_true",
                    help="Log every subdirectory under models/ with a training_report.json")
    args = ap.parse_args()

    if args.all:
        if not os.path.isdir(MODELS_DIR):
            print(f"ERROR: models directory not found: {MODELS_DIR}")
            sys.exit(1)
        dirs = [os.path.join(MODELS_DIR, d) for d in sorted(os.listdir(MODELS_DIR))
                if os.path.isdir(os.path.join(MODELS_DIR, d))]
    else:
        dirs = [args.model_dir]

    n = 0
    for d in dirs:
        if log_one(d):
            n += 1

    print(f"\nLogged {n} run(s) into TensorBoard at {RUNS_DIR}")
    print(f"Open the UI with:  tensorboard --logdir runs")
    print(f"Then go to:        http://localhost:6006")


if __name__ == "__main__":
    main()
