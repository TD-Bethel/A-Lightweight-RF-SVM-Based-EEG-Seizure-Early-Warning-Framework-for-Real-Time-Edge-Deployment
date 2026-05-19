"""
log_model_to_mlflow.py  --  Backfill MLflow with an already-trained model

Reads training_report.json from a saved model directory (e.g. RF60+SVM40_C2)
and creates a matching MLflow run with all params, metrics, and artifacts.
Useful for getting historical training results into the MLflow UI without
having to re-train.

Usage:
    python scripts/utils/log_model_to_mlflow.py
    python scripts/utils/log_model_to_mlflow.py --model-dir models/RF60+SVM40_C2
    python scripts/utils/log_model_to_mlflow.py --all          # log every model dir

View later with:
    mlflow ui
    (then open http://localhost:5000)
"""

import argparse
import json
import os
import sys

import mlflow

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models")


def log_one(model_dir: str) -> bool:
    report_path = os.path.join(model_dir, "training_report.json")
    if not os.path.exists(report_path):
        print(f"  SKIP  no training_report.json in {model_dir}")
        return False

    with open(report_path, "r", encoding="utf-8") as f:
        r = json.load(f)

    run_name = os.path.basename(os.path.normpath(model_dir))
    print(f"  -> logging {run_name}")

    with mlflow.start_run(run_name=run_name):
        # Params (training configuration)
        mlflow.log_params({
            "model"           : r.get("model"),
            "trained_on"      : r.get("trained_on"),
            "split"           : r.get("split"),
            "n_features"      : r.get("n_features"),
            "n_selected"      : r.get("n_selected"),
            "n_channels"      : r.get("n_channels"),
            "feature_selector": r.get("feature_selector"),
            "normalisation"   : r.get("normalisation"),
            "svm_class_weight": r.get("svm_class_weight"),
            "normal_threshold": r.get("normal_threshold"),
        })

        # Metrics per split
        for split in ("training", "validation", "test"):
            m = r.get(split, {})
            if not m:
                continue
            short = {"training": "train", "validation": "val", "test": "test"}[split]
            for key in ("accuracy", "f1", "seizure_recall"):
                if key in m:
                    mlflow.log_metric(f"{short}_{key}", float(m[key]))
            for cls_name, cls_m in (m.get("per_class") or {}).items():
                tag = cls_name.lower().replace("-", "_").replace(" ", "_")
                for k, v in cls_m.items():
                    mlflow.log_metric(f"{short}_{tag}_{k}", float(v))

        # Derived metric
        try:
            mlflow.log_metric(
                "overfitting_gap",
                float(r["training"]["accuracy"]) - float(r["validation"]["accuracy"]),
            )
        except KeyError:
            pass

        # Tag with timestamp and source folder
        mlflow.set_tag("source_timestamp", r.get("timestamp", "unknown"))
        mlflow.set_tag("source_path", model_dir)

        # Artifacts: report, all .pkl model files, every PNG diagnostic figure
        mlflow.log_artifact(report_path)
        for fname in os.listdir(model_dir):
            full = os.path.join(model_dir, fname)
            if fname.endswith(".pkl"):
                mlflow.log_artifact(full, artifact_path="models")
            elif fname.endswith(".png"):
                mlflow.log_artifact(full, artifact_path="figures")
            elif fname == "HOW_TO_USE.txt":
                mlflow.log_artifact(full)

    print(f"     OK")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(MODELS_DIR, "RF60+SVM40_C2"),
                    help="Saved model directory to log (default: RF60+SVM40_C2)")
    ap.add_argument("--all", action="store_true",
                    help="Log every subdirectory under models/ that has a training_report.json")
    args = ap.parse_args()

    mlflow.set_tracking_uri(f"file:{os.path.join(ROOT, 'mlruns')}")
    mlflow.set_experiment("neurowatch_training")

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

    print(f"\nLogged {n} run(s) into MLflow at file:{os.path.join(ROOT, 'mlruns')}")
    print(f"Open the UI with:  mlflow ui   (then http://localhost:5000)")


if __name__ == "__main__":
    main()
