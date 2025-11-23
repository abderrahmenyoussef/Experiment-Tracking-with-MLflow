# src/train_cv.py
import os
import argparse
import mlflow
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# ---- Import robuste de utils.set_global_seed (package OU fichier) ----
def _load_set_global_seed():
    try:
        # Cas 1 : lancé avec "python -m src.train_cv"
        from .utils import set_global_seed  # type: ignore
        return set_global_seed
    except Exception:
        # Cas 2 : lancé avec "python src/train_cv.py"
        import importlib.util
        utils_path = Path(__file__).resolve().parent / "utils.py"
        spec = importlib.util.spec_from_file_location("utils_local", utils_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)  # type: ignore
        return getattr(mod, "set_global_seed")

set_global_seed = _load_set_global_seed()

# Aligne l’URI MLflow pour le script ET les callbacks Ultralytics
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

def latest_run_dir(base="runs/detect"):
    paths = sorted(Path(base).glob("*"), key=lambda p: p.stat().st_mtime)
    return paths[-1] if paths else None

def log_yolo_artifacts(run_dir: Path):
    for f in ["results.png", "confusion_matrix.png", "PR_curve.png", "labels_correlogram.jpg"]:
        p = run_dir / f
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path="yolo_plots")
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        mlflow.log_artifact(str(best), artifact_path="weights")

def log_yolo_metrics(run_dir: Path):
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    last = df.iloc[-1].to_dict()
    # Compat multi-versions Ultralytics (noms de colonnes)
    candidates = {
        "precision":   ["metrics/precision(B)", "metrics/precision"],
        "recall":      ["metrics/recall(B)", "metrics/recall"],
        "mAP50":       ["metrics/mAP50(B)", "metrics/mAP50"],
        "mAP50-95":    ["metrics/mAP50-95(B)", "metrics/mAP50-95", "metrics/mAP50-95(M)"],
    }
    for m, cols in candidates.items():
        for c in cols:
            if c in last:
                try:
                    mlflow.log_metric(m, float(last[c]))
                except Exception:
                    pass
                break

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tiny_coco.yaml")
    ap.add_argument("--model", default="yolov8n.pt")  # ou yolo11n.pt
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--lr0", type=float, default=0.005)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp-name", default="cv_yolo_tiny")
    args = ap.parse_args()

    set_global_seed(args.seed)
    mlflow.set_experiment(args.exp_name)
    run_name = f"{Path(args.model).stem}_e{args.epochs}_sz{args.imgsz}_lr{args.lr0}_s{args.seed}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": args.model, "epochs": args.epochs, "imgsz": args.imgsz,
            "lr0": args.lr0, "batch": args.batch, "seed": args.seed, "data": args.data
        })
        mlflow.set_tags({"task": "object-detection", "dataset": "tiny_coco_person"})

        model = YOLO(args.model)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            lr0=args.lr0,
            batch=args.batch,
            seed=args.seed,
            project="runs",
            name=run_name,
            verbose=False
        )

        run_dir = Path("runs/detect") / run_name
        if not run_dir.exists():
            run_dir = latest_run_dir()
        if run_dir:
            log_yolo_metrics(run_dir)
            log_yolo_artifacts(run_dir)

if __name__ == "__main__":
    main()
