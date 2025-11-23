# MLflow CV Tiny (YOLO)

Objectif : tracer et comparer plusieurs runs YOLO tiny sur un mini-dataset (1 classe `person`) avec MLflow.

## Quickstart
```bash
docker compose up -d   # MLflow (5000), MinIO (9001)
pip install -r requirements.txt
python tools/make_tiny_person_from_coco128.py
```

Baseline (ex.) : python src/train_cv.py --epochs 3 --imgsz 320 --exp-name cv_yolo_tiny

Grille de runs :

    Linux/macOS : bash scripts/run_grid.sh

    Windows (PS) : powershell -ExecutionPolicy Bypass -File scripts\run_grid.ps1

    Windows (CMD) : scripts\run_grid.cmd

UI MLflow : http://localhost:5000


Décision : compléter reports/templates/decision_template.md (captures MLflow à l’appui).