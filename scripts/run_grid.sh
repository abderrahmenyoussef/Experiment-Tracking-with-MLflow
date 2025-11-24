#!/usr/bin/env bash
# scripts/run_grid.sh
set -e

# Grille très légère : 2x tailles + 2x LR + 2 seeds = 8 runs
for SIZE in 320 416; do
  for LR in 0.005 0.010; do
    for SEED in 1 42; do
      python -u src/train_cv.py --data data/tiny_coco.yaml \
        --model yolov8n.pt --epochs 3 --imgsz $SIZE --lr0 $LR --seed $SEED \
        --exp-name cv_yolo_tiny
    done
  done
done
