@echo off
setlocal enabledelayedexpansion

for %%S in (320 416) do (
  for %%L in (0.005 0.010) do (
    for %%E in (1 42) do (
      python -u src\train_cv.py ^
        --data data\tiny_coco.yaml ^
        --model yolov8n.pt ^
        --epochs 3 ^
        --imgsz %%S ^
        --lr0 %%L ^
        --seed %%E ^
        --exp-name cv_yolo_tiny
    )
  )
)

endlocal
