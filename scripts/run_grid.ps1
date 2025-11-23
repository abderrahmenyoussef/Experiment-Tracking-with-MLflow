$Sizes = @(320, 416)
$LRs   = @(0.005, 0.010)
$Seeds = @(1, 42)

foreach ($size in $Sizes) {
  foreach ($lr in $LRs) {
    foreach ($seed in $Seeds) {
      python -u src/train_cv.py `
        --data data/tiny_coco.yaml `
        --model yolov8n.pt `
        --epochs 3 `
        --imgsz $size `
        --lr0 $lr `
        --seed $seed `
        --exp-name cv_yolo_tiny
    }
  }
}
