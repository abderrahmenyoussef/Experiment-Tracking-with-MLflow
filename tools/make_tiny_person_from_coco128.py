import os, zipfile, io, shutil, random, csv, glob, requests
from pathlib import Path
from PIL import Image

random.seed(42)

COCO128_ZIP = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"  # doc Ultralytics / vous pouvez changer le dataset
BUILD_DIR = Path("build/coco128")
OUT_DIR   = Path("data/tiny_coco")
TARGET_COUNTS = {"train": 40, "val": 10, "test": 10}
RESIZE_TO = 320  

def ensure_dirs():
    for split in ["train", "val", "test"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def download_and_extract():
    if BUILD_DIR.exists():
        return
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    print("Téléchargement COCO128...")
    r = requests.get(COCO128_ZIP, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(BUILD_DIR)
    print("OK.")

def collect_person_samples():
    img_dir = next((BUILD_DIR / "coco128").rglob("images/train2017"))
    lbl_dir = next((BUILD_DIR / "coco128").rglob("labels/train2017"))
    pairs = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        lab_path = lbl_dir / (img_path.stem + ".txt")
        if not lab_path.exists():
            continue
        with open(lab_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        person_lines = [ln for ln in lines if ln.split()[0] == "0"]
        if person_lines:
            pairs.append((img_path, lab_path, person_lines))
    return pairs

def resize_save(src_img, dst_img, size=320):
    img = Image.open(src_img).convert("RGB")
    img = img.resize((size, size))
    img.save(dst_img, quality=92)

def write_labels(person_lines, dst_lbl):
    with open(dst_lbl, "w") as f:
        for ln in person_lines:
            f.write(ln + "\n")

def main():
    ensure_dirs()
    download_and_extract()
    pairs = collect_person_samples()
    random.shuffle(pairs)
    needed = sum(TARGET_COUNTS.values())
    pairs = pairs[:needed] if len(pairs) >= needed else pairs

    splits = []
    splits += [("train",) * TARGET_COUNTS["train"]]
    splits += [("val",)   * TARGET_COUNTS["val"]]
    splits += [("test",)  * TARGET_COUNTS["test"]]
    flat_splits = [s for tup in splits for s in tup]

    if len(pairs) < len(flat_splits):
        raise SystemExit(f"Trop peu d'images personnes trouvées ({len(pairs)}). Baissez les quotas.")

    idx = 0
    for split in flat_splits:
        img_src, lbl_src, person_lines = pairs[idx]
        idx += 1
        img_dst = OUT_DIR / "images" / split / f"{img_src.stem}.jpg"
        lbl_dst = OUT_DIR / "labels" / split / f"{img_src.stem}.txt"
        resize_save(img_src, img_dst, RESIZE_TO)
        write_labels(person_lines, lbl_dst)

    print(f"Dataset construit sous {OUT_DIR} (train/val/test).")

if __name__ == "__main__":
    main()
