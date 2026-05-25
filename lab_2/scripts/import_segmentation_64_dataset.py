"""Import manually checked segmentation_64_dataset.zip into lab layout."""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import config

Image.MAX_IMAGE_PIXELS = None

ZIP_NAME = "segmentation_64_dataset.zip"
ARCHIVE_ROOT = "segmentation_64_dataset"
EXTRACT_DIR = config.ROOT / ARCHIVE_ROOT

# folder names inside the zip archive
CROP_FOLDER_MAP = {
    "Дорога": "Дорога",
    "Зелёное поле": "Зеленое_поле",
    "Убранное поле": "Убранное_поле",
    "Грунт": "Грунт",
}


def extract_archive() -> Path:
    zpath = config.ROOT / ZIP_NAME
    if not zpath.is_file():
        raise FileNotFoundError(f"Архив не найден: {zpath}")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(config.ROOT)
    return EXTRACT_DIR


def import_labelme(src_dir: Path) -> None:
    labelme_src = src_dir / "rectangles_64_labelme.json"
    meta = json.loads(labelme_src.read_text(encoding="utf-8"))
    meta["imagePath"] = f"../IMAGES/{config.IMAGE_NAME}"
    config.ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    config.LABELME_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def import_crops(src_dir: Path) -> dict[str, int]:
    if config.DATASET.exists():
        shutil.rmtree(config.DATASET)
    stats: dict[str, int] = {}
    crops_root = src_dir / "crops"
    for cls in config.CLASSES:
        out_dir = config.DATASET / cls
        out_dir.mkdir(parents=True)
        src_name = CROP_FOLDER_MAP[cls]
        src_dir_cls = crops_root / src_name
        count = 0
        for src in sorted(src_dir_cls.glob("*.png")):
            dst = out_dir / src.name
            shutil.copy2(src, dst)
            count += 1
        stats[cls] = count
    return stats


def save_previews(src_dir: Path) -> None:
    config.ensure_dirs()
    for name in ("overview_rectangles_64.jpg", "preview_checked_patches.jpg"):
        src = src_dir / name
        if src.is_file():
            shutil.copy2(src, config.OUTPUT_PREVIEW / name)

    arr, scale = config.load_rgb_array()
    Image.fromarray(arr).save(config.OUTPUT_PREVIEW / "source_preview.jpg", quality=92)

    meta = json.loads(config.LABELME_JSON.read_text(encoding="utf-8"))
    preview = Image.fromarray(arr)
    draw = ImageDraw.Draw(preview)
    for shape in meta["shapes"]:
        pts = shape["points"]
        sx1, sy1 = int(pts[0][0] * scale), int(pts[0][1] * scale)
        sx2, sy2 = int(pts[1][0] * scale), int(pts[1][1] * scale)
        color = config.CLASS_COLORS.get(shape["label"], (255, 0, 0))
        draw.rectangle([sx1, sy1, sx2, sy2], outline=color, width=2)
    preview.save(config.OUTPUT_PREVIEW / "annotations_preview.jpg", quality=95)


def save_montage(stats: dict[str, int]) -> None:
    fig, axes = plt.subplots(1, len(config.CLASSES), figsize=(12, 3))
    for ax, cls in zip(axes, config.CLASSES):
        files = sorted((config.DATASET / cls).glob("*.png"))
        if files:
            ax.imshow(Image.open(files[len(files) // 2]))
        ax.set_title(f"{cls}\n({stats[cls]})", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Патчи {config.PATCH_SIZE}×{config.PATCH_SIZE} (ручная разметка)", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_PREVIEW / "dataset_montage.png", dpi=160)
    plt.close(fig)


def save_summary(stats: dict[str, int], src_dir: Path) -> None:
    with Image.open(config.IMAGE_PATH) as im:
        full_w, full_h = im.size
    _, scale = config.load_rgb_array()
    summary = {
        "source": str(src_dir.relative_to(config.ROOT)),
        "image": config.IMAGE_NAME,
        "patch_size": config.PATCH_SIZE,
        "mode": "imported_manual_dataset",
        "infer_scale": scale,
        "full_size": [full_w, full_h],
        "counts": stats,
        "total": sum(stats.values()),
    }
    (config.OUTPUT_PREVIEW / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    config.ensure_dirs()
    src_dir = extract_archive()
    import_labelme(src_dir)
    stats = import_crops(src_dir)
    save_previews(src_dir)
    save_montage(stats)
    save_summary(stats, src_dir)
    print("Imported dataset:", stats, "total", sum(stats.values()))
    print("  annotations ->", config.LABELME_JSON)
    print("  crops       ->", config.DATASET)


if __name__ == "__main__":
    main()
