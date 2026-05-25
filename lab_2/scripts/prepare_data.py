"""Validate annotations + dataset, build preview images (no zip, no duplicates)."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

_LAB = Path(__file__).resolve().parent.parent
for p in (_LAB, _LAB / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config

Image.MAX_IMAGE_PIXELS = None
EXPECTED = 39


def load_annotations() -> dict:
    if not config.LABELME_JSON.is_file():
        raise FileNotFoundError(f"Нет разметки: {config.LABELME_JSON}")
    return json.loads(config.LABELME_JSON.read_text(encoding="utf-8"))


def count_dataset() -> dict[str, int]:
    stats: dict[str, int] = {}
    for cls in config.CLASSES:
        stats[cls] = len(list((config.DATASET / cls).glob("*.png")))
    return stats


def save_previews(meta: dict) -> None:
    config.ensure_dirs()
    arr, scale = config.load_rgb_array()
    Image.fromarray(arr).save(config.OUTPUT_PREVIEW / "source_preview.jpg", quality=92)

    preview = Image.fromarray(arr)
    draw = ImageDraw.Draw(preview)
    for shape in meta["shapes"]:
        pts = shape["points"]
        sx1, sy1 = int(pts[0][0] * scale), int(pts[0][1] * scale)
        sx2, sy2 = int(pts[1][0] * scale), int(pts[1][1] * scale)
        color = config.CLASS_COLORS.get(shape["label"], (255, 0, 0))
        draw.rectangle([sx1, sy1, sx2, sy2], outline=color, width=2)
    preview.save(config.OUTPUT_PREVIEW / "annotations_preview.jpg", quality=92)


def save_montage(stats: dict[str, int]) -> None:
    fig, axes = plt.subplots(1, len(config.CLASSES), figsize=(12, 3))
    for ax, cls in zip(axes, config.CLASSES):
        files = sorted((config.DATASET / cls).glob("*.png"))
        if files:
            ax.imshow(Image.open(files[len(files) // 2]))
        ax.set_title(f"{cls}\n({stats[cls]})", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Патчи {config.PATCH_SIZE}x{config.PATCH_SIZE}", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_PREVIEW / "dataset_montage.png", dpi=160)
    plt.close(fig)


def save_summary(stats: dict[str, int]) -> None:
    with Image.open(config.IMAGE_PATH) as im:
        full_w, full_h = im.size
    _, scale = config.load_rgb_array()
    summary = {
        "image": config.IMAGE_NAME,
        "patch_size": config.PATCH_SIZE,
        "annotations": str(config.LABELME_JSON.relative_to(config.ROOT)),
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
    meta = load_annotations()
    n_ann = len(meta["shapes"])
    ann_counts = Counter(s["label"] for s in meta["shapes"])
    stats = count_dataset()

    if n_ann != EXPECTED:
        raise RuntimeError(f"В source.json ожидалось {EXPECTED} боксов, есть {n_ann}")
    if sum(stats.values()) != EXPECTED:
        raise RuntimeError(f"В dataset/ ожидалось {EXPECTED} png, есть {sum(stats.values())}")

    save_previews(meta)
    save_montage(stats)
    save_summary(stats)
    print("Annotations:", dict(ann_counts), "total", n_ann)
    print("Dataset:", stats, "total", sum(stats.values()))


if __name__ == "__main__":
    main()
