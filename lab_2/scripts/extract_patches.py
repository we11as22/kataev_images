"""Extract square patches from labelme rectangle annotations (center of each box)."""
from __future__ import annotations

import json
import shutil

import numpy as np
from PIL import Image

import config

Image.MAX_IMAGE_PIXELS = None


def box_center(points: list[list[float]]) -> tuple[int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return int((min(xs) + max(xs)) / 2), int((min(ys) + max(ys)) / 2)


def extract_patch(arr: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray | None:
    h, w = arr.shape[:2]
    half = size // 2
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return arr[y0:y1, x0:x1]


def _save_source_preview(arr: np.ndarray) -> None:
    preview = Image.fromarray(arr)
    w, h = preview.size
    scale = min(1.0, config.INFER_MAX_SIDE / max(w, h))
    if scale < 1.0:
        preview = preview.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    preview.save(config.OUTPUT_PREVIEW / "source_preview.jpg", quality=92)


def main() -> None:
    config.ensure_dirs()
    meta = json.loads(config.LABELME_JSON.read_text(encoding="utf-8"))

    if config.DATASET.exists():
        shutil.rmtree(config.DATASET)
    for cls in config.CLASSES:
        (config.DATASET / cls).mkdir(parents=True)

    with Image.open(config.IMAGE_PATH) as im:
        full_w, full_h = im.size
    arr, scale = config.load_rgb_array()
    _save_source_preview(arr)

    stats: dict[str, int] = {cls: 0 for cls in config.CLASSES}
    manifest = []

    for shape in meta["shapes"]:
        label = shape["label"]
        if label not in config.CLASSES:
            continue
        cx, cy = box_center(shape["points"])
        cx, cy = int(cx * scale), int(cy * scale)
        patch = extract_patch(arr, cx, cy, config.PATCH_SIZE)
        if patch is None:
            continue
        idx = stats[label]
        fname = f"{label.replace(' ', '_')}_{idx:03d}.jpg"
        out_path = config.DATASET / label / fname
        Image.fromarray(patch).save(out_path, quality=95)
        manifest.append({"file": str(out_path.relative_to(config.ROOT)), "class": label, "cx": cx, "cy": cy})
        stats[label] += 1

    summary = {
        "image": config.IMAGE_NAME,
        "patch_size": config.PATCH_SIZE,
        "mode": "center_of_rectangle",
        "infer_scale": scale,
        "full_size": [full_w, full_h],
        "counts": stats,
        "total": sum(stats.values()),
        "patches": manifest,
    }
    (config.OUTPUT_PREVIEW / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_montage(stats)
    print("Saved patches:", stats, "total", summary["total"])


def _save_montage(stats: dict[str, int]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(config.CLASSES), figsize=(12, 3))
    for ax, cls in zip(axes, config.CLASSES):
        files = sorted((config.DATASET / cls).glob("*.jpg"))
        if files:
            img = Image.open(files[len(files) // 2])
            ax.imshow(img)
        ax.set_title(f"{cls}\n({stats[cls]})", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Примеры патчей {config.PATCH_SIZE}×{config.PATCH_SIZE}", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_PREVIEW / "dataset_montage.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
