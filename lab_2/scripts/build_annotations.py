"""Build labelme annotations by placing boxes on homogeneous image regions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import config

Image.MAX_IMAGE_PIXELS = None


def rect(label: str, cx: int, cy: int, bw: int, bh: int, w: int, h: int) -> dict | None:
    x1, y1 = cx - bw // 2, cy - bh // 2
    x2, y2 = cx + bw // 2, cy + bh // 2
    x1, x2 = max(128, x1), min(w - 128, x2)
    y1, y2 = max(128, y1), min(h - 128, y2)
    if x2 - x1 < 180 or y2 - y1 < 180:
        return None
    return {
        "label": label,
        "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
        "group_id": None,
        "shape_type": "rectangle",
        "flags": {},
    }


def build_shapes(w: int, h: int) -> list[dict]:
    shapes: list[dict] = []
    for cy in np.linspace(450, 3350, 8, dtype=int):
        for cx in np.linspace(550, 2050, 5, dtype=int):
            if s := rect("Зелёное поле", int(cx), int(cy), 340, 300, w, h):
                shapes.append(s)
    for cy in np.linspace(400, 3350, 7, dtype=int):
        if s := rect("Дорога", 2910, int(cy), 220, 260, w, h):
            shapes.append(s)
    for cy in np.linspace(500, 3300, 7, dtype=int):
        if s := rect("Кусты", 2610, int(cy), 200, 260, w, h):
            shapes.append(s)
    for cy in np.linspace(450, 3350, 8, dtype=int):
        for cx in np.linspace(3450, 5100, 5, dtype=int):
            if s := rect("Убранное поле", int(cx), int(cy), 340, 300, w, h):
                shapes.append(s)
    return shapes


def main() -> None:
    config.ensure_dirs()
    with Image.open(config.IMAGE_PATH) as im:
        w, h = im.size

    shapes = build_shapes(w, h)
    meta = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": f"../IMAGES/{config.IMAGE_NAME}",
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }
    config.LABELME_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.LABELME_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    with Image.open(config.IMAGE_PATH) as im:
        preview = im.convert("RGB")
    preview.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    ps = 2048 / max(w, h)
    draw = ImageDraw.Draw(preview)
    for shape in shapes:
        pts = shape["points"]
        color = config.CLASS_COLORS.get(shape["label"], (255, 0, 0))
        draw.rectangle(
            [pts[0][0] * ps, pts[0][1] * ps, pts[1][0] * ps, pts[1][1] * ps],
            outline=color,
            width=2,
        )
    preview.save(config.OUTPUT_PREVIEW / "annotations_preview.jpg", quality=95)
    print(f"Saved {len(shapes)} boxes -> {config.LABELME_JSON}")


if __name__ == "__main__":
    main()
