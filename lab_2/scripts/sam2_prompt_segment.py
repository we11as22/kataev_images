"""SAM 2.1 segmentation with box prompts from labelme annotations."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

import config
from sam2_loader import build_predictor

Image.MAX_IMAGE_PIXELS = None


def load_image(max_side: int) -> tuple[np.ndarray, float]:
    return config.load_rgb_array(max_side)


def box_xyxy(points: list[list[float]], scale: float) -> np.ndarray:
    xs = [p[0] * scale for p in points]
    ys = [p[1] * scale for p in points]
    return np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32)


def overlay_semantic(
    image: np.ndarray,
    class_map: np.ndarray,
    score_map: np.ndarray,
    alpha: float,
    min_score: float = 0.0,
) -> np.ndarray:
    out = image.astype(np.float32).copy()
    for idx, cls in enumerate(config.CLASSES):
        mask = (class_map == idx) & (score_map >= min_score)
        if not mask.any():
            continue
        color = np.array(config.CLASS_COLORS[cls], dtype=np.float32)
        out[mask] = (1 - alpha) * out[mask] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_annotations_preview(image: np.ndarray, scale: float) -> None:
    meta = json.loads(config.LABELME_JSON.read_text(encoding="utf-8"))
    preview = Image.fromarray(image)
    draw = ImageDraw.Draw(preview)
    for shape in meta["shapes"]:
        box = box_xyxy(shape["points"], 1.0 if scale == 1.0 else 1.0)
        # points already in original coords; scale image was applied in load_image
        pts = shape["points"]
        xs = [p[0] * scale for p in pts]
        ys = [p[1] * scale for p in pts]
        color = config.CLASS_COLORS.get(shape["label"], (255, 0, 0))
        draw.rectangle([min(xs), min(ys), max(xs), max(ys)], outline=color, width=3)
        draw.text((min(xs) + 4, min(ys) + 4), shape["label"], fill=color)
    preview.save(config.OUTPUT_PREVIEW / "annotations_preview.jpg", quality=92)


def draw_legend() -> None:
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.axis("off")
    for i, cls in enumerate(config.CLASSES):
        c = np.array(config.CLASS_COLORS[cls]) / 255.0
        ax.add_patch(plt.Rectangle((0.05, 0.78 - i * 0.22), 0.12, 0.14, color=c))
        ax.text(0.22, 0.85 - i * 0.22, cls, va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_PREVIEW / "legend.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not config.SAM2_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"SAM checkpoint missing: {config.SAM2_CHECKPOINT}\n"
            "Run: bash checkpoints/download.sh"
        )

    config.ensure_dirs()
    image, scale = load_image(config.INFER_MAX_SIDE)
    h, w = image.shape[:2]
    meta = json.loads(config.LABELME_JSON.read_text(encoding="utf-8"))

    draw_annotations_preview(image, scale)
    draw_legend()

    predictor = build_predictor()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        predictor.set_image(image)

        score_volume = np.zeros((h, w, len(config.CLASSES)), dtype=np.float32)
        masks_debug = []

        for shape in meta["shapes"]:
            label = shape["label"]
            if label not in config.CLASSES:
                continue
            cls_idx = config.CLASSES.index(label)
            box = box_xyxy(shape["points"], scale)
            x0, y0, x1, y1 = [int(v) for v in box]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            masks, iou_preds, _ = predictor.predict(box=box, multimask_output=False)
            mask = masks[0].astype(bool)
            # keep segmentation inside the prompt box — avoids bleed to whole frame
            region = np.zeros((h, w), dtype=bool)
            region[y0:y1, x0:x1] = True
            mask &= region
            score = float(iou_preds[0])
            if score < 0.5:
                continue
            score_volume[:, :, cls_idx] = np.maximum(score_volume[:, :, cls_idx], mask.astype(np.float32) * score)
            masks_debug.append({"label": label, "box": box.tolist(), "iou": score, "area": int(mask.sum())})

        class_map = score_volume.argmax(axis=2)
        score_map = score_volume.max(axis=2)
        overlay = overlay_semantic(image, class_map, score_map, config.OVERLAY_ALPHA, min_score=0.35)

        Image.fromarray(image).save(config.OUTPUT_PREVIEW / "source_resized.jpg", quality=92)
        Image.fromarray(overlay).save(config.OUTPUT_SAM2 / "sam2_prompt_overlay.jpg", quality=92)

        # per-class masks montage
        fig, axes = plt.subplots(1, len(config.CLASSES), figsize=(12, 3))
        for idx, cls in enumerate(config.CLASSES):
            mask = (class_map == idx) & (score_map > 0)
            vis = image.copy()
            vis[mask] = (0.5 * vis[mask] + 0.5 * np.array(config.CLASS_COLORS[cls])).astype(np.uint8)
            axes[idx].imshow(vis)
            axes[idx].set_title(cls)
            axes[idx].axis("off")
        fig.tight_layout()
        fig.savefig(config.OUTPUT_SAM2 / "sam2_prompt_classes.png", dpi=160)
        plt.close(fig)

    report = {
        "method": "SAM2.1 box prompts from labelme",
        "checkpoint": str(config.SAM2_CHECKPOINT.name),
        "config": config.SAM2_CONFIG,
        "image_size": [w, h],
        "scale": scale,
        "alpha": config.OVERLAY_ALPHA,
        "masks": masks_debug,
        "class_pixels": {
            cls: int(((class_map == i) & (score_map > 0)).sum()) for i, cls in enumerate(config.CLASSES)
        },
    }
    (config.OUTPUT_SAM2 / "sam2_prompt_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved sam2_prompt_overlay.jpg")


if __name__ == "__main__":
    main()
