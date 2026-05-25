"""SAM 2.1: box-prompts from labelme rectangles (manual 64×64 dataset)."""
from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image, ImageDraw

import config
from plot_utils import cyrillic_font, save_class_legend
from sam2_loader import build_predictor

Image.MAX_IMAGE_PIXELS = None


def scaled_box(points: list[list[float]], scale: float) -> np.ndarray:
    xs = [p[0] * scale for p in points]
    ys = [p[1] * scale for p in points]
    return np.array([[min(xs), min(ys), max(xs), max(ys)]], dtype=np.float32)


def overlay_semantic(
    image: np.ndarray,
    class_map: np.ndarray,
    score_map: np.ndarray,
    alpha: float,
    min_score: float,
) -> np.ndarray:
    out = image.astype(np.float32).copy()
    for idx, cls in enumerate(config.CLASSES):
        mask = (class_map == idx) & (score_map >= min_score)
        if not mask.any():
            continue
        color = np.array(config.CLASS_COLORS[cls], dtype=np.float32)
        out[mask] = (1 - alpha) * out[mask] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def save_class_montage(image: np.ndarray, class_map: np.ndarray, score_map: np.ndarray, path) -> None:
    tiles = []
    font = cyrillic_font(18)
    for idx, cls in enumerate(config.CLASSES):
        vis = Image.fromarray(image.copy())
        mask = (class_map == idx) & (score_map > 0.35)
        if mask.any():
            arr = np.asarray(vis).copy()
            arr[mask] = (0.5 * arr[mask] + 0.5 * np.array(config.CLASS_COLORS[cls])).astype(np.uint8)
            vis = Image.fromarray(arr)
        draw = ImageDraw.Draw(vis)
        draw.text((8, 8), cls, fill=(255, 255, 255), font=font)
        tiles.append(vis.resize((512, 512), Image.Resampling.LANCZOS))
    montage = Image.new("RGB", (512 * len(tiles), 512))
    for i, t in enumerate(tiles):
        montage.paste(t, (i * 512, 0))
    montage.save(path, quality=92)


def main() -> None:
    if not config.SAM2_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"SAM checkpoint missing: {config.SAM2_CHECKPOINT}\n"
            "Run: bash sam/checkpoints/download.sh"
        )

    config.ensure_dirs()
    image, scale = config.load_rgb_array(config.INFER_MAX_SIDE)
    h, w = image.shape[:2]
    meta = json.loads(config.LABELME_JSON.read_text(encoding="utf-8"))

    save_class_legend(config.SAM_OUTPUT / "sam2_legend.png")

    predictor = build_predictor()
    score_volume = np.zeros((h, w, len(config.CLASSES)), dtype=np.float32)
    masks_debug = []

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        predictor.set_image(image)

        for shape in meta["shapes"]:
            label = shape["label"]
            if label not in config.CLASSES:
                continue
            cls_idx = config.CLASSES.index(label)
            box = scaled_box(shape["points"], scale)

            masks, scores, _ = predictor.predict(
                box=box,
                multimask_output=False,
            )
            mask = masks[0].astype(bool)
            score = float(scores[0])
            if score < 0.5:
                continue

            score_volume[:, :, cls_idx] = np.maximum(score_volume[:, :, cls_idx], mask.astype(np.float32) * score)
            masks_debug.append(
                {
                    "label": label,
                    "box": box[0].tolist(),
                    "score": score,
                    "area": int(mask.sum()),
                }
            )

    class_map = score_volume.argmax(axis=2)
    score_map = score_volume.max(axis=2)
    overlay = overlay_semantic(image, class_map, score_map, config.OVERLAY_ALPHA, min_score=0.35)

    preview = Image.fromarray(image)
    draw = ImageDraw.Draw(preview)
    for shape in meta["shapes"]:
        pts = shape["points"]
        sx1, sy1 = int(pts[0][0] * scale), int(pts[0][1] * scale)
        sx2, sy2 = int(pts[1][0] * scale), int(pts[1][1] * scale)
        color = config.CLASS_COLORS.get(shape["label"], (255, 0, 0))
        draw.rectangle([sx1, sy1, sx2, sy2], outline=color, width=2)
    preview.save(config.SAM_OUTPUT / "sam2_boxes_preview.jpg", quality=92)

    Image.fromarray(overlay).save(config.SAM_OUTPUT / "sam2_prompt_overlay.jpg", quality=92)
    save_class_montage(image, class_map, score_map, config.SAM_OUTPUT / "sam2_prompt_classes.png")

    report = {
        "method": "SAM2.1 box prompts from labelme rectangles",
        "checkpoint": config.SAM2_CHECKPOINT.name,
        "num_prompts": len(masks_debug),
        "image_size": [w, h],
        "scale": scale,
        "masks": masks_debug,
    }
    (config.SAM_OUTPUT / "sam2_prompt_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(masks_debug)} masks -> sam/output/sam2_prompt_overlay.jpg")


if __name__ == "__main__":
    main()
