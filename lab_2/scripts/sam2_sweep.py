"""Compare SAM2 segmentation under different parameter settings."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import config
from sam2_loader import build_amg

Image.MAX_IMAGE_PIXELS = None


def load_image(max_side: int) -> np.ndarray:
    with Image.open(config.IMAGE_PATH) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        return np.asarray(im).copy()


def run_amg(image: np.ndarray, points_per_side: int, pred_iou: float, stability: float) -> tuple[np.ndarray, int]:
    amg = build_amg(
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou,
        stability_score_thresh=stability,
        min_mask_region_area=0,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        masks = amg.generate(image)

    out = image.astype(np.float32).copy()
    rng = np.random.default_rng(config.SEED + points_per_side + int(pred_iou * 100))
    for ann in sorted(masks, key=lambda x: x["area"], reverse=True):
        color = rng.integers(40, 220, size=3).astype(np.float32)
        out[ann["segmentation"]] = 0.55 * out[ann["segmentation"]] + 0.45 * color
    return np.clip(out, 0, 255).astype(np.uint8), len(masks)


def main() -> None:
    if not config.SAM2_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"SAM checkpoint missing: {config.SAM2_CHECKPOINT}\n"
            "Run: bash checkpoints/download.sh"
        )

    config.ensure_dirs()
    image = load_image(config.INFER_MAX_SIDE)

    sweeps = [
        {"name": "iou_0.70", "points_per_side": 16, "pred_iou_thresh": 0.70, "stability_score_thresh": 0.88},
        {"name": "iou_0.82", "points_per_side": 24, "pred_iou_thresh": 0.82, "stability_score_thresh": 0.92},
        {"name": "iou_0.92", "points_per_side": 24, "pred_iou_thresh": 0.92, "stability_score_thresh": 0.96},
        {"name": "points_16", "points_per_side": 16, "pred_iou_thresh": 0.82, "stability_score_thresh": 0.92},
        {"name": "points_32", "points_per_side": 32, "pred_iou_thresh": 0.82, "stability_score_thresh": 0.92},
    ]

    results = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()

    axes[0].imshow(image)
    axes[0].set_title("Исходное изображение")
    axes[0].axis("off")

    for ax, sweep in zip(axes[1:], sweeps):
        vis, n_masks = run_amg(
            image,
            points_per_side=sweep["points_per_side"],
            pred_iou=sweep["pred_iou_thresh"],
            stability=sweep["stability_score_thresh"],
        )
        ax.imshow(vis)
        ax.set_title(f"{sweep['name']}\nmasks={n_masks}", fontsize=9)
        ax.axis("off")
        results.append({**sweep, "num_masks": n_masks})

    fig.tight_layout()
    fig.savefig(config.OUTPUT_SAM2 / "sam2_parameter_sweep.png", dpi=160)
    plt.close(fig)

    (config.OUTPUT_SAM2 / "sam2_sweep_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved output/sam2/sam2_parameter_sweep.png")


if __name__ == "__main__":
    main()
