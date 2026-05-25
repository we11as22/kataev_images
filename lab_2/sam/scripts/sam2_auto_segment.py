"""SAM 2.1 automatic mask generation on the full image."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import config
from sam2_loader import build_amg

Image.MAX_IMAGE_PIXELS = None


def load_image(max_side: int) -> tuple[np.ndarray, float]:
    with Image.open(config.IMAGE_PATH) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        return np.asarray(im).copy(), scale


def colorize_masks(image: np.ndarray, masks: list[dict], alpha: float = 0.5) -> np.ndarray:
    out = image.astype(np.float32).copy()
    rng = np.random.default_rng(config.SEED)
    for ann in sorted(masks, key=lambda x: x["area"], reverse=True):
        color = rng.integers(40, 220, size=3).astype(np.float32)
        m = ann["segmentation"]
        out[m] = (1 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    if not config.SAM2_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"SAM checkpoint missing: {config.SAM2_CHECKPOINT}\n"
            "Run: bash checkpoints/download.sh"
        )

    config.ensure_dirs()
    image, scale = load_image(config.INFER_MAX_SIDE)

    amg = build_amg(
        points_per_side=config.AMG_POINTS_PER_SIDE,
        pred_iou_thresh=config.AMG_PRED_IOU_THRESH,
        stability_score_thresh=config.AMG_STABILITY_SCORE_THRESH,
        min_mask_region_area=0,
    )

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        masks = amg.generate(image)

    overlay = colorize_masks(image, masks, alpha=config.OVERLAY_ALPHA)
    Image.fromarray(overlay).save(config.SAM_OUTPUT / "sam2_auto_overlay.jpg", quality=92)

    # show top masks by area
    top = sorted(masks, key=lambda x: x["area"], reverse=True)[:12]
    cols = 4
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 9))
    for ax, ann in zip(axes.ravel(), top):
        vis = image.copy()
        m = ann["segmentation"]
        vis[m] = (0.4 * vis[m] + 0.6 * np.array([255, 80, 80])).astype(np.uint8)
        ax.imshow(vis)
        ax.set_title(f"A={ann['area']} IoU={ann['predicted_iou']:.2f}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(config.SAM_OUTPUT / "sam2_auto_top_masks.png", dpi=160)
    plt.close(fig)

    report = {
        "method": "SAM2.1 automatic mask generator",
        "num_masks": len(masks),
        "points_per_side": config.AMG_POINTS_PER_SIDE,
        "pred_iou_thresh": config.AMG_PRED_IOU_THRESH,
        "stability_score_thresh": config.AMG_STABILITY_SCORE_THRESH,
        "scale": scale,
        "top_areas": [int(m["area"]) for m in top],
    }
    (config.SAM_OUTPUT / "sam2_auto_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved sam2_auto_overlay.jpg ({len(masks)} masks)")


if __name__ == "__main__":
    main()
