"""CNN inference: grid tiles over full image -> semi-transparent colored squares."""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

import config
from train_cnn import SmallCNN

Image.MAX_IMAGE_PIXELS = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> tuple[SmallCNN, list[str], int]:
    ckpt = torch.load(config.CNN_MODEL_PATH, map_location=DEVICE, weights_only=False)
    classes = ckpt["classes"]
    input_size = ckpt["input_size"]
    model = SmallCNN(len(classes)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, classes, input_size


def load_image(max_side: int) -> tuple[np.ndarray, float]:
    return config.load_rgb_array(max_side)


def predict_tile(model: SmallCNN, tile: np.ndarray, input_size: int) -> tuple[int, float]:
    if tile.shape[0] != input_size or tile.shape[1] != input_size:
        pil = Image.fromarray(tile).resize((input_size, input_size), Image.Resampling.LANCZOS)
        tile = np.asarray(pil)
    x = torch.from_numpy(tile.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    return int(probs.argmax()), float(probs.max())


def overlay_grid(
    image: np.ndarray,
    model: SmallCNN,
    classes: list[str],
    tile_size: int,
    input_size: int,
    alpha: float,
) -> tuple[np.ndarray, dict[str, int]]:
    h, w = image.shape[:2]
    out = image.astype(np.float32).copy()
    counts = {cls: 0 for cls in classes}

    for y0 in range(0, h - tile_size + 1, tile_size):
        for x0 in range(0, w - tile_size + 1, tile_size):
            tile = image[y0 : y0 + tile_size, x0 : x0 + tile_size]
            cls_idx, _conf = predict_tile(model, tile, input_size)
            cls = classes[cls_idx]
            counts[cls] += 1
            color = np.array(config.CLASS_COLORS[cls], dtype=np.float32)
            region = out[y0 : y0 + tile_size, x0 : x0 + tile_size]
            out[y0 : y0 + tile_size, x0 : x0 + tile_size] = (1 - alpha) * region + alpha * color

    return np.clip(out, 0, 255).astype(np.uint8), counts


def draw_legend(path) -> None:
    preview = Image.new("RGB", (320, 140), "white")
    draw = ImageDraw.Draw(preview)
    for i, cls in enumerate(config.CLASSES):
        y = 10 + i * 32
        draw.rectangle([10, y, 34, y + 22], fill=config.CLASS_COLORS[cls])
        draw.text((44, y + 2), cls, fill=(0, 0, 0))
    preview.save(path)


def run(tile_size: int, out_name: str) -> dict:
    model, classes, input_size = load_model()
    image, scale = load_image(config.INFER_MAX_SIDE)
    overlay, counts = overlay_grid(
        image, model, classes, tile_size, input_size, config.OVERLAY_ALPHA
    )
    out_path = config.OUTPUT_CNN / out_name
    Image.fromarray(image).save(config.OUTPUT_PREVIEW / "source_resized.jpg", quality=92)
    Image.fromarray(overlay).save(out_path, quality=92)
    return {
        "tile_size": tile_size,
        "input_size": input_size,
        "scale": scale,
        "image_size": list(image.shape[1::-1]),
        "tile_counts": counts,
        "output": out_name,
    }


def compare_sizes(results: list[dict]) -> None:
    imgs = []
    for r in results:
        path = config.OUTPUT_CNN / r["output"]
        if path.exists():
            imgs.append((r["tile_size"], Image.open(path)))
    if len(imgs) < 2:
        return
    fig, axes = plt.subplots(1, len(imgs), figsize=(6 * len(imgs), 5))
    if len(imgs) == 1:
        axes = [axes]
    for ax, (size, im) in zip(axes, imgs):
        ax.imshow(im)
        ax.set_title(f"Тайл {size}×{size}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_CNN / "cnn_patch_size_compare.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--out", default=None)
    parser.add_argument("--compare-all", action="store_true")
    args = parser.parse_args()

    config.ensure_dirs()
    draw_legend(config.OUTPUT_CNN / "cnn_legend.png")

    if args.compare_all:
        results = []
        for size in config.CNN_TILE_SIZES:
            name = f"cnn_overlay_{size}.jpg"
            results.append(run(size, name))
            print(f"Saved {name}")
        compare_sizes(results)
        (config.OUTPUT_CNN / "cnn_infer_report.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return

    out = args.out or f"cnn_overlay_{args.tile_size}.jpg"
    stats = run(args.tile_size, out)
    (config.OUTPUT_CNN / "cnn_infer_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {config.OUTPUT_CNN / out}")


if __name__ == "__main__":
    main()
