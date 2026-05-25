"""Paths and hyperparameters for lab 2."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

IMAGES = ROOT / "IMAGES"
ANNOTATIONS = ROOT / "annotations"
DATASET = ROOT / "dataset"
MODELS = ROOT / "models"
OUTPUT = ROOT / "output"
OUTPUT_PREVIEW = OUTPUT / "preview"
OUTPUT_CNN = OUTPUT / "cnn"
SAM_ROOT = ROOT / "sam"
SAM_OUTPUT = SAM_ROOT / "output"
SAM_CHECKPOINTS = SAM_ROOT / "checkpoints"
CHECKPOINTS = SAM_CHECKPOINTS  # legacy alias

IMAGE_NAME = "грунт дорога пое зеленое и убранное.tiff"
IMAGE_PATH = IMAGES / IMAGE_NAME
LABELME_JSON = ANNOTATIONS / "source.json"

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CHECKPOINT = SAM_CHECKPOINTS / "sam2.1_hiera_tiny.pt"
SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
)

PATCH_SIZE = 64
SEED = 42

CLASSES = ["Дорога", "Зелёное поле", "Убранное поле", "Грунт"]
CLASS_COLORS = {
    "Дорога": (210, 120, 40),
    "Зелёное поле": (40, 160, 50),
    "Убранное поле": (200, 190, 150),
    "Грунт": (140, 90, 50),
}

CNN_EPOCHS = 50
CNN_BATCH_SIZE = 16
CNN_LR = 5e-4
CNN_WEIGHT_DECAY = 1e-4
CNN_LABEL_SMOOTHING = 0.02
CNN_TRAIN_RATIO = 0.8
CNN_EARLY_STOP_PATIENCE = 15
CNN_MIN_EPOCHS = 12
CNN_MODEL_PATH = MODELS / "cnn_best.pt"
CNN_INPUT_SIZE = 64

INFER_MAX_SIDE = 2048
OVERLAY_ALPHA = 0.5
CNN_TILE_SIZES = [64, 128]

AMG_POINTS_PER_SIDE = 24
AMG_PRED_IOU_THRESH = 0.82
AMG_STABILITY_SCORE_THRESH = 0.92


def ensure_dirs() -> None:
    for path in (
        DATASET,
        MODELS,
        OUTPUT,
        OUTPUT_PREVIEW,
        OUTPUT_CNN,
        SAM_OUTPUT,
        SAM_CHECKPOINTS,
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_rgb_array(max_side: int | None = None) -> tuple["np.ndarray", float]:
    """Load image, optionally downscale — same path for training patches and inference."""
    import numpy as np
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    max_side = max_side or INFER_MAX_SIDE
    with Image.open(IMAGE_PATH) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        return np.asarray(im).copy(), scale


def sam2_repo_hint() -> Path:
    """Suggested clone location for segment-anything-2 (documentation only)."""
    env = os.environ.get("SAM2_REPO")
    if env:
        return Path(env).expanduser()
    return ROOT.parent / "segment-anything-2"
