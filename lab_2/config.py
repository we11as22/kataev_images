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
OUTPUT_SAM2 = OUTPUT / "sam2"
CHECKPOINTS = ROOT / "checkpoints"

IMAGE_NAME = "грунт дорога пое зеленое и убранное.tiff"
IMAGE_PATH = IMAGES / IMAGE_NAME
LABELME_JSON = ANNOTATIONS / "source.json"

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CHECKPOINT = CHECKPOINTS / "sam2.1_hiera_tiny.pt"
SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
)

PATCH_SIZE = 128
PATCHES_PER_BOX = 12
SEED = 42

CLASSES = ["Дорога", "Зелёное поле", "Убранное поле", "Кусты"]
CLASS_COLORS = {
    "Дорога": (210, 120, 40),
    "Зелёное поле": (40, 160, 50),
    "Убранное поле": (200, 190, 150),
    "Кусты": (30, 100, 30),
}

CNN_EPOCHS = 30
CNN_BATCH_SIZE = 32
CNN_LR = 1e-3
CNN_TRAIN_RATIO = 0.75
CNN_MODEL_PATH = MODELS / "cnn_best.pt"
CNN_INPUT_SIZE = 128

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
        OUTPUT_SAM2,
        CHECKPOINTS,
    ):
        path.mkdir(parents=True, exist_ok=True)


def sam2_repo_hint() -> Path:
    """Suggested clone location for segment-anything-2 (documentation only)."""
    env = os.environ.get("SAM2_REPO")
    if env:
        return Path(env).expanduser()
    return ROOT.parent / "segment-anything-2"
