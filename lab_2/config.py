from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Официальный репозиторий SAM 2 (pip install -e ...)
SAM2_REPO = Path("/home/asudakov/projects/sharaga/kataev_sources/sam2")

IMAGES = ROOT / "IMAGES"
ANNOTATIONS = ROOT / "annotations"
DATASET = ROOT / "dataset"
OUTPUT = ROOT / "output"
CHECKPOINTS = ROOT / "checkpoints"

IMAGE_NAME = "грунт дорога пое зеленое и убранное.tiff"
IMAGE_PATH = IMAGES / IMAGE_NAME
LABELME_JSON = ANNOTATIONS / "source.json"

# SAM 2.1 Hiera-Tiny — как в README официального репозитория
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CHECKPOINT = CHECKPOINTS / "sam2.1_hiera_tiny.pt"

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

INFER_MAX_SIDE = 2048
OVERLAY_ALPHA = 0.45

AMG_POINTS_PER_SIDE = 24
AMG_PRED_IOU_THRESH = 0.82
AMG_STABILITY_SCORE_THRESH = 0.92
