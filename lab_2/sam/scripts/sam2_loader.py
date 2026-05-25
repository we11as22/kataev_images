"""Load SAM 2.1 via the official build_sam2 API — no checkpoint patching."""
from __future__ import annotations

import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import config


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_model(device: str | None = None):
    device = device or _device()
    return build_sam2(
        config_file=config.SAM2_CONFIG,
        ckpt_path=str(config.SAM2_CHECKPOINT),
        device=device,
    )


def build_predictor(device: str | None = None) -> SAM2ImagePredictor:
    return SAM2ImagePredictor(_build_model(device))


def build_amg(device: str | None = None, **amg_kwargs) -> SAM2AutomaticMaskGenerator:
    return SAM2AutomaticMaskGenerator(_build_model(device), **amg_kwargs)
