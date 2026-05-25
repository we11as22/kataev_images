"""Plot helpers with Cyrillic-safe legend rendering."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

import config

_FONT: Path | None = None


def cyrillic_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    global _FONT
    candidates = [
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def setup_matplotlib_cyrillic() -> None:
    for path in (
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            return


def save_class_legend(path: Path) -> None:
    img = Image.new("RGB", (360, 40 + 34 * len(config.CLASSES)), "white")
    draw = ImageDraw.Draw(img)
    font = cyrillic_font(16)
    for i, cls in enumerate(config.CLASSES):
        y = 12 + i * 34
        draw.rectangle([16, y, 44, y + 22], fill=config.CLASS_COLORS[cls])
        draw.text((54, y), cls, fill=(0, 0, 0), font=font)
    img.save(path, quality=95)
