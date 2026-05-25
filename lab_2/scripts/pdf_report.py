"""PDF reports compatible with GitHub viewer (PIL JPEG pages, no matplotlib PDF)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from plot_utils import cyrillic_font

# A4-ish at 150 dpi
PAGE_W, PAGE_H = 1240, 1754
MARGIN = 48


def _new_page() -> Image.Image:
    return Image.new("RGB", (PAGE_W, PAGE_H), "white")


def text_page(title: str, lines: list[str]) -> Image.Image:
    page = _new_page()
    draw = ImageDraw.Draw(page)
    title_font = cyrillic_font(28)
    body_font = cyrillic_font(18)
    y = 120
    draw.text((PAGE_W // 2, y), title, fill=(0, 0, 0), font=title_font, anchor="mm")
    y += 70
    for line in lines:
        draw.text((MARGIN, y), line, fill=(30, 30, 30), font=body_font)
        y += 32
    return page


def title_page(lines: list[str]) -> Image.Image:
    page = _new_page()
    draw = ImageDraw.Draw(page)
    font_l = cyrillic_font(32)
    font_m = cyrillic_font(22)
    font_s = cyrillic_font(18)
    ys = [420, 520, 580, 720, 1500]
    fonts = [font_l, font_m, font_m, font_s, font_s]
    for text, y, font in zip(lines, ys, fonts):
        draw.text((PAGE_W // 2, y), text, fill=(0, 0, 0), font=font, anchor="mm")
    return page


def image_page(title: str, caption: str, image_path: Path) -> Image.Image | None:
    if not image_path.is_file():
        return None
    page = _new_page()
    draw = ImageDraw.Draw(page)
    title_font = cyrillic_font(22)
    cap_font = cyrillic_font(16)
    draw.text((PAGE_W // 2, 56), title, fill=(0, 0, 0), font=title_font, anchor="mm")
    y_cap = 96
    for line in caption.split("\n"):
        if line.strip():
            draw.text((MARGIN, y_cap), line.strip(), fill=(50, 50, 50), font=cap_font)
            y_cap += 24
    top = max(y_cap + 12, 120)
    avail_h = PAGE_H - top - MARGIN
    avail_w = PAGE_W - 2 * MARGIN
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im.thumbnail((avail_w, avail_h), Image.Resampling.LANCZOS)
        x = (PAGE_W - im.width) // 2
        y = top + (avail_h - im.height) // 2
        page.paste(im, (x, y))
    return page


def save_pdf(pages: list[Image.Image], path: Path) -> None:
    if not pages:
        raise RuntimeError("No pages for PDF")
    path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(
        path,
        save_all=True,
        append_images=pages[1:],
        format="PDF",
        resolution=150.0,
        quality=92,
    )
