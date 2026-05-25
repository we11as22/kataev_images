"""Сборка sam/report.pdf — отдельный отчёт по SAM 2.1."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

import config
from plot_utils import setup_matplotlib_cyrillic

PDF = config.SAM_ROOT / "report.pdf"
PDF_IMAGE_MAX = 1400


def load_page_image(path: Path):
    import numpy as np

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > PDF_IMAGE_MAX:
            scale = PDF_IMAGE_MAX / max(w, h)
            im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        return np.asarray(im)


def add_title_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.72, "МФТИ", ha="center", fontsize=16)
    fig.text(0.5, 0.58, "Лабораторная работа 2 — SAM 2.1", ha="center", fontsize=18, weight="bold")
    fig.text(0.5, 0.52, "Дополнительная сегментация", ha="center", fontsize=14)
    fig.text(0.5, 0.38, "Судаков Алексей", ha="center", fontsize=14)
    fig.text(0.5, 0.12, "Долгопрудный, 2026", ha="center", fontsize=12)
    pdf.savefig(fig)
    plt.close(fig)


def add_section(pdf: PdfPages, title: str, text: str, images: list[Path]) -> None:
    existing = [p for p in images if p.exists()]
    if not existing:
        return
    for i, img_path in enumerate(existing):
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        y = 0.94
        if i == 0:
            fig.text(0.5, y, title, ha="center", fontsize=14, weight="bold")
            y -= 0.06
            for line in text.split("\n"):
                fig.text(0.08, y, line, ha="left", va="top", fontsize=10)
                y -= 0.045
        ax = fig.add_axes([0.06, 0.08, 0.88, 0.55 if i == 0 else 0.82])
        ax.imshow(load_page_image(img_path))
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)


def prompt_metrics() -> str:
    p = config.SAM_OUTPUT / "sam2_prompt_report.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    return f"Box-prompts: {d.get('num_prompts', 0)} масок из labelme (39 прямоугольников)."


def auto_metrics() -> str:
    p = config.SAM_OUTPUT / "sam2_auto_report.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    return f"Automatic masks: {d.get('num_masks', 0)} масок."


def main() -> None:
    setup_matplotlib_cyrillic()
    config.ensure_dirs()

    sections = [
        (
            "Разметка для SAM",
            "Промпты — прямоугольники 64×64 из annotations/source.json.\n"
            "Классы: дорога, зелёное поле, убранное поле, грунт.",
            [
                config.OUTPUT_PREVIEW / "overview_rectangles_64.jpg",
                config.SAM_OUTPUT / "sam2_boxes_preview.jpg",
            ],
        ),
        (
            "SAM 2.1 — box prompts",
            "Segment Anything 2.1 (hiera_tiny) по box из labelme.\n" + prompt_metrics(),
            [
                config.SAM_OUTPUT / "sam2_legend.png",
                config.SAM_OUTPUT / "sam2_prompt_overlay.jpg",
                config.SAM_OUTPUT / "sam2_prompt_classes.png",
            ],
        ),
        (
            "SAM 2.1 — automatic mask generator",
            "Автоматическая генерация масок без промптов.\n" + auto_metrics(),
            [
                config.SAM_OUTPUT / "sam2_auto_overlay.jpg",
                config.SAM_OUTPUT / "sam2_auto_top_masks.png",
            ],
        ),
        (
            "Сравнение параметров AMG",
            "pred_iou_thresh и points_per_side.",
            [config.SAM_OUTPUT / "sam2_parameter_sweep.png"],
        ),
    ]

    conclusions = [
        "SAM 2.1 использован как дополнительный метод сегментации.",
        "Box-prompts берутся из ручной разметки (39 прямоугольников 64×64).",
        "Automatic mask generator даёт обзор всех объектов на кадре.",
        "Параметры AMG влияют на число и качество масок.",
    ]

    with PdfPages(PDF) as pdf:
        add_title_page(pdf)
        for title, text, images in sections:
            add_section(pdf, title, text, images)
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        fig.text(0.5, 0.94, "Выводы", ha="center", fontsize=14, weight="bold")
        y = 0.86
        for p in conclusions:
            fig.text(0.08, y, f"• {p}", ha="left", va="top", fontsize=11)
            y -= 0.1
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved {PDF} ({PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
