"""Сборка report.pdf без emacs/LaTeX — только matplotlib."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
PDF = ROOT / "report.pdf"


def add_title_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.72, "МФТИ", ha="center", fontsize=16)
    fig.text(0.5, 0.58, "Лабораторная работа 2", ha="center", fontsize=18, weight="bold")
    fig.text(0.5, 0.52, "Сегментация аэрофотоснимка (SAM 2.1)", ha="center", fontsize=14)
    fig.text(0.5, 0.38, "Судаков Алексей", ha="center", fontsize=14)
    fig.text(0.5, 0.12, "Долгопрудный, 2026", ha="center", fontsize=12)
    pdf.savefig(fig)
    plt.close(fig)


def add_section(pdf: PdfPages, title: str, text: str, images: list[Path]) -> None:
    for i, img_path in enumerate(images):
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        y = 0.94
        if i == 0:
            fig.text(0.5, y, title, ha="center", fontsize=14, weight="bold")
            y -= 0.06
            for line in text.split("\n"):
                fig.text(0.08, y, line, ha="left", va="top", fontsize=10, wrap=True)
                y -= 0.045
        img = Image.open(img_path)
        ax = fig.add_axes([0.06, 0.08, 0.88, 0.55 if i == 0 else 0.82])
        ax.imshow(img)
        ax.axis("off")
        if len(images) > 1:
            fig.text(0.5, 0.02, img_path.name, ha="center", fontsize=8, color="gray")
        pdf.savefig(fig)
        plt.close(fig)


def add_text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.94, title, ha="center", fontsize=14, weight="bold")
    y = 0.86
    for p in paragraphs:
        fig.text(0.08, y, p, ha="left", va="top", fontsize=11, wrap=True)
        y -= 0.08 + 0.02 * p.count("\n")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    sections = [
        (
            "Исходные данные",
            "Снимок: грунт дорога пое зеленое и убранное.tiff (5472×3648).\n"
            "Классы: дорога, зелёное поле, убранное поле, кусты.\n"
            "Разметка: labelme (annotations/source.json).",
            [OUT / "source_preview.jpg", OUT / "annotations_preview.jpg"],
        ),
        (
            "SAM 2.1 — сегментация по box-prompts",
            "Модель sam2.1_hiera_tiny.pt. Box-prompts из labelme → карта классов\n"
            "с полупрозрачной цветной наложенной сегментацией.",
            [OUT / "sam2_prompt_overlay.jpg", OUT / "sam2_prompt_classes.png", OUT / "legend.png"],
        ),
        (
            "SAM 2.1 — automatic mask generator",
            "Automatic Mask Generator: маски по сетке точек на всём изображении.",
            [OUT / "sam2_auto_overlay.jpg", OUT / "sam2_auto_top_masks.png"],
        ),
        (
            "Влияние параметров",
            "Сравнение pred_iou_thresh, stability_score_thresh, points_per_side.",
            [OUT / "sam2_parameter_sweep.png"],
        ),
    ]

    with PdfPages(PDF) as pdf:
        add_title_page(pdf)
        for title, text, images in sections:
            existing = [p for p in images if p.exists()]
            if not existing:
                continue
            add_section(pdf, title, text, existing)
        add_text_page(
            pdf,
            "Выводы",
            [
                "SAM 2.1 по box-prompts из labelme даёт семантическую карту поверхностей "
                "с полупрозрачной цветной сегментацией.",
                "При увеличении порогов pred_iou и stability число масок уменьшается — "
                "остаются более уверенные регионы.",
                "Увеличение points_per_side повышает детализацию, но увеличивает время работы.",
                "Датасет: 228 патчей 128×128 (папка dataset/), нарезанных из labelme-разметки.",
            ],
        )

    print(f"Saved {PDF} ({PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
