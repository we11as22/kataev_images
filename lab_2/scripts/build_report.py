"""Сборка report.pdf."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

import config

PDF = config.ROOT / "report.pdf"


def add_title_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.72, "МФТИ", ha="center", fontsize=16)
    fig.text(0.5, 0.58, "Лабораторная работа 2", ha="center", fontsize=18, weight="bold")
    fig.text(0.5, 0.52, "Сегментация аэрофотоснимка", ha="center", fontsize=14)
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
        ax.imshow(Image.open(img_path))
        ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)


def cnn_metrics_text() -> str:
    p = config.OUTPUT_CNN / "cnn_training_report.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    return f"Лучшая точность на val: {d.get('best_val_acc', 0):.3f}. Train/val: {d.get('train_size')}/{d.get('val_size')}."


def main() -> None:
    config.ensure_dirs()
    sections = [
        (
            "Исходные данные",
            "Снимок: грунт дорога пое зеленое и убранное.tiff (5472×3648).\n"
            "Классы: дорога, зелёное поле, убранное поле, кусты.\n"
            "Разметка labelme: annotations/source.json.",
            [
                config.OUTPUT_PREVIEW / "source_preview.jpg",
                config.OUTPUT_PREVIEW / "annotations_preview.jpg",
            ],
        ),
        (
            "Датасет патчей",
            "Патчи 128×128 из прямоугольников labelme (scripts/extract_patches.py).\n"
            "Папка dataset/ — обучающая выборка для CNN.",
            [config.OUTPUT_PREVIEW / "dataset_montage.png"],
        ),
        (
            "Обучение CNN",
            "Свёрточная сеть, классификация патчей по 4 классам.\n" + cnn_metrics_text(),
            [
                config.OUTPUT_CNN / "cnn_training_curves.png",
                config.OUTPUT_CNN / "cnn_legend.png",
            ],
        ),
        (
            "Инференс CNN — сетка на всём изображении",
            "Изображение разбито на квадратные тайлы; каждый тайл классифицируется.\n"
            "Полупрозрачные квадраты — результат сегментации (как в задании).",
            [
                config.OUTPUT_CNN / "cnn_overlay_128.jpg",
                config.OUTPUT_CNN / "cnn_overlay_64.jpg",
                config.OUTPUT_CNN / "cnn_patch_size_compare.png",
            ],
        ),
        (
            "SAM 2.1 — дополнительно (box-prompts)",
            "Модель sam2.1_hiera_tiny.pt, сегментация по рамкам labelme.",
            [config.OUTPUT_SAM2 / "sam2_prompt_overlay.jpg"],
        ),
        (
            "Влияние параметров",
            "CNN: сравнение размера тайла 64 vs 128.\n"
            "SAM2: pred_iou_thresh, points_per_side.",
            [
                config.OUTPUT_CNN / "cnn_patch_size_compare.png",
                config.OUTPUT_SAM2 / "sam2_parameter_sweep.png",
            ],
        ),
    ]

    conclusions = [
        "Датасет нарезан из labelme-разметки — патчи 128×128.",
        "CNN классифицирует тайлы; на всём кадре получается карта поверхностей "
        "с полупрозрачными цветными квадратами.",
        "Тайлы 128×128 дают более грубую, но устойчивую карту; 64×64 — детальнее.",
        "SAM 2.1 использован как дополнительная программа сегментации по box-prompts.",
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
            fig.text(0.08, y, f"• {p}", ha="left", va="top", fontsize=11, wrap=True)
            y -= 0.1
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved {PDF} ({PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
