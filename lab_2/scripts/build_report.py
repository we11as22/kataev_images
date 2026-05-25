"""Сборка report.pdf — CNN-сегментация."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

import config
from plot_utils import setup_matplotlib_cyrillic

PDF = config.ROOT / "report.pdf"
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
    fig.text(0.5, 0.58, "Лабораторная работа 2", ha="center", fontsize=18, weight="bold")
    fig.text(0.5, 0.52, "Сегментация аэрофотоснимка (CNN)", ha="center", fontsize=14)
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


def dataset_stats_text() -> str:
    p = config.OUTPUT_PREVIEW / "dataset_summary.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    lines = [f"{k}: {v}" for k, v in d.get("counts", {}).items()]
    return "Патчей по классам: " + ", ".join(lines) + f". Всего: {d.get('total', 0)}."


def cnn_metrics_text() -> str:
    p = config.OUTPUT_CNN / "cnn_training_report.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    return (
        f"Лучшая val accuracy: {d.get('best_val_acc', 0):.3f} (эпоха {d.get('best_epoch')}).\n"
        f"Train/val: {d.get('train_size')}/{d.get('val_size')}."
    )


def main() -> None:
    setup_matplotlib_cyrillic()
    config.ensure_dirs()
    sections = [
        (
            "Исходные данные",
            "Снимок: грунт дорога пое зеленое и убранное.tiff (5472×3648).\n"
            "Классы: дорога, зелёное поле, убранное поле, грунт.\n"
            "Разметка: 39 прямоугольников 64×64 (ручная проверка).",
            [
                config.OUTPUT_PREVIEW / "source_preview.jpg",
                config.OUTPUT_PREVIEW / "overview_rectangles_64.jpg",
                config.OUTPUT_PREVIEW / "preview_checked_patches.jpg",
            ],
        ),
        (
            "Датасет патчей",
            dataset_stats_text() + "\nПатчи 64x64 в dataset/ для обучения CNN.",
            [config.OUTPUT_PREVIEW / "dataset_montage.png"],
        ),
        (
            "Обучение CNN",
            "Свёрточная сеть, 4 класса.\n" + cnn_metrics_text(),
            [
                config.OUTPUT_CNN / "cnn_training_curves.png",
                config.OUTPUT_CNN / "cnn_legend.png",
            ],
        ),
        (
            "Инференс — сетка на всём изображении",
            "Кадр разбит на тайлы 64×64 и 128×128; каждый классифицируется CNN.\n"
            "Полупрозрачные квадраты — результат сегментации.",
            [
                config.OUTPUT_CNN / "cnn_overlay_64.jpg",
                config.OUTPUT_CNN / "cnn_overlay_128.jpg",
                config.OUTPUT_CNN / "cnn_patch_size_compare.png",
            ],
        ),
        (
            "Сравнение размера тайла",
            "64×64 — детальнее; 128×128 — грубее, меньше тайлов.",
            [config.OUTPUT_CNN / "cnn_patch_size_compare.png"],
        ),
    ]

    conclusions = [
        "Датасет: 39 патчей 64×64 с ручной проверкой.",
        "CNN классифицирует тайлы; на всём кадре — цветная карта классов.",
        "Тайл 64×64 даёт более детальную сегментацию, чем 128×128.",
        "SAM 2.1 — в отдельной папке sam/ (см. sam/report.pdf).",
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
