"""Сборка report.pdf — CNN-сегментация (GitHub-safe PDF)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_LAB = Path(__file__).resolve().parent.parent
for p in (_LAB, _LAB / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config
from pdf_report import image_page, save_pdf, text_page, title_page

PDF = config.ROOT / "report.pdf"


def dataset_stats_text() -> str:
    p = config.OUTPUT_PREVIEW / "dataset_summary.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    parts = [f"{k}: {v}" for k, v in d.get("counts", {}).items()]
    return "Патчей: " + ", ".join(parts) + f". Всего: {d.get('total', 0)}."


def cnn_metrics_text() -> str:
    p = config.OUTPUT_CNN / "cnn_training_report.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    return (
        f"Val accuracy: {d.get('best_val_acc', 0):.3f} (эпоха {d.get('best_epoch')}). "
        f"Train/val: {d.get('train_size')}/{d.get('val_size')}."
    )


def main() -> None:
    config.ensure_dirs()
    pages = [
        title_page(
            [
                "МФТИ",
                "Лабораторная работа 2",
                "Сегментация аэрофотоснимка (CNN)",
                "Судаков Алексей",
                "Долгопрудный, 2026",
            ]
        ),
    ]

    sections = [
        (
            "Исходные данные",
            "Снимок 5472x3648. Классы: дорога, зелёное поле, убранное поле, грунт.\n"
            "Разметка: annotations/source.json, 39 прямоугольников 64x64.",
            config.OUTPUT_PREVIEW / "source_preview.jpg",
        ),
        (
            "Разметка labelme",
            "Прямоугольники 64x64 на исходном кадре.",
            config.OUTPUT_PREVIEW / "annotations_preview.jpg",
        ),
        (
            "Датасет патчей",
            dataset_stats_text(),
            config.OUTPUT_PREVIEW / "dataset_montage.png",
        ),
        (
            "Обучение CNN",
            cnn_metrics_text(),
            config.OUTPUT_CNN / "cnn_training_curves.png",
        ),
        (
            "Легенда классов",
            "",
            config.OUTPUT_CNN / "cnn_legend.png",
        ),
        (
            "Инференс: тайл 64x64",
            "Полупрозрачная сетка классификации CNN.",
            config.OUTPUT_CNN / "cnn_overlay_64.jpg",
        ),
        (
            "Инференс: тайл 128x128",
            "Более крупные тайлы — грубее карта.",
            config.OUTPUT_CNN / "cnn_overlay_128.jpg",
        ),
        (
            "Сравнение 64 vs 128",
            "",
            config.OUTPUT_CNN / "cnn_patch_size_compare.png",
        ),
    ]

    for title, caption, img in sections:
        page = image_page(title, caption, img)
        if page:
            pages.append(page)

    pages.append(
        text_page(
            "Выводы",
            [
                "Датасет: 39 патчей 64x64 с ручной проверкой.",
                "CNN классифицирует тайлы; на кадре — цветная карта классов.",
                "Тайл 64x64 детальнее, чем 128x128.",
                "SAM 2.1 — отдельный отчёт sam/report.pdf.",
            ],
        )
    )

    save_pdf(pages, PDF)
    print(f"Saved {PDF} ({PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
