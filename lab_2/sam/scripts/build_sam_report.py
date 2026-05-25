"""Сборка sam/report.pdf — SAM 2.1 (GitHub-safe PDF)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_LAB = Path(__file__).resolve().parents[2]
for p in (_LAB, _LAB / "scripts", _LAB / "sam" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config
from pdf_report import image_page, save_pdf, text_page, title_page

PDF = config.SAM_ROOT / "report.pdf"


def prompt_metrics() -> str:
    p = config.SAM_OUTPUT / "sam2_prompt_report.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    return f"Box-prompts: {d.get('num_prompts', 0)} масок из 39 прямоугольников labelme."


def auto_metrics() -> str:
    p = config.SAM_OUTPUT / "sam2_auto_report.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    return f"Automatic masks: {d.get('num_masks', 0)}."


def main() -> None:
    config.ensure_dirs()
    pages = [
        title_page(
            [
                "МФТИ",
                "Лабораторная работа 2 — SAM 2.1",
                "Дополнительная сегментация",
                "Судаков Алексей",
                "Долгопрудный, 2026",
            ]
        ),
    ]

    sections = [
        (
            "Разметка для SAM",
            "Промпты из annotations/source.json.",
            config.OUTPUT_PREVIEW / "annotations_preview.jpg",
        ),
        (
            "Box-prompts на кадре",
            "",
            config.SAM_OUTPUT / "sam2_boxes_preview.jpg",
        ),
        (
            "SAM 2.1 — box prompts",
            prompt_metrics(),
            config.SAM_OUTPUT / "sam2_prompt_overlay.jpg",
        ),
        (
            "Классы SAM",
            "",
            config.SAM_OUTPUT / "sam2_prompt_classes.png",
        ),
        (
            "Automatic mask generator",
            auto_metrics(),
            config.SAM_OUTPUT / "sam2_auto_overlay.jpg",
        ),
        (
            "Топ масок AMG",
            "",
            config.SAM_OUTPUT / "sam2_auto_top_masks.png",
        ),
        (
            "Сравнение параметров AMG",
            "pred_iou_thresh, points_per_side.",
            config.SAM_OUTPUT / "sam2_parameter_sweep.png",
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
                "SAM 2.1 — дополнительный метод сегментации.",
                "Box-prompts из ручной разметки 64x64.",
                "AMG даёт обзор объектов на кадре.",
                "Параметры AMG влияют на число масок.",
            ],
        )
    )

    save_pdf(pages, PDF)
    print(f"Saved {PDF} ({PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
