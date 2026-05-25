"""Проверка готовности lab_2 к сдаче."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_LAB = Path(__file__).resolve().parent.parent
for p in (_LAB, _LAB / "scripts"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

import config

ROOT = config.ROOT
ERRORS: list[str] = []
WARNINGS: list[str] = []


def need(path: Path, label: str) -> None:
    if not path.exists():
        ERRORS.append(f"Нет файла: {label} ({path.relative_to(ROOT)})")
    elif path.stat().st_size == 0:
        ERRORS.append(f"Пустой файл: {label}")


def check_pdf(path: Path, label: str) -> None:
    need(path, label)
    if path.is_file() and path.read_bytes()[:5] != b"%PDF-":
        ERRORS.append(f"Битый PDF: {label}")


def check_image(path: Path, label: str) -> None:
    if not path.exists():
        ERRORS.append(f"Нет изображения: {label}")
        return
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
    except Exception as exc:
        ERRORS.append(f"Не открывается: {label} — {exc}")


def main() -> int:
    # исходники
    need(config.IMAGE_PATH, "исходный снимок")
    need(config.LABELME_JSON, "разметка labelme")
    need(config.DATASET_ZIP, "архив датасета")

    meta = json.loads(config.LABELME_JSON.read_text(encoding="utf-8"))
    n = len(meta.get("shapes", []))
    if n != 39:
        ERRORS.append(f"Ожидалось 39 прямоугольников в source.json, есть {n}")

    # датасет
    total = 0
    for cls in config.CLASSES:
        files = list((config.DATASET / cls).glob("*.png"))
        total += len(files)
        if not files:
            ERRORS.append(f"Нет патчей класса «{cls}» в dataset/")
    if total != 39:
        ERRORS.append(f"Ожидалось 39 патчей в dataset/, есть {total}")

    need(config.CNN_MODEL_PATH, "модель CNN")

    # отчёты
    check_pdf(ROOT / "report.pdf", "report.pdf (CNN)")
    check_pdf(config.SAM_ROOT / "report.pdf", "sam/report.pdf (SAM)")

    # ключевые картинки
    for p, name in [
        (config.OUTPUT_CNN / "cnn_overlay_64.jpg", "CNN overlay 64"),
        (config.OUTPUT_CNN / "cnn_training_curves.png", "кривые обучения"),
        (config.OUTPUT_PREVIEW / "overview_rectangles_64.jpg", "разметка overview"),
        (config.OUTPUT_PREVIEW / "dataset_montage.png", "montage датасета"),
        (config.SAM_OUTPUT / "sam2_prompt_overlay.jpg", "SAM overlay"),
        (config.SAM_OUTPUT / "sam2_parameter_sweep.png", "SAM sweep"),
    ]:
        check_image(p, name)

    # метрики
    tr = config.OUTPUT_CNN / "cnn_training_report.json"
    if tr.exists():
        d = json.loads(tr.read_text())
        acc = d.get("best_val_acc", 0)
        if acc < 0.9:
            WARNINGS.append(f"CNN val accuracy низкая: {acc:.3f}")
        print(f"CNN: val_acc={acc:.3f}, epoch={d.get('best_epoch')}, train/val={d.get('train_size')}/{d.get('val_size')}")

    pr = config.SAM_OUTPUT / "sam2_prompt_report.json"
    if pr.exists():
        d = json.loads(pr.read_text())
        print(f"SAM: {d.get('num_prompts')} box-prompts")

    if WARNINGS:
        print("\nПредупреждения:")
        for w in WARNINGS:
            print(f"  ! {w}")

    if ERRORS:
        print("\nОШИБКИ:")
        for e in ERRORS:
            print(f"  x {e}")
        return 1

    print("\nOK — готово к сдаче.")
    print("  report.pdf")
    print("  sam/report.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
