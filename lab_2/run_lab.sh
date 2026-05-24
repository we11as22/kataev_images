#!/usr/bin/env bash
# Лабораторная 2: CNN-сегментация + SAM 2.1
set -euo pipefail

cd "$(dirname "$0")"
source ~/envs/.venv/bin/activate
unset PYTHONPATH

mkdir -p output dataset models

echo "=== 1/7: нарезка патчей из labelme ==="
python extract_patches.py

echo "=== 2/7: обучение CNN ==="
python train_cnn.py

echo "=== 3/7: CNN инференс (сетка 64 и 128) ==="
python infer_cnn.py --compare-all

echo "=== 4/7: SAM 2.1 box-prompts ==="
python sam2_prompt_segment.py

echo "=== 5/7: SAM 2.1 automatic masks ==="
python sam2_auto_segment.py

echo "=== 6/7: SAM 2.1 sweep параметров ==="
python sam2_sweep.py

echo "=== 7/7: сборка report.pdf ==="
python build_report.py

echo
echo "Готово. Отчёт: report.pdf"
ls -lh report.pdf output/cnn_overlay_*.jpg 2>/dev/null || true
