#!/usr/bin/env bash
# Лабораторная 2: CNN-сегментация (+ SAM в sam/)
set -euo pipefail

cd "$(dirname "$0")"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  :
elif [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [ -f "${HOME}/envs/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/envs/.venv/bin/activate"
else
  echo "Сначала: bash setup.sh" >&2
  exit 1
fi

unset PYTHONPATH
export PYTHONPATH="$(pwd):$(pwd)/scripts"

mkdir -p output/preview output/cnn dataset models

echo "=== 1/4: импорт датасета 64×64 ==="
python scripts/import_segmentation_64_dataset.py

echo "=== 2/4: обучение CNN ==="
python scripts/train_cnn.py

echo "=== 3/4: CNN инференс (64 и 128) ==="
python scripts/infer_cnn.py --compare-all

echo "=== 4/4: report.pdf (CNN) ==="
python scripts/build_report.py

echo
echo "CNN готово:"
echo "  report.pdf"
echo "  output/cnn/cnn_overlay_64.jpg"
echo
echo "SAM (отдельно): bash sam/run_sam.sh"
