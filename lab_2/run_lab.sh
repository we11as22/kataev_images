#!/usr/bin/env bash
# Лабораторная 2: CNN-сегментация + SAM 2.1
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
  echo "Сначала: bash setup.sh  (или активируйте venv)" >&2
  exit 1
fi

unset PYTHONPATH
export PYTHONPATH="$(pwd):$(pwd)/scripts"

if [ ! -f checkpoints/sam2.1_hiera_tiny.pt ]; then
  echo "Чекпоинт SAM не найден — запускаю checkpoints/download.sh"
  bash checkpoints/download.sh
fi

if ! python -c "import sam2" 2>/dev/null; then
  echo "Пакет sam2 не установлен — запустите: bash setup.sh" >&2
  exit 1
fi

mkdir -p output/preview output/cnn output/sam2 dataset models

echo "=== 1/7: нарезка патчей из labelme ==="
python scripts/extract_patches.py

echo "=== 2/7: обучение CNN ==="
python scripts/train_cnn.py

echo "=== 3/7: CNN инференс (сетка 64 и 128) ==="
python scripts/infer_cnn.py --compare-all

echo "=== 4/7: SAM 2.1 box-prompts ==="
python scripts/sam2_prompt_segment.py

echo "=== 5/7: SAM 2.1 automatic masks ==="
python scripts/sam2_auto_segment.py

echo "=== 6/7: SAM 2.1 sweep параметров ==="
python scripts/sam2_sweep.py

echo "=== 7/7: сборка report.pdf ==="
python scripts/build_report.py

echo
echo "Готово."
echo "  Отчёт:     report.pdf"
echo "  CNN:       output/cnn/cnn_overlay_128.jpg"
echo "  SAM2:      output/sam2/sam2_prompt_overlay.jpg"
