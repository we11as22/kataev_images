#!/usr/bin/env bash
# SAM 2.1 — отдельный пайплайн (папка sam/)
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$LAB_ROOT"

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
export PYTHONPATH="${LAB_ROOT}:${LAB_ROOT}/scripts:${LAB_ROOT}/sam/scripts"

if [ ! -f sam/checkpoints/sam2.1_hiera_tiny.pt ]; then
  echo "Чекпоинт SAM — bash sam/checkpoints/download.sh"
  bash sam/checkpoints/download.sh
fi

if ! python -c "import sam2" 2>/dev/null; then
  echo "Пакет sam2 не установлен — bash setup.sh" >&2
  exit 1
fi

mkdir -p sam/output

echo "=== 1/4: SAM box-prompts ==="
python sam/scripts/sam2_prompt_segment.py

echo "=== 2/4: SAM automatic masks ==="
python sam/scripts/sam2_auto_segment.py

echo "=== 3/4: SAM sweep ==="
python sam/scripts/sam2_sweep.py

echo "=== 4/4: sam/report.pdf ==="
python sam/scripts/build_sam_report.py

echo
echo "Готово: sam/report.pdf, sam/output/"
