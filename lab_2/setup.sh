#!/usr/bin/env bash
# Подготовка окружения: зависимости, SAM 2, чекпоинт.
set -euo pipefail

cd "$(dirname "$0")"

activate_venv() {
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    return 0
  fi
  if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    return 0
  fi
  if [ -f "${HOME}/envs/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/envs/.venv/bin/activate"
    return 0
  fi
  echo "Активируйте venv или создайте: python -m venv .venv && source .venv/bin/activate" >&2
  exit 1
}

activate_venv

echo "=== Python deps (lab_2/requirements.txt) ==="
python -m pip install -q -r requirements.txt

echo "=== SAM 2 package ==="
if ! python -c "import sam2" 2>/dev/null; then
  SAM2_DIR="${SAM2_REPO:-$(dirname "$PWD")/segment-anything-2}"
  if [ ! -d "$SAM2_DIR" ]; then
    echo "Клонируйте SAM 2 и установите:" >&2
    echo "  git clone https://github.com/facebookresearch/segment-anything-2.git \"$SAM2_DIR\"" >&2
    echo "  pip install -e \"$SAM2_DIR\"" >&2
    echo "  export SAM2_REPO=\"$SAM2_DIR\"  # опционально" >&2
    exit 1
  fi
  python -m pip install -q -e "$SAM2_DIR"
fi

echo "=== SAM 2.1 checkpoint ==="
bash sam/checkpoints/download.sh

echo
echo "Setup OK."
echo "  bash run_all.sh                    # CNN + SAM + отчёты"
echo "  python scripts/verify_submission.py # проверка перед сдачей"
