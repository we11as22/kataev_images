#!/usr/bin/env bash
# Полный пайплайн: CNN + SAM + оба отчёта
set -euo pipefail
cd "$(dirname "$0")"
bash run_lab.sh
bash sam/run_sam.sh
python scripts/verify_submission.py
