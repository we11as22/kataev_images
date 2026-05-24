#!/usr/bin/env bash
# Лабораторная 2: сегментация аэрофотоснимка через SAM 2.1
set -euo pipefail

cd "$(dirname "$0")"
source ~/envs/.venv/bin/activate

# Важно: не подмешивать старую копию SAM2 из lab_2/sources — только pip install -e из kataev_sources/sam2
unset PYTHONPATH

mkdir -p output dataset

echo "=== 1/4: нарезка патчей из labelme-разметки ==="
python extract_patches.py

echo "=== 2/4: SAM 2.1 — сегментация по box-prompts ==="
python sam2_prompt_segment.py

echo "=== 3/4: SAM 2.1 — automatic mask generator ==="
python sam2_auto_segment.py

echo "=== 4/4: сравнение параметров ==="
python sam2_sweep.py

echo
echo "Готово. Результаты: output/"
ls -lh output/
