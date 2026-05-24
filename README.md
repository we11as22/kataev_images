# kataev_images — Судаков Алексей

Лабораторные работы по обработке и сегментации аэрофотоснимков (МФТИ).

| Работа | Отчёт | Описание |
|--------|-------|----------|
| **Lab 1** | [lab_1/report.pdf](lab_1/report.pdf) | Гистограммы, статистика RGB, преобразования изображений |
| **Lab 2** | [lab_2/report.pdf](lab_2/report.pdf) | Сегментация с SAM 2.1 Hiera-Tiny, labelme, датасет патчей |

## Lab 2 — быстрый просмотр результатов

- Сегментация по разметке: [lab_2/output/sam2_prompt_overlay.jpg](lab_2/output/sam2_prompt_overlay.jpg)
- Автоматические маски: [lab_2/output/sam2_auto_overlay.jpg](lab_2/output/sam2_auto_overlay.jpg)
- Влияние параметров: [lab_2/output/sam2_parameter_sweep.png](lab_2/output/sam2_parameter_sweep.png)

## Lab 2 — воспроизведение

```bash
source ~/envs/.venv/bin/activate

# SAM 2 (один раз)
cd /home/asudakov/projects/sharaga/kataev_sources/sam2
pip install -e .

# Чекпоинт (~149 MB) — положить в lab_2/checkpoints/
# см. lab_2/checkpoints/README.md

cd /home/asudakov/projects/sharaga/kataev_images/lab_2
bash run_lab.sh          # весь пайплайн
python build_report.py   # пересобрать report.pdf
```

Перед запуском: `unset PYTHONPATH` (или это делает `run_lab.sh`).

Исходное изображение lab_2: `lab_2/IMAGES/грунт дорога пое зеленое и убранное.tiff`  
Разметка: `lab_2/annotations/source.json`  
Датасет патчей 128×128: `lab_2/dataset/` (228 файлов)
