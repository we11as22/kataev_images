# Lab 2 — сегментация аэрофотоснимка

CNN-классификация патчей + SAM 2.1 (отдельная папка `sam/`).

**Отчёты для сдачи:**
- [`report.pdf`](report.pdf) — CNN-сегментация
- [`sam/report.pdf`](sam/report.pdf) — SAM 2.1

## Структура

```
lab_2/
├── run_all.sh              # CNN + SAM + оба отчёта
├── run_lab.sh              # только CNN → report.pdf
├── report.pdf
├── IMAGES/                 # исходный .tiff (5472×3648)
├── annotations/source.json # labelme, 39 прямоугольников 64×64
├── dataset/                # патчи 64×64 для обучения CNN
├── models/cnn_best.pt
├── output/
│   ├── preview/            # превью разметки, montage
│   └── cnn/                # overlay, кривые, метрики
└── sam/
    ├── run_sam.sh
    ├── report.pdf
    ├── checkpoints/        # sam2.1_hiera_tiny.pt (скачать, не в git)
    └── output/
```

## Воспроизведение с нуля

```bash
cd lab_2
python -m venv .venv && source .venv/bin/activate
pip install -U pip

git clone https://github.com/facebookresearch/segment-anything-2.git ../segment-anything-2
pip install -e ../segment-anything-2

bash setup.sh      # deps + чекпоинт SAM
bash run_all.sh    # всё → report.pdf + sam/report.pdf
python scripts/verify_submission.py
```

## Классы

| Класс | Цвет overlay |
|-------|--------------|
| Дорога | оранжевый |
| Зелёное поле | зелёный |
| Убранное поле | бежевый |
| Грунт | коричневый |

## Датасет

39 патчей 64×64 в `dataset/` (ручная разметка в `annotations/source.json`).

| Класс | Патчей |
|-------|--------|
| Дорога | 10 |
| Зелёное поле | 12 |
| Убранное поле | 12 |
| Грунт | 5 |

## Что в отчётах

**report.pdf (CNN):** исходный снимок, разметка, датасет, кривые обучения, overlay-сетка 64×64 и 128×128, сравнение размеров тайла.

**sam/report.pdf:** box-prompts SAM 2.1 по разметке, automatic mask generator, сравнение параметров AMG.

## Примечания

- Чекпоинт SAM (~149 MB) не в git: `bash sam/checkpoints/download.sh`
- Перед SAM: `unset PYTHONPATH` (делает `run_sam.sh` автоматически)
