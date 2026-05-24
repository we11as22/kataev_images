# Lab 2 — сегментация аэрофотоснимка

CNN-классификация патчей + SAM 2.1 (дополнительно).  
Отчёт: [report.pdf](report.pdf)

## Структура

```
lab_2/
├── README.md
├── config.py              # пути и гиперпараметры
├── requirements.txt       # numpy, torch, matplotlib, …
├── setup.sh               # deps + SAM2 + скачивание чекпоинта
├── run_lab.sh             # полный пайплайн
├── report.pdf
├── IMAGES/                # исходный .tiff
├── annotations/           # labelme source.json
├── dataset/               # патчи 128×128 (генерируется)
├── models/                # cnn_best.pt (генерируется)
├── checkpoints/           # sam2.1_hiera_tiny.pt (скачать, не в git)
├── scripts/
│   ├── extract_patches.py
│   ├── train_cnn.py
│   ├── infer_cnn.py
│   ├── build_report.py
│   ├── sam2_loader.py
│   ├── sam2_prompt_segment.py
│   ├── sam2_auto_segment.py
│   └── sam2_sweep.py
└── output/
    ├── preview/           # исходник, разметка, montage
    ├── cnn/               # обучение и overlay-сетка
    └── sam2/              # SAM 2.1 результаты
```

## Быстрый старт (с нуля)

```bash
git clone https://github.com/we11as22/kataev_images.git
cd kataev_images/lab_2

python -m venv .venv
source .venv/bin/activate
pip install -U pip

# SAM 2 (один раз): клон рядом с репозиторием или в любое место
git clone https://github.com/facebookresearch/segment-anything-2.git ../segment-anything-2
export SAM2_REPO="../segment-anything-2"   # опционально

bash setup.sh    # pip deps, pip install -e SAM2, wget checkpoint
bash run_lab.sh  # полный пайплайн → report.pdf
```

**Важно:** перед запуском SAM выполняется `unset PYTHONPATH` — старые копии `sam2` в `PYTHONPATH` ломают импорт.

## Чекпоинт SAM 2.1

Файл `checkpoints/sam2.1_hiera_tiny.pt` (~149 MB) не в git (лимит GitHub).

```bash
bash checkpoints/download.sh
```

Или вручную: [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)

## Что делает run_lab.sh

| Шаг | Скрипт | Результат |
|-----|--------|-----------|
| 1 | `extract_patches.py` | `dataset/`, `output/preview/` |
| 2 | `train_cnn.py` | `models/cnn_best.pt`, кривые обучения |
| 3 | `infer_cnn.py` | `output/cnn/cnn_overlay_{64,128}.jpg` |
| 4 | `sam2_prompt_segment.py` | сегментация по box из labelme |
| 5 | `sam2_auto_segment.py` | automatic mask generator |
| 6 | `sam2_sweep.py` | сравнение параметров AMG |
| 7 | `build_report.py` | `report.pdf` |

## Классы

| Класс | Цвет overlay |
|-------|--------------|
| Дорога | оранжевый |
| Зелёное поле | зелёный |
| Убранное поле | бежевый |
| Кусты | тёмно-зелёный |

## Только CNN (без SAM)

```bash
unset PYTHONPATH
export PYTHONPATH="$(pwd):$(pwd)/scripts"
python scripts/extract_patches.py
python scripts/train_cnn.py
python scripts/infer_cnn.py --compare-all
python scripts/build_report.py
```

## Зависимости

- Python 3.10+
- PyTorch + torchvision (CPU или CUDA)
- [segment-anything-2](https://github.com/facebookresearch/segment-anything-2) — `pip install -e`

Остальное: `pip install -r requirements.txt`
