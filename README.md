# kataev_images — Судаков Алексей

| Работа | Отчёт |
|--------|-------|
| Lab 1 | [lab_1/report.pdf](lab_1/report.pdf) |
| Lab 2 | [lab_2/report.pdf](lab_2/report.pdf) |

## Lab 2 — главный результат

- CNN overlay 128: [lab_2/output/cnn/cnn_overlay_128.jpg](lab_2/output/cnn/cnn_overlay_128.jpg)
- CNN overlay 64: [lab_2/output/cnn/cnn_overlay_64.jpg](lab_2/output/cnn/cnn_overlay_64.jpg)

## Запуск lab 2 с нуля

Подробная инструкция: **[lab_2/README.md](lab_2/README.md)**

```bash
cd lab_2
python -m venv .venv && source .venv/bin/activate
git clone https://github.com/facebookresearch/segment-anything-2.git ../segment-anything-2
bash setup.sh
bash run_lab.sh
```

Пайплайн: патчи → CNN → инференс сеткой → SAM 2.1 → `report.pdf`

Чекпоинт SAM (~149 MB, не в git): [lab_2/checkpoints/README.md](lab_2/checkpoints/README.md)
