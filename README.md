# kataev_images — Судаков Алексей

| Работа | Отчёт |
|--------|-------|
| Lab 1 | [lab_1/report.pdf](lab_1/report.pdf) |
| Lab 2 | [lab_2/report.pdf](lab_2/report.pdf) |

## Lab 2 — главный результат

- CNN overlay 128: [lab_2/output/cnn_overlay_128.jpg](lab_2/output/cnn_overlay_128.jpg)
- CNN overlay 64: [lab_2/output/cnn_overlay_64.jpg](lab_2/output/cnn_overlay_64.jpg)

## Запуск lab 2

```bash
source ~/envs/.venv/bin/activate
cd lab_2
bash run_lab.sh
```

Скрипт: патчи → CNN → инференс сеткой → SAM 2.1 → report.pdf

**SAM 2** (один раз): `pip install -e` из `kataev_sources/sam2`  
**Чекпоинт SAM** (~149 MB, не в git): [lab_2/checkpoints/README.md](lab_2/checkpoints/README.md)

Перед запуском SAM: `unset PYTHONPATH`
