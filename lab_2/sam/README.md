# Lab 2 — SAM 2.1 (отдельно от CNN)

```
sam/
├── run_sam.sh           # полный пайплайн SAM
├── report.pdf           # отчёт SAM
├── checkpoints/         # sam2.1_hiera_tiny.pt
├── output/              # overlay, sweep, json-отчёты
└── scripts/
    ├── sam2_prompt_segment.py   # box-prompts из labelme
    ├── sam2_auto_segment.py     # automatic mask generator
    ├── sam2_sweep.py            # сравнение параметров
    └── build_sam_report.py
```

Запуск:

```bash
bash sam/run_sam.sh
```

CNN — отдельно: `bash run_lab.sh` → `report.pdf`
