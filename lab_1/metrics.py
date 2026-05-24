from pathlib import Path
import json
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
IMG = Image.open(ROOT / 'img' / 'source.jpg').convert('RGB')
ARR = np.asarray(IMG, dtype=np.float32)

def gray(a):
    return 0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]

def block_stats(x, y, size=18):
    h = size // 2
    p = ARR[y-h:y+h, x-h:x+h]
    return {
        'R': float(p[...,0].mean()),
        'G': float(p[...,1].mean()),
        'B': float(p[...,2].mean()),
        'Gray': float(gray(p).mean()),
    }

if __name__ == '__main__':
    for surface, points in DATA['samples'].items():
        vals = [block_stats(x, y, DATA['block_size']) for x, y in points]
        print(surface, vals[0])
