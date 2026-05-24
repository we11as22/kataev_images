from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
IMG = ROOT / 'img'
DATA = json.loads((ROOT / 'data.json').read_text(encoding='utf-8'))
summary = DATA['summary']
colors = {
    'Дорога': '#d99a28',
    'Трава': '#3a9b35',
    'Поле': '#8c6d31',
    'Столбики в поле': '#2f6fbd',
}

def mean_var(channel_key='Gray', file_label='Grey'):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, rec in enumerate(summary):
        m = rec[f'{channel_key}_mean']
        v = rec[f'{channel_key}_var']
        left = max(0, m - v / 2)
        width = min(v, 255 - left)
        ax.add_patch(Rectangle((left, i - 0.3), width, 0.6,
                               facecolor=colors[rec['Surface']], alpha=0.45,
                               edgecolor='black'))
        ax.plot(m, i, 'ko', markersize=4)
        ax.text(m + 2, i, f'{m:.1f}', va='center', fontsize=8)
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels([r['Surface'] for r in summary])
    ax.set_xlim(0, 255)
    ax.set_xlabel('Яркость')
    ax.set_title(f'Среднее и дисперсия, {file_label}')
    ax.grid(axis='x', linestyle=':', alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG / f'mean_and_error_{file_label}.png', dpi=180)
    plt.close(fig)

if __name__ == '__main__':
    for file_label, channel_key in [('Grey', 'Gray'), ('Red', 'R'), ('Green', 'G'), ('Blue', 'B')]:
        mean_var(channel_key, file_label)
    print('Saved mean_and_error_Grey/Red/Green/Blue.png')
