"""Train CNN classifier on patches from dataset/."""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], augment: bool = False):
        self.items = items
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        if self.augment and random.random() < 0.5:
            arr = arr[:, ::-1].copy()
        if self.augment and random.random() < 0.5:
            arr = arr[::-1, :].copy()
        return torch.from_numpy(arr).permute(2, 0, 1), label


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


def collect_items() -> list[tuple[Path, int]]:
    items = []
    for i, cls in enumerate(config.CLASSES):
        for path in sorted((config.DATASET / cls).glob("*.jpg")):
            items.append((path, i))
    return items


def split_items(items: list[tuple[Path, int]], ratio: float):
    rng = random.Random(config.SEED)
    by_class: dict[int, list] = {}
    for item in items:
        by_class.setdefault(item[1], []).append(item)
    train, val = [], []
    for cls_items in by_class.values():
        rng.shuffle(cls_items)
        n = max(1, int(len(cls_items) * ratio))
        train.extend(cls_items[:n])
        val.extend(cls_items[n:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    items = collect_items()
    if not items:
        raise RuntimeError("dataset/ пуст — сначала запустите extract_patches.py")

    train_items, val_items = split_items(items, config.CNN_TRAIN_RATIO)
    train_loader = DataLoader(
        PatchDataset(train_items, augment=True),
        batch_size=config.CNN_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        PatchDataset(val_items, augment=False),
        batch_size=config.CNN_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    model = SmallCNN(len(config.CLASSES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.CNN_LR)
    criterion = nn.CrossEntropyLoss()

    config.MODELS.mkdir(parents=True, exist_ok=True)
    config.OUTPUT.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, config.CNN_EPOCHS + 1):
        model.train()
        running = correct = total = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        train_loss = running / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader)
        for k, v in zip(
            ["train_loss", "val_loss", "train_acc", "val_acc"],
            [train_loss, val_loss, train_acc, val_acc],
        ):
            history[k].append(v)
        print(f"epoch {epoch:02d}: train={train_acc:.3f} val={val_acc:.3f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": config.CLASSES,
                    "input_size": config.CNN_INPUT_SIZE,
                    "val_acc": val_acc,
                },
                config.CNN_MODEL_PATH,
            )

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.OUTPUT / "cnn_training_curves.png", dpi=160)
    plt.close(fig)

    report = {
        "best_val_acc": best_acc,
        "train_size": len(train_items),
        "val_size": len(val_items),
        "classes": config.CLASSES,
        **history,
    }
    (config.OUTPUT / "cnn_training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Best val acc: {best_acc:.3f} -> {config.CNN_MODEL_PATH}")


if __name__ == "__main__":
    main()
