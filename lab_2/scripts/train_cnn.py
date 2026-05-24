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
        n_train = max(2, int(len(cls_items) * ratio))
        if n_train >= len(cls_items):
            n_train = max(1, len(cls_items) - 1)
        train.extend(cls_items[:n_train])
        val.extend(cls_items[n_train:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / max(total, 1), correct / max(total, 1)


def smooth(values: list[float], window: int = 5) -> list[float]:
    if len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(float(np.mean(values[lo : i + 1])))
    return out


def main() -> None:
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(config.SEED)

    items = collect_items()
    if not items:
        raise RuntimeError("dataset/ пуст — сначала запустите extract_patches.py")

    train_items, val_items = split_items(items, config.CNN_TRAIN_RATIO)
    train_loader = DataLoader(
        PatchDataset(train_items, augment=True),
        batch_size=min(config.CNN_BATCH_SIZE, len(train_items)),
        shuffle=True,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        PatchDataset(val_items, augment=False),
        batch_size=min(config.CNN_BATCH_SIZE, max(1, len(val_items))),
        shuffle=False,
        num_workers=0,
    )

    model = SmallCNN(len(config.CLASSES)).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.CNN_LR,
        weight_decay=config.CNN_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.CNN_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.CNN_LABEL_SMOOTHING)

    config.ensure_dirs()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_left = config.CNN_EARLY_STOP_PATIENCE

    for epoch in range(1, config.CNN_EPOCHS + 1):
        model.train()
        running = correct = total = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        scheduler.step()
        train_loss = running / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        for k, v in zip(
            ["train_loss", "val_loss", "train_acc", "val_acc"],
            [train_loss, val_loss, train_acc, val_acc],
        ):
            history[k].append(v)
        print(f"epoch {epoch:02d}: loss={train_loss:.3f}/{val_loss:.3f} acc={train_acc:.3f}/{val_acc:.3f}")

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            patience_left = config.CNN_EARLY_STOP_PATIENCE
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": config.CLASSES,
                    "input_size": config.CNN_INPUT_SIZE,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "epoch": epoch,
                },
                config.CNN_MODEL_PATH,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop at epoch {epoch}")
                break

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(smooth(history["train_loss"]), label="train", linewidth=2)
    ax[0].plot(smooth(history["val_loss"]), label="val", linewidth=2)
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(smooth(history["train_acc"]), label="train", linewidth=2)
    ax[1].plot(smooth(history["val_acc"]), label="val", linewidth=2)
    ax[1].set_title("Accuracy")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_CNN / "cnn_training_curves.png", dpi=160)
    plt.close(fig)

    ckpt = torch.load(config.CNN_MODEL_PATH, map_location="cpu", weights_only=False)
    report = {
        "best_val_loss": best_val_loss,
        "best_val_acc": ckpt.get("val_acc"),
        "best_epoch": ckpt.get("epoch"),
        "train_size": len(train_items),
        "val_size": len(val_items),
        "classes": config.CLASSES,
        **history,
    }
    (config.OUTPUT_CNN / "cnn_training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Best val loss: {best_val_loss:.3f}, acc: {ckpt.get('val_acc', 0):.3f} -> {config.CNN_MODEL_PATH}")


if __name__ == "__main__":
    main()
