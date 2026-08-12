"""
Shared training utilities: epoch loop, checkpointing, plotting, seed, device.
"""

from __future__ import annotations

import os
import time
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    """Make experiments as reproducible as possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import torch

def get_device(prefer: str = "auto") -> torch.device:
    import torch

    print("Requested device:", prefer)
    print("CUDA available:", torch.cuda.is_available())
    print("MPS available:", torch.backends.mps.is_available())

    prefer = prefer.lower()

    if prefer == "mps":
        if torch.backends.mps.is_available():
            print("Returning MPS")
            return torch.device("mps")

    if prefer == "cuda":
        if torch.cuda.is_available():
            print("Returning CUDA")
            return torch.device("cuda")

    if prefer == "auto":
        if torch.cuda.is_available():
            print("Returning CUDA")
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            print("Returning MPS")
            return torch.device("mps")

    print("Returning CPU")
    return torch.device("cpu")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    epoch: int = 0,
) -> Dict[str, float]:
    """Run a single training epoch and return average loss & accuracy."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        preds = outputs.argmax(dim=1)
        acc = (preds == targets).float().mean().item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validation pass."""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in tqdm(loader, desc="[val]", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, targets)

        preds = outputs.argmax(dim=1)
        acc = (preds == targets).float().mean().item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

    return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Training Curves",
) -> None:
    """
    Plot train/val loss and accuracy side by side.
    history keys expected: train_loss, val_loss, train_acc, val_acc
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Val", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Save a confusion-matrix heatmap."""
    import seaborn as sns

    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")
