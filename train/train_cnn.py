"""
Phase 1 – Train the baseline CNN on CIFAR-10.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.cifar import get_cifar10_loaders, get_class_names
from models.cnn import BaselineCNN, CNNConfig
from evaluate.metrics import evaluate_model, print_metrics
from train.utils import (
    set_seed,
    get_device,
    train_one_epoch,
    validate,
    save_checkpoint,
    plot_training_curves,
    plot_confusion_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline CNN on CIFAR-10")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["lr"]
    seed = args.seed or cfg["training"]["seed"]
    device_str = args.device or cfg.get("device", "cuda")

    set_seed(seed)
    device = get_device(device_str)
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    loaders = get_cifar10_loaders(
        root=cfg["dataset"]["root"],
        batch_size=batch_size,
        num_workers=cfg["training"]["num_workers"],
        seed=seed,
        mean=tuple(cfg["dataset"]["mean"]),
        std=tuple(cfg["dataset"]["std"]),
    )
    class_names = get_class_names()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = BaselineCNN(CNNConfig(num_classes=cfg["dataset"]["num_classes"]))
    model = model.to(device)
    print(f"CNN parameters: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["training"]["label_smoothing"])
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg["training"]["weight_decay"],
        betas=tuple(cfg["optimizer"]["betas"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg["scheduler"]["min_lr"])
    scaler = GradScaler() if (device.type == "cuda" and cfg["training"]["amp"]) else None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    weights_dir = Path(cfg["paths"]["weights_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    weights_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler, epoch
        )
        val_metrics = validate(model, loaders["val"], criterion, device)
        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics,
                str(weights_dir / "cnn_best.pt"),
            )

    # Final checkpoint
    save_checkpoint(
        model, optimizer, epochs, val_metrics, str(weights_dir / "cnn_last.pt")
    )

    # ------------------------------------------------------------------
    # Plots & final evaluation
    # ------------------------------------------------------------------
    plot_training_curves(
        history,
        save_path=str(figures_dir / "cnn_training_curves.png"),
        title="Baseline CNN – Training Curves",
    )

    # Test evaluation with best weights
    ckpt = torch.load(weights_dir / "cnn_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate_model(
        model, loaders["test"], device, criterion, return_predictions=True
    )
    print_metrics(test_metrics, title="Baseline CNN – Test Results")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names,
        save_path=str(figures_dir / "cnn_confusion_matrix.png"),
        title="Baseline CNN – Confusion Matrix",
    )

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Weights saved to {weights_dir}")


if __name__ == "__main__":
    main()
