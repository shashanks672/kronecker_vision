"""
CIFAR-10 Dataset utilities for the Kronecker Vision project.

Provides train / val / test splits with standard CIFAR-10
normalization and optional simple augmentations for training.
"""

from __future__ import annotations

import os
from typing import Tuple, Optional, Dict, Any

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms(
    train: bool = True,
    image_size: int = 32,
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
    std: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616),
) -> transforms.Compose:
    """
    Build torchvision transform pipeline.

    Training uses light augmentation suitable for small 32×32 images.
    Validation / test use only normalization.
    """
    if train:
        return transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_cifar10_loaders(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    seed: int = 42,
    image_size: int = 32,
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
    std: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616),
) -> Dict[str, DataLoader]:
    """
    Create train / validation / test DataLoaders for CIFAR-10.

    The official CIFAR-10 training set is further split into
    train + validation according to `val_ratio`.

    Returns
    -------
    dict with keys 'train', 'val', 'test'
    """
    os.makedirs(root, exist_ok=True)

    train_transform = get_transforms(train=True, image_size=image_size, mean=mean, std=std)
    eval_transform = get_transforms(train=False, image_size=image_size, mean=mean, std=std)

    # Full training set (will be split)
    full_train = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=train_transform,
    )

    # For validation we need the same images but without augmentation.
    # We therefore create a second dataset object pointing to the same files.
    full_train_eval = datasets.CIFAR10(
        root=root,
        train=True,
        download=False,
        transform=eval_transform,
    )

    test_set = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=eval_transform,
    )

    # Deterministic split
    n_total = len(full_train)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(n_total),
        [n_train, n_val],
        generator=generator,
    )

    # Build subset datasets
    train_set = torch.utils.data.Subset(full_train, train_indices.indices)
    val_set = torch.utils.data.Subset(full_train_eval, val_indices.indices)

    loaders = {
        "train": DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
    return loaders


def get_class_names() -> list:
    """Return the official CIFAR-10 class names."""
    return [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]


if __name__ == "__main__":
    # Quick sanity check
    loaders = get_cifar10_loaders(batch_size=4, num_workers=0)
    x, y = next(iter(loaders["train"]))
    print(f"Train batch shape: {x.shape}, labels: {y.shape}")
    print(f"Classes: {get_class_names()}")
