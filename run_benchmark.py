"""
Run a full comparison of CNN / ViT / Kronecker-ViT on the test set.
Requires that the three models have already been trained and checkpoints exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from datasets.cifar import get_cifar10_loaders
from models.cnn import BaselineCNN, CNNConfig
from models.vit import ViTConfig, build_vit, build_kvit
from evaluate.benchmark import run_benchmark, format_comparison_table
from train.utils import get_device, set_seed


def main():
    with open(ROOT / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = get_device(cfg.get("device", "cuda"))

    loaders = get_cifar10_loaders(
        root=cfg["dataset"]["root"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        seed=cfg["training"]["seed"],
    )

    vit_cfg = ViTConfig(
        img_size=cfg["dataset"]["image_size"],
        patch_size=cfg["patch"]["size"],
        embed_dim=cfg["model"]["embed_dim"],
        depth=cfg["model"]["depth"],
        num_heads=cfg["model"]["num_heads"],
        local_feat_dim=cfg["kronecker"]["local_feat_dim"],
        pos_dim=cfg["kronecker"]["pos_dim"],
        num_local_features=cfg["kronecker"]["num_local_features"],
    )

    models = {}
    for name, builder, weight_name in [
        ("CNN", lambda: BaselineCNN(CNNConfig()), "cnn_best.pt"),
        ("ViT", lambda: build_vit(vit_cfg), "vit_best.pt"),
        ("Kronecker-ViT", lambda: build_kvit(vit_cfg), "kvit_best.pt"),
    ]:
        m = builder()
        wpath = ROOT / "weights" / weight_name
        if wpath.exists():
            ckpt = torch.load(wpath, map_location=device)
            m.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded {wpath}")
        else:
            print(f"WARNING: {wpath} missing – random weights")
        models[name] = m

    criterion = nn.CrossEntropyLoss()
    results = run_benchmark(
        models,
        loaders["test"],
        device,
        criterion=criterion,
        save_dir=str(ROOT / "results"),
    )

    print("\n" + format_comparison_table(results))


if __name__ == "__main__":
    main()
