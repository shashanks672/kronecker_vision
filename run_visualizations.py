"""
Generate all research visualisations after models have been trained.

Usage:
    python run_visualizations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from datasets.cifar import get_cifar10_loaders, get_class_names
from models.cnn import BaselineCNN, CNNConfig
from models.vit import ViTConfig, build_vit, build_kvit
from evaluate.metrics import evaluate_model
from visualization.tsne import extract_embeddings, plot_tsne
from visualization.pca import plot_pca
from visualization.heatmap import plot_embedding_heatmap, plot_cosine_similarity
from visualization.attention import plot_attention_map, plot_multi_head_attention
from train.utils import get_device, set_seed


def load_model(name: str, cfg: dict, device):
    weights = {
        "cnn": ROOT / "weights" / "cnn_best.pt",
        "vit": ROOT / "weights" / "vit_best.pt",
        "kvit": ROOT / "weights" / "kvit_best.pt",
    }
    if name == "cnn":
        model = BaselineCNN(CNNConfig(num_classes=10))
    else:
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
        model = build_vit(vit_cfg) if name == "vit" else build_kvit(vit_cfg)

    path = weights[name]
    if path.exists():
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded {path}")
    else:
        print(f"WARNING: {path} not found – using random weights")
    return model.to(device).eval()


def main():
    with open(ROOT / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = get_device(cfg.get("device", "cuda"))
    figures = Path(cfg["paths"]["figures_dir"])
    figures.mkdir(parents=True, exist_ok=True)

    loaders = get_cifar10_loaders(
        root=cfg["dataset"]["root"],
        batch_size=128,
        num_workers=2,
        seed=cfg["training"]["seed"],
    )
    class_names = get_class_names()

    for name in ["cnn", "vit", "kvit"]:
        print(f"\n=== Visualisations for {name.upper()} ===")
        model = load_model(name, cfg, device)

        # Embeddings for t-SNE / PCA
        emb, lab = extract_embeddings(model, loaders["test"], device, max_samples=1500)
        plot_tsne(
            emb, lab, class_names,
            title=f"t-SNE – {name.upper()}",
            save_path=str(figures / f"{name}_tsne.png"),
        )
        plot_pca(
            emb, lab, class_names,
            title=f"PCA – {name.upper()}",
            save_path=str(figures / f"{name}_pca.png"),
        )

        # Kronecker-specific + attention
        if name == "kvit":
            # Take one batch
            images, _ = next(iter(loaders["test"]))
            images = images[:4].to(device)
            with torch.no_grad():
                patch_emb = model.get_patch_embeddings(images)  # (4, 64, 128)
                logits, attn_maps = model(images, return_attention=True)

            # Heatmap of first sample
            plot_embedding_heatmap(
                patch_emb[0].cpu().numpy(),
                title="Kronecker Patch Embedding (sample 0)",
                save_path=str(figures / "kvit_embedding_heatmap.png"),
            )
            plot_cosine_similarity(
                patch_emb[0].cpu().numpy(),
                title="Cosine similarity of KPE (sample 0)",
                save_path=str(figures / "kvit_cosine_sim.png"),
            )
            # Attention of last layer
            plot_multi_head_attention(
                attn_maps,
                layer=-1,
                max_heads=4,
                save_path=str(figures / "kvit_attention_heads.png"),
            )

        if name == "vit":
            images, _ = next(iter(loaders["test"]))
            images = images[:1].to(device)
            with torch.no_grad():
                _, attn_maps = model(images, return_attention=True)
            plot_multi_head_attention(
                attn_maps,
                layer=-1,
                max_heads=4,
                save_path=str(figures / "vit_attention_heads.png"),
            )

    print("\nAll visualisations written to", figures)


if __name__ == "__main__":
    main()
