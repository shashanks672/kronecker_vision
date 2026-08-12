"""
t-SNE visualisation of penultimate-layer embeddings.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect (embedding, label) pairs from the model.

    For ViT / KViT we use the [CLS] token.
    For the CNN we use the feature vector just before the final linear layer.
    """
    model.eval()
    embeddings = []
    labels = []
    n = 0

    for images, targets in tqdm(loader, desc="Extracting embeddings", leave=False):
        images = images.to(device)
        targets = targets.cpu().numpy()

        # Hook-style extraction depending on model type
        if hasattr(model, "forward_features"):
            feats, _ = model.forward_features(images)
        elif hasattr(model, "fc1"):  # BaselineCNN
            # Re-implement the forward up to the penultimate layer
            x = model.conv1(images)
            x = model.bn1(x)
            x = torch.relu(x)
            x = model.pool1(x)
            x = model.conv2(x)
            x = model.bn2(x)
            x = torch.relu(x)
            x = model.pool2(x)
            x = torch.flatten(x, 1)
            x = model.fc1(x)
            x = torch.relu(x)
            feats = x
        else:
            # Fallback: use the logits themselves (not ideal)
            out = model(images)
            if isinstance(out, tuple):
                out = out[0]
            feats = out

        embeddings.append(feats.cpu().numpy())
        labels.append(targets)
        n += images.size(0)
        if n >= max_samples:
            break

    emb = np.concatenate(embeddings, axis=0)[:max_samples]
    lab = np.concatenate(labels, axis=0)[:max_samples]
    return emb, lab


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[list] = None,
    title: str = "t-SNE of embeddings",
    save_path: Optional[str] = None,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> plt.Figure:
    """
    Compute 2-D t-SNE and produce a scatter plot coloured by class.
    """
    print(f"Running t-SNE on {embeddings.shape[0]} samples …")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap="tab10",
        s=12,
        alpha=0.7,
        edgecolors="none",
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    if class_names is not None:
        # Create a legend
        handles = [
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=plt.cm.tab10(i / 10),
                markersize=8,
                label=class_names[i],
            )
            for i in range(len(class_names))
        ]
        ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"t-SNE plot saved to {save_path}")
    return fig
