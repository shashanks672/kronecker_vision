"""
Embedding heatmap and cosine-similarity matrix visualisations.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_embedding_heatmap(
    embeddings: np.ndarray,
    title: str = "Patch Embedding Heatmap (mean over batch)",
    save_path: Optional[str] = None,
    max_patches: int = 64,
) -> plt.Figure:
    """
    Visualise a matrix of shape (num_patches, embed_dim) as a heatmap.
    Useful for inspecting the Kronecker Patch Embedding output.
    """
    data = embeddings[:max_patches]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        data,
        ax=ax,
        cmap="viridis",
        cbar_kws={"label": "Activation"},
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Patch index")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Embedding heatmap saved to {save_path}")
    return fig


def plot_cosine_similarity(
    embeddings: np.ndarray,
    title: str = "Cosine Similarity of Patch Embeddings",
    save_path: Optional[str] = None,
    max_patches: int = 64,
) -> plt.Figure:
    """
    Compute pairwise cosine similarity between patch embeddings and
    display it as a symmetric heatmap.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    data = embeddings[:max_patches]
    sim = cosine_similarity(data)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        sim,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Cosine similarity"},
    )
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Cosine similarity matrix saved to {save_path}")
    return fig
