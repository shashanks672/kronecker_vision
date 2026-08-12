"""
PCA visualisation of penultimate-layer embeddings.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_pca(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[list] = None,
    title: str = "PCA of embeddings",
    save_path: Optional[str] = None,
    n_components: int = 2,
) -> plt.Figure:
    """
    Project embeddings onto the first two principal components
    and produce a scatter plot coloured by class.
    """
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_

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
    ax.set_title(
        f"{title}\n(explained variance: {explained[0]*100:.1f}% / {explained[1]*100:.1f}%)",
        fontsize=13,
    )
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")

    if class_names is not None:
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
        print(f"PCA plot saved to {save_path}")
    return fig
