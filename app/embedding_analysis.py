"""
Embedding Analysis helpers for the Kronecker Vision Streamlit demo.

Extracts real embeddings from CNN / ViT / Kronecker-ViT and provides
statistics + matplotlib visualisations used by the Embedding Analysis section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Optional t-SNE (can be slow; guarded)
try:
    from sklearn.manifold import TSNE
    _HAS_TSNE = True
except ImportError:
    _HAS_TSNE = False


@dataclass
class EmbeddingResult:
    """Container for a single model's embedding analysis."""
    model_name: str
    embedding: np.ndarray          # 1-D primary vector (CLS or CNN feature)
    patch_matrix: Optional[np.ndarray] = None  # (N_patches, D) if available
    logits: Optional[np.ndarray] = None
    pred_idx: Optional[int] = None
    confidence: Optional[float] = None
    source: str = ""               # human-readable description of extraction point


def extract_cnn_embedding(
    model: nn.Module,
    x: torch.Tensor,
) -> EmbeddingResult:
    """CNN: penultimate features (after fc1 + ReLU), dim = 256 by default."""
    model.eval()
    with torch.no_grad():
        emb = model.get_embedding(x)           # (B, 256)
        logits = model(x)                      # (B, 10)
    emb_np = emb[0].cpu().numpy()
    logits_np = logits[0].cpu().numpy()
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())
    return EmbeddingResult(
        model_name="CNN",
        embedding=emb_np,
        patch_matrix=None,
        logits=logits_np,
        pred_idx=pred,
        confidence=float(probs[pred]),
        source="Penultimate layer (fc1 + ReLU) before classifier — 256-dim",
    )


def extract_vit_embedding(
    model: nn.Module,
    x: torch.Tensor,
    name: str = "ViT",
) -> EmbeddingResult:
    """ViT / Kronecker-ViT: CLS token before head + optional patch matrix."""
    model.eval()
    with torch.no_grad():
        cls = model.get_embedding(x)                    # (B, D)
        patch_mat = model.get_patch_embeddings(x)       # (B, N, D)
        out = model(x, return_attention=False)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
    emb_np = cls[0].cpu().numpy()
    patch_np = patch_mat[0].cpu().numpy()
    logits_np = logits[0].cpu().numpy()
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())
    return EmbeddingResult(
        model_name=name,
        embedding=emb_np,
        patch_matrix=patch_np,
        logits=logits_np,
        pred_idx=pred,
        confidence=float(probs[pred]),
        source="[CLS] token after Transformer encoder, before classification head",
    )


def compute_stats(vec: np.ndarray) -> Dict[str, float]:
    """Basic statistics for a 1-D embedding vector."""
    return {
        "dim": int(vec.size),
        "mean": float(vec.mean()),
        "std": float(vec.std()),
        "min": float(vec.min()),
        "max": float(vec.max()),
        "l2_norm": float(np.linalg.norm(vec)),
        "l1_norm": float(np.abs(vec).sum()),
        "sparsity": float((np.abs(vec) < 1e-4).mean()),
    }


def plot_embedding_bar(
    vec: np.ndarray,
    n_show: int = 64,
    title: str = "Embedding values (first N dims)",
    figsize: Tuple[float, float] = (10, 2.8),
) -> plt.Figure:
    """Bar chart of the first N embedding dimensions."""
    n = min(n_show, len(vec))
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in vec[:n]]
    ax.bar(range(n), vec[:n], color=colors, width=0.85, edgecolor="none")
    ax.axhline(0, color="#30363d", linewidth=0.8)
    ax.set_xlabel("Dimension", fontsize=8)
    ax.set_ylabel("Value", fontsize=8)
    ax.set_title(title, fontsize=10, color="#f0f6fc", pad=6)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_embedding_heatmap(
    data: np.ndarray,
    title: str = "Embedding heatmap",
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Heatmap for either:
    - 1-D vector → shown as (1, D)
    - 2-D patch matrix → (N_patches, D)
    """
    arr = np.atleast_2d(data)
    if figsize is None:
        fig_h = max(2.2, min(5.5, arr.shape[0] * 0.1 + 1.4))
        figsize = (10, fig_h)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xlabel("Embedding dimension", fontsize=8)
    ax.set_ylabel("Patch" if arr.shape[0] > 1 else "", fontsize=8)
    ax.set_title(title, fontsize=10, color="#f0f6fc", pad=6)
    ax.tick_params(labelsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=7, colors="#8b949e")
    fig.tight_layout()
    return fig


def plot_histogram(
    vec: np.ndarray,
    title: str = "Embedding value distribution",
    bins: int = 50,
    figsize: Tuple[float, float] = (8, 3.0),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(vec, bins=bins, color="#388bfd", edgecolor="#0d1117", alpha=0.9)
    ax.axvline(vec.mean(), color="#3fb950", linestyle="--", linewidth=1.4, label=f"μ={vec.mean():.3f}")
    ax.axvline(0, color="#8b949e", linestyle=":", linewidth=1)
    ax.set_xlabel("Value", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title(title, fontsize=10, color="#f0f6fc", pad=6)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_pca_patches(
    patch_matrix: np.ndarray,
    title: str = "PCA of patch embeddings",
    figsize: Tuple[float, float] = (5.5, 3.8),
) -> Optional[plt.Figure]:
    """2-D PCA on patch tokens (N, D). Colours by patch index (spatial order)."""
    if patch_matrix is None or patch_matrix.ndim != 2 or patch_matrix.shape[0] < 3:
        return None
    n = patch_matrix.shape[0]
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(patch_matrix)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=np.arange(n), cmap="viridis", s=28, alpha=0.85, edgecolors="none",
    )
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)", fontsize=8)
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)", fontsize=8)
    ax.set_title(title, fontsize=10, color="#f0f6fc", pad=6)
    ax.tick_params(labelsize=7)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Patch", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def plot_tsne_patches(
    patch_matrix: np.ndarray,
    title: str = "t-SNE of patch embeddings",
    perplexity: float = 15.0,
    figsize: Tuple[float, float] = (5.5, 3.8),
) -> Optional[plt.Figure]:
    if not _HAS_TSNE or patch_matrix is None or patch_matrix.ndim != 2 or patch_matrix.shape[0] < 5:
        return None
    n = patch_matrix.shape[0]
    perp = min(perplexity, max(2, n // 3))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(patch_matrix)

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=np.arange(n), cmap="plasma", s=28, alpha=0.85, edgecolors="none",
    )
    ax.set_xlabel("t-SNE 1", fontsize=8)
    ax.set_ylabel("t-SNE 2", fontsize=8)
    ax.set_title(title, fontsize=10, color="#f0f6fc", pad=6)
    ax.tick_params(labelsize=7)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Patch", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def format_raw_values(vec: np.ndarray, n_show: int = 32) -> str:
    """Pretty-print first N values for the Raw Embedding tab."""
    lines = [f"{v:+.4f}" for v in vec[:n_show]]
    body = "\n".join(lines)
    if len(vec) > n_show:
        body += f"\n… ({len(vec) - n_show} more dimensions)"
    return body
