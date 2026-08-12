"""
Attention-map visualisation for Vision Transformers.
"""

from __future__ import annotations

import os
from typing import Optional, List

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_attention_map(
    attn: torch.Tensor,
    patch_size: int = 4,
    img_size: int = 32,
    head: int = 0,
    layer: int = -1,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualise a single attention head from a transformer layer.

    Parameters
    ----------
    attn : Tensor of shape (B, num_heads, N, N) or (num_heads, N, N)
           where N = 1 + num_patches (includes CLS token).
    head : which attention head to plot
    layer : only used for the title (the caller selects the layer)
    """
    if attn.dim() == 4:
        attn = attn[0]  # take first sample in batch
    attn = attn.detach().cpu().numpy()

    # Focus on the attention from the CLS token to all patches
    # (row 0 of the attention matrix, excluding the CLS→CLS entry)
    cls_attn = attn[head, 0, 1:]  # (num_patches,)

    n = int(np.sqrt(len(cls_attn)))
    grid = cls_attn.reshape(n, n)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap="inferno", interpolation="nearest")
    ax.set_title(title or f"CLS Attention (layer {layer}, head {head})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Attention map saved to {save_path}")
    return fig


def plot_multi_head_attention(
    attn_maps: List[torch.Tensor],
    layer: int = -1,
    max_heads: int = 4,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot several heads of a chosen layer side-by-side.
    """
    attn = attn_maps[layer]
    if attn.dim() == 4:
        attn = attn[0]
    num_heads = min(attn.shape[0], max_heads)

    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 3.5))
    if num_heads == 1:
        axes = [axes]

    for h in range(num_heads):
        cls_attn = attn[h, 0, 1:].cpu().numpy()
        n = int(np.sqrt(len(cls_attn)))
        grid = cls_attn.reshape(n, n)
        im = axes[h].imshow(grid, cmap="inferno")
        axes[h].set_title(f"Head {h}")
        axes[h].set_xticks([])
        axes[h].set_yticks([])
        fig.colorbar(im, ax=axes[h], fraction=0.046, pad=0.04)

    fig.suptitle(f"CLS Attention – Layer {layer if layer >= 0 else len(attn_maps)+layer}", fontsize=13)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Multi-head attention plot saved to {save_path}")
    return fig
