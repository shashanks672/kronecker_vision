"""
Transformer building blocks used by both the baseline ViT and the Kronecker-ViT.

Implements:
- Multi-Head Self-Attention (with optional attention dropout)
- MLP / Feed-Forward network
- Transformer Encoder Block (Pre-Norm residual)
- Stochastic Depth (DropPath)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic Depth (DropPath) regularisation."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # Work with any number of dimensions
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample when applied in residual blocks."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention.

    Returns both the output and the attention weights (useful for visualisation).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        x : (B, N, C)
        return_attention : if True also return the attention map (B, heads, N, N)

        Returns
        -------
        out : (B, N, C)
        attn : optional attention weights
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        if return_attention:
            return out, attn
        return out, None


class MLP(nn.Module):
    """Two-layer feed-forward network with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer Encoder Block.

        x = x + DropPath(Attention(LN(x)))
        x = x + DropPath(MLP(LN(x)))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden, drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Attention residual
        attn_out, attn_weights = self.attn(self.norm1(x), return_attention=return_attention)
        x = x + self.drop_path(attn_out)
        # MLP residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """Stack of TransformerBlocks with optional final LayerNorm."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        # Linearly increasing drop-path rate (stochastic depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        attn_maps = [] if return_attention else None
        for blk in self.blocks:
            x, attn = blk(x, return_attention=return_attention)
            if return_attention:
                attn_maps.append(attn)
        x = self.norm(x)
        return x, attn_maps
