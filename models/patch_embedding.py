"""
Standard Linear Patch Embedding used by the baseline Vision Transformer.

This module implements the classic ViT patch embedding:
    1. Split image into non-overlapping patches
    2. Flatten each patch
    3. Project with a linear layer (or equivalently a Conv2d with kernel=stride=patch_size)

We provide both implementations for clarity and educational value.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LinearPatchEmbedding(nn.Module):
    """
    Classic ViT linear patch embedding.

    Parameters
    ----------
    img_size : int
        Spatial size of the input image (assumed square).
    patch_size : int
        Spatial size of each patch (assumed square).
    in_channels : int
        Number of input channels (3 for RGB).
    embed_dim : int
        Output embedding dimension.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 128,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size  # 48 for CIFAR-10 4×4

        # Linear projection of flattened patches
        self.proj = nn.Linear(self.patch_dim, embed_dim)

        # Optional: we keep a Conv2d version as well (numerically equivalent)
        # for users who prefer the convolutional formulation.
        self.proj_conv = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.use_conv = False  # switch to True if desired
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        # Conv path
        nn.init.trunc_normal_(self.proj_conv.weight, std=0.02)
        if self.proj_conv.bias is not None:
            nn.init.zeros_(self.proj_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (B, C, H, W)

        Returns
        -------
        embeddings : Tensor of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size

        if self.use_conv:
            # (B, embed_dim, H/p, W/p) → (B, num_patches, embed_dim)
            x = self.proj_conv(x)
            x = x.flatten(2).transpose(1, 2)
        else:
            # Explicit unfold → flatten → linear
            # (B, C, H, W) → (B, num_patches, patch_dim)
            x = x.unfold(2, self.patch_size, self.patch_size) \
                 .unfold(3, self.patch_size, self.patch_size)
            # x shape: (B, C, nH, nW, p, p)
            x = x.contiguous().view(B, C, -1, self.patch_size * self.patch_size)
            x = x.permute(0, 2, 1, 3).contiguous()  # (B, num_patches, C, p*p)
            x = x.view(B, self.num_patches, -1)     # (B, num_patches, patch_dim)
            x = self.proj(x)

        return x


class PatchEmbedding(nn.Module):
    """
    Thin wrapper that adds a learnable class token and absolute
    positional embeddings on top of a linear (or Kronecker) patch embedding.

    This is the module actually used inside the Vision Transformer.
    """

    def __init__(
        self,
        patch_embed: nn.Module,
        embed_dim: int,
        num_patches: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = patch_embed
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        tokens : (B, 1 + num_patches, embed_dim)
        """
        B = x.shape[0]
        x = self.patch_embed(x)                     # (B, N, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)       # (B, 1+N, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x


if __name__ == "__main__":
    pe = LinearPatchEmbedding(img_size=32, patch_size=4, embed_dim=128)
    x = torch.randn(2, 3, 32, 32)
    out = pe(x)
    print(f"Patch embeddings shape: {out.shape}")  # (2, 64, 128)
    print(f"Parameters: {sum(p.numel() for p in pe.parameters()):,}")
