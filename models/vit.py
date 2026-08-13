"""
Vision Transformer (ViT) and Kronecker-ViT for CIFAR-10.

Both models share the same transformer backbone; they differ only in the
patch embedding layer:

- Baseline ViT  → LinearPatchEmbedding
- Kronecker-ViT → KroneckerPatchEmbedding

This design makes fair comparison straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

from .patch_embedding import LinearPatchEmbedding, PatchEmbedding
from .kronecker_embedding import KroneckerPatchEmbedding, KroneckerPatchEmbeddingConfig
from .transformer import TransformerEncoder


@dataclass
class ViTConfig:
    """Shared configuration for both ViT variants."""
    img_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    embed_dim: int = 128
    depth: int = 6
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.0
    drop_path: float = 0.05
    # Kronecker-specific (ignored by baseline)
    local_feat_dim: int = 8
    pos_dim: int = 16
    num_local_features: int = 4


class VisionTransformer(nn.Module):
    """
    Vision Transformer that can be instantiated with either a linear or a
    Kronecker patch embedding.

    Parameters
    ----------
    config : ViTConfig
    use_kronecker : bool
        If True, use KroneckerPatchEmbedding; otherwise LinearPatchEmbedding.
    """

    def __init__(
        self,
        config: Optional[ViTConfig] = None,
        use_kronecker: bool = False,
    ):
        super().__init__()
        self.config = config or ViTConfig()
        cfg = self.config
        self.use_kronecker = use_kronecker

        # ------------------------------------------------------------------
        # Patch embedding
        # ------------------------------------------------------------------
        if use_kronecker:
            kpe_cfg = KroneckerPatchEmbeddingConfig(
                img_size=cfg.img_size,
                patch_size=cfg.patch_size,
                in_channels=cfg.in_channels,
                embed_dim=cfg.embed_dim,
                local_feat_dim=cfg.local_feat_dim,
                pos_dim=cfg.pos_dim,
                num_local_features=cfg.num_local_features,
                dropout=cfg.dropout,
            )
            patch_embed = KroneckerPatchEmbedding(kpe_cfg)
        else:
            patch_embed = LinearPatchEmbedding(
                img_size=cfg.img_size,
                patch_size=cfg.patch_size,
                in_channels=cfg.in_channels,
                embed_dim=cfg.embed_dim,
            )

        num_patches = (cfg.img_size // cfg.patch_size) ** 2
        self.patch_embed = PatchEmbedding(
            patch_embed=patch_embed,
            embed_dim=cfg.embed_dim,
            num_patches=num_patches,
            dropout=cfg.dropout,
        )

        # ------------------------------------------------------------------
        # Transformer encoder
        # ------------------------------------------------------------------
        self.encoder = TransformerEncoder(
            dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            drop=cfg.dropout,
            attn_drop=cfg.attn_dropout,
            drop_path_rate=cfg.drop_path,
        )

        # ------------------------------------------------------------------
        # Classification head
        # ------------------------------------------------------------------
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward_features(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Extract the [CLS] token representation (and optional attention maps).

        Returns
        -------
        cls_token : (B, embed_dim)
        attn_maps : list of attention tensors (one per block) or None
        """
        x = self.patch_embed(x)                     # (B, 1+N, D)
        x, attn_maps = self.encoder(x, return_attention=return_attention)
        cls_token = x[:, 0]                         # (B, D)
        return cls_token, attn_maps

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Parameters
        ----------
        x : (B, C, H, W)
        return_attention : whether to also return attention maps

        Returns
        -------
        logits : (B, num_classes)
        attn_maps : optional list of attention maps
        """
        features, attn_maps = self.forward_features(x, return_attention)
        logits = self.head(features)
        if return_attention:
            return logits, attn_maps
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Utility used by the Streamlit app and visualisations:
        return the raw patch embeddings *before* the class token / pos-emb.
        Shape: (B, num_patches, embed_dim)
        """
        return self.patch_embed.patch_embed(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the [CLS] token embedding immediately before the classification head.
        Shape: (B, embed_dim)
        """
        cls_token, _ = self.forward_features(x, return_attention=False)
        return cls_token


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def build_vit(config: Optional[ViTConfig] = None) -> VisionTransformer:
    """Baseline Vision Transformer with linear patch embedding."""
    return VisionTransformer(config=config, use_kronecker=False)


def build_kvit(config: Optional[ViTConfig] = None) -> VisionTransformer:
    """Kronecker-ViT with Kronecker Patch Embedding."""
    return VisionTransformer(config=config, use_kronecker=True)


if __name__ == "__main__":
    cfg = ViTConfig()
    vit = build_vit(cfg)
    kvit = build_kvit(cfg)

    x = torch.randn(2, 3, 32, 32)
    print("=== Baseline ViT ===")
    logits = vit(x)
    print(f"Output shape : {logits.shape}")
    print(f"Parameters   : {vit.count_parameters():,}")

    print("\n=== Kronecker-ViT ===")
    logits, attns = kvit(x, return_attention=True)
    print(f"Output shape : {logits.shape}")
    print(f"Parameters   : {kvit.count_parameters():,}")
    print(f"Attention maps: {len(attns)} layers, each {attns[0].shape}")
