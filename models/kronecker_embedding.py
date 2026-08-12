"""
Kronecker Patch Embedding (KPE)

A novel patch embedding module proposed in this research prototype.

Core idea
---------
Instead of a single linear projection of the flattened patch, we:

1. Extract several *local feature vectors* from the patch via a small MLP
   (or a tiny convolutional stem).
2. Maintain a learnable *position embedding* for each local feature location
   inside the patch.
3. Form the Kronecker (outer) product of each local feature with its
   corresponding position embedding:
        e_i  ⊗  p_i
4. Sum the Kronecker products to obtain the final patch embedding:

        E_patch  =  Σ_i  (e_i ⊗ p_i)

This realises a *content × position* composition that is the hallmark of
classical Kronecker embeddings, while remaining fully differentiable and
GPU-friendly.

Mathematical formulation
------------------------
Let a patch be reshaped / projected into K local feature vectors
e_i ∈ ℝ^{d_f}  (i = 1 … K).

Let p_i ∈ ℝ^{d_p} be the corresponding learnable position vectors.

The Kronecker product e_i ⊗ p_i ∈ ℝ^{d_f · d_p} is the vectorised outer product.

The patch embedding is the sum:

    E = Σ_{i=1}^K  (e_i ⊗ p_i)   ∈ ℝ^{d_f · d_p}

We choose d_f = 8, d_p = 16 ⇒ d_f · d_p = 128, matching the transformer
embedding dimension used throughout the project.

Implementation notes
--------------------
- We never call torch.kron on raw pixels.
- The local features are produced by a two-layer MLP applied to the
  flattened patch (48 → 64 → K·d_f).
- Position embeddings are shared across all patches (they encode relative
  location *inside* a patch, not absolute image coordinates).
- Absolute patch positions are still added later by the standard ViT
  positional embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KroneckerPatchEmbeddingConfig:
    """Hyper-parameters of the Kronecker Patch Embedding."""
    img_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    embed_dim: int = 128          # must equal local_feat_dim * pos_dim
    local_feat_dim: int = 8
    pos_dim: int = 16
    num_local_features: int = 4   # K
    mlp_hidden: int = 64
    dropout: float = 0.0


class KroneckerPatchEmbedding(nn.Module):
    """
    Kronecker Patch Embedding layer.

    Parameters
    ----------
    config : KroneckerPatchEmbeddingConfig
        All hyper-parameters (see dataclass above).
    """

    def __init__(self, config: Optional[KroneckerPatchEmbeddingConfig] = None):
        super().__init__()
        self.cfg = config or KroneckerPatchEmbeddingConfig()
        cfg = self.cfg

        assert cfg.embed_dim == cfg.local_feat_dim * cfg.pos_dim, (
            f"embed_dim ({cfg.embed_dim}) must equal "
            f"local_feat_dim * pos_dim ({cfg.local_feat_dim * cfg.pos_dim})"
        )
        assert cfg.img_size % cfg.patch_size == 0

        self.img_size = cfg.img_size
        self.patch_size = cfg.patch_size
        self.num_patches = (cfg.img_size // cfg.patch_size) ** 2
        self.patch_dim = cfg.in_channels * cfg.patch_size * cfg.patch_size  # 48
        self.K = cfg.num_local_features
        self.d_f = cfg.local_feat_dim
        self.d_p = cfg.pos_dim
        self.embed_dim = cfg.embed_dim

        # ------------------------------------------------------------------
        # 1. Local feature extractor (MLP on flattened patch)
        #    48 → 64 → K * d_f
        # ------------------------------------------------------------------
        self.feature_mlp = nn.Sequential(
            nn.Linear(self.patch_dim, cfg.mlp_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, self.K * self.d_f),
        )

        # ------------------------------------------------------------------
        # 2. Learnable position embeddings for the K local locations
        #    Shape: (K, d_p)
        # ------------------------------------------------------------------
        self.local_pos_embed = nn.Parameter(
            torch.zeros(1, self.K, self.d_p)
        )

        # ------------------------------------------------------------------
        # 3. Optional final LayerNorm on the summed Kronecker embedding
        # ------------------------------------------------------------------
        self.norm = nn.LayerNorm(self.embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.feature_mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.local_pos_embed, std=0.02)

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unfold the image into non-overlapping patches.

        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        patches : (B, num_patches, patch_dim)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        # (B, C, nH, nW, p, p)
        x = x.unfold(2, p, p).unfold(3, p, p)
        # → (B, num_patches, C, p, p) → (B, num_patches, C*p*p)
        x = x.contiguous().view(B, C, -1, p * p)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, self.num_patches, -1)
        return x

    def _kronecker_sum(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Σ_i (e_i ⊗ p_i) in a vectorised, memory-efficient way.

        Parameters
        ----------
        features  : (B, N, K, d_f)   local feature vectors
        positions : (1, K, d_p)      or (B, N, K, d_p) local position vectors

        Returns
        -------
        embedding : (B, N, d_f * d_p)
        """
        # Expand positions to match batch & patch dimensions if necessary
        if positions.dim() == 3:  # (1, K, d_p)
            positions = positions.unsqueeze(0).expand(
                features.size(0), features.size(1), -1, -1
            )  # (B, N, K, d_p)

        # Outer product via broadcasting:
        # e_i[..., :, None] * p_i[..., None, :]  → (..., d_f, d_p)
        # Then flatten the last two dims and sum over the local index K.
        outer = features.unsqueeze(-1) * positions.unsqueeze(-2)  # (B, N, K, d_f, d_p)
        outer = outer.flatten(-2)                                 # (B, N, K, d_f*d_p)
        embedding = outer.sum(dim=2)                              # (B, N, embed_dim)
        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (B, C, H, W)

        Returns
        -------
        patch_embeddings : Tensor of shape (B, num_patches, embed_dim)
        """
        B = x.shape[0]

        # 1. Extract flattened patches
        patches = self._extract_patches(x)          # (B, N, 48)

        # 2. Local feature extraction
        #    (B, N, 48) → (B, N, K * d_f) → (B, N, K, d_f)
        feat = self.feature_mlp(patches)
        feat = feat.view(B, self.num_patches, self.K, self.d_f)

        # 3. Kronecker composition + sum
        emb = self._kronecker_sum(feat, self.local_pos_embed)  # (B, N, 128)

        # 4. Normalisation
        emb = self.norm(emb)
        return emb

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"patch_size={self.patch_size}, "
            f"num_patches={self.num_patches}, "
            f"K={self.K}, d_f={self.d_f}, d_p={self.d_p}, "
            f"embed_dim={self.embed_dim}"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_kronecker_patch_embedding(
    img_size: int = 32,
    patch_size: int = 4,
    embed_dim: int = 128,
    local_feat_dim: int = 8,
    pos_dim: int = 16,
    num_local_features: int = 4,
    **kwargs,
) -> KroneckerPatchEmbedding:
    """Factory function used by the training scripts."""
    cfg = KroneckerPatchEmbeddingConfig(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        local_feat_dim=local_feat_dim,
        pos_dim=pos_dim,
        num_local_features=num_local_features,
        **kwargs,
    )
    return KroneckerPatchEmbedding(cfg)


if __name__ == "__main__":
    model = build_kronecker_patch_embedding()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"KPE output shape : {out.shape}")          # (2, 64, 128)
    print(f"Parameters       : {model.count_parameters():,}")
    print(model)
