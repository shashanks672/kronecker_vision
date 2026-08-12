"""
Baseline CNN for CIFAR-10.

Architecture (as specified):
    Conv → ReLU → MaxPool →
    Conv → ReLU → MaxPool →
    Fully Connected → Softmax

This serves as the simple convolutional baseline against which
Vision Transformer and Kronecker-ViT are compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CNNConfig:
    """Configuration for the baseline CNN."""
    num_classes: int = 10
    in_channels: int = 3
    # First conv block
    conv1_out: int = 32
    conv1_kernel: int = 3
    # Second conv block
    conv2_out: int = 64
    conv2_kernel: int = 3
    # After two MaxPools (2×2) on 32×32 → 8×8 feature maps
    fc_hidden: int = 256
    dropout: float = 0.3


class BaselineCNN(nn.Module):
    """
    Simple two-stage convolutional network for CIFAR-10.

    Input : (B, 3, 32, 32)
    Output: (B, num_classes) logits
    """

    def __init__(self, config: Optional[CNNConfig] = None):
        super().__init__()
        self.config = config or CNNConfig()
        cfg = self.config

        # Block 1
        self.conv1 = nn.Conv2d(
            cfg.in_channels,
            cfg.conv1_out,
            kernel_size=cfg.conv1_kernel,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(cfg.conv1_out)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32 → 16

        # Block 2
        self.conv2 = nn.Conv2d(
            cfg.conv1_out,
            cfg.conv2_out,
            kernel_size=cfg.conv2_kernel,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(cfg.conv2_out)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16 → 8

        # After two pools: spatial size = 8×8, channels = conv2_out
        flattened = cfg.conv2_out * 8 * 8

        self.fc1 = nn.Linear(flattened, cfg.fc_hidden)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(cfg.fc_hidden, cfg.num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (B, 3, 32, 32)

        Returns
        -------
        logits : Tensor of shape (B, num_classes)
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.pool2(x)

        # Classifier
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = BaselineCNN()
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {model.count_parameters():,}")
