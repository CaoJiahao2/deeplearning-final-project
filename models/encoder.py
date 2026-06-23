"""CNN Encoder for image feature extraction.

Two modes:
- Global features: [batch, embed_dim] from adaptive avg pool (for baseline)
- Spatial features: [batch, num_pixels, embed_dim] from intermediate layer (for attention)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple


class CNNEncoder(nn.Module):
    """ResNet-based CNN encoder for image captioning.

    Args:
        backbone: 'resnet50' or 'resnet101'
        embed_dim: output feature dimension
        pretrained: use ImageNet pretrained weights
        fine_tune_layers: which layers to fine-tune ('all', 'layer4', or 'none')
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        embed_dim: int = 512,
        pretrained: bool = True,
        fine_tune_layers: str = "layer4",
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Load pretrained backbone
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = models.resnet50(weights=weights)
        elif backbone == "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = models.resnet101(weights=weights)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Split ResNet into modules for flexible feature extraction
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # /4,  256 channels
        self.layer2 = resnet.layer2  # /8,  512 channels
        self.layer3 = resnet.layer3  # /16, 1024 channels
        self.layer4 = resnet.layer4  # /32, 2048 channels

        self.backbone_out_dim = 2048  # ResNet50/101 layer4 output channels

        # Projection to embed_dim
        self.projection = nn.Sequential(
            nn.Linear(self.backbone_out_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

        # Spatial projection for attention model
        self.spatial_projection = nn.Sequential(
            nn.Linear(self.backbone_out_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

        # Freeze/unfreeze layers
        self._set_fine_tune(fine_tune_layers)

    def _set_fine_tune(self, fine_tune_layers: str):
        """Control which layers are trainable."""
        # Freeze everything first
        for param in self.parameters():
            param.requires_grad = False

        if fine_tune_layers == "none":
            return
        elif fine_tune_layers == "all":
            for param in self.parameters():
                param.requires_grad = True
        elif fine_tune_layers == "layer4":
            for param in self.layer4.parameters():
                param.requires_grad = True
            for param in self.projection.parameters():
                param.requires_grad = True
            for param in self.spatial_projection.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown fine_tune_layers: {fine_tune_layers}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract global features: [batch, embed_dim]."""
        features = self._extract_features(images)  # [B, 2048, 7, 7]
        features = features.mean(dim=[2, 3])        # [B, 2048]
        features = self.projection(features)          # [B, embed_dim]
        return features

    def forward_spatial(self, images: torch.Tensor) -> torch.Tensor:
        """Extract spatial features: [batch, num_pixels, embed_dim]."""
        features = self._extract_features(images)    # [B, 2048, 7, 7]
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1)      # [B, 7, 7, 2048]
        features = features.reshape(B, H * W, C)     # [B, 49, 2048]
        features = self.spatial_projection(features)  # [B, 49, embed_dim]
        return features

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Run full ResNet forward up to layer4."""
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
