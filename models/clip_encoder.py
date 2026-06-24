"""CLIP ViT Encoder for image feature extraction.

Uses a frozen HuggingFace CLIP vision model as the backbone.
Provides both global features (CLS token) and spatial features (patch tokens).
Only the projection layer is trainable.
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel
from typing import Optional


class CLIPEncoder(nn.Module):
    """CLIP ViT-based encoder for image captioning.

    Loads a pretrained CLIP vision model, freezes it entirely,
    and projects features to the target embed_dim.

    Args:
        clip_path: path to local CLIP checkpoint directory
        embed_dim: output feature dimension for the decoder
        freeze: whether to freeze CLIP weights (should always be True)
    """

    def __init__(
        self,
        clip_path: str,
        embed_dim: int = 512,
        freeze: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Load pretrained CLIP vision model
        self.clip = CLIPVisionModel.from_pretrained(clip_path)

        # CLIP ViT-L/14 hidden size is 1024, projection_dim is 768
        clip_dim = self.clip.config.hidden_size  # 1024 for ViT-L/14

        # Trainable projection: CLIP hidden -> embed_dim
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Spatial projection (for attention model, same architecture)
        self.spatial_projection = nn.Sequential(
            nn.Linear(clip_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

        if freeze:
            self._freeze()

    def _freeze(self):
        """Freeze all CLIP parameters."""
        for param in self.clip.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract global features: [batch, embed_dim].

        Uses the CLS token output from CLIP's vision encoder.
        """
        outputs = self.clip(pixel_values=images)
        # last_hidden_state: [B, num_patches+1, hidden_size] (CLS + 256 patches)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]
        features = self.projection(cls_token)             # [B, embed_dim]
        return features

    def forward_spatial(self, images: torch.Tensor) -> torch.Tensor:
        """Extract spatial features: [batch, num_patches, embed_dim].

        Returns patch token embeddings (excluding CLS token).
        For ViT-L/14 with 224x224 input: [B, 256, embed_dim]
        """
        outputs = self.clip(pixel_values=images)
        # Exclude CLS token (index 0), keep only patch tokens
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # [B, num_patches, hidden_size]
        features = self.spatial_projection(patch_tokens)      # [B, num_patches, embed_dim]
        return features
