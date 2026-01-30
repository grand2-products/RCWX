"""HuBERT feature extractor - direct loading without fairseq.

This module loads the RVC hubert_base.pt checkpoint directly,
without requiring the fairseq package.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConvFeatureExtractor(nn.Module):
    """Convolutional feature extractor (matches HuBERT architecture)."""

    def __init__(
        self,
        conv_layers: List[tuple],
        dropout: float = 0.0,
        conv_bias: bool = False,
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList()

        for i, (in_c, out_c, k, s) in enumerate(conv_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_c, out_c, k, stride=s, bias=conv_bias),
                    nn.Dropout(dropout),
                    nn.GroupNorm(out_c, out_c) if i == 0 else nn.GELU(),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T] -> [B, 1, T]
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x  # [B, C, T']


class TransformerEncoder(nn.Module):
    """Transformer encoder layer."""

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=padding_mask)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x

        return x


class HuBERTModel(nn.Module):
    """HuBERT model architecture (simplified for inference)."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Convolutional feature extractor (7 layers)
        conv_layers = [
            (1, 512, 10, 5),     # Layer 0
            (512, 512, 3, 2),   # Layer 1
            (512, 512, 3, 2),   # Layer 2
            (512, 512, 3, 2),   # Layer 3
            (512, 512, 3, 2),   # Layer 4
            (512, 512, 2, 2),   # Layer 5
            (512, 512, 2, 2),   # Layer 6
        ]
        self.feature_extractor = ConvFeatureExtractor(conv_layers)

        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, embed_dim),
            nn.Dropout(dropout),
        )

        # Positional encoding (convolutional)
        self.pos_conv = nn.Conv1d(embed_dim, embed_dim, 128, padding=64, groups=16)

        # Layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoder(embed_dim, ffn_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Final projection (for ContentVec/v1 models)
        self.final_proj = nn.Linear(embed_dim, 256)

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        output_layer: int = 12,
    ) -> tuple:
        """Extract features from audio.

        Args:
            source: Audio tensor [B, T]
            padding_mask: Padding mask [B, T] (True = padded)
            output_layer: Which layer to extract from (1-indexed)

        Returns:
            Tuple of (features, padding_mask)
        """
        # Convolutional features
        features = self.feature_extractor(source)  # [B, 512, T']
        features = features.transpose(1, 2)  # [B, T', 512]

        # Project to embed_dim
        features = self.feature_projection(features)  # [B, T', embed_dim]

        # Add positional encoding
        features_t = features.transpose(1, 2)  # [B, embed_dim, T']
        pos = self.pos_conv(features_t)
        pos = pos[:, :, :features_t.size(2)]  # Trim to match size
        features = features + pos.transpose(1, 2)

        # Layer norm
        features = self.layer_norm(features)

        # Transformer layers
        for i, layer in enumerate(self.layers):
            features = layer(features, padding_mask)
            if i + 1 == output_layer:
                break

        return (features,)  # Return tuple for compatibility


class HuBERTFairseq(nn.Module):
    """
    HuBERT feature extractor that loads RVC's hubert_base.pt directly.

    This doesn't require fairseq - it loads the checkpoint and maps weights
    to a compatible model architecture.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """Load HuBERT model from checkpoint."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"HuBERT model not found: {model_path}")

        logger.info(f"Loading HuBERT from: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Get model weights
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Create model
        self.model = HuBERTModel()

        # Map weights from fairseq format to our format
        self._load_weights(state_dict)

        self.model.to(self.device)
        if self.dtype == torch.float16:
            self.model.half()
        else:
            self.model.float()

        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        logger.info("HuBERT loaded successfully (direct loading)")

    def _load_weights(self, state_dict: dict) -> None:
        """Map fairseq weights to our model."""
        new_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Map feature extractor
            if key.startswith("feature_extractor.conv_layers"):
                # fairseq: feature_extractor.conv_layers.0.0.weight
                # ours: feature_extractor.conv_layers.0.0.weight
                new_key = key

            # Map feature projection
            elif key.startswith("layer_norm."):
                new_key = key.replace("layer_norm.", "feature_projection.0.")
            elif key.startswith("post_extract_proj."):
                new_key = key.replace("post_extract_proj.", "feature_projection.1.")

            # Map positional conv
            elif key.startswith("encoder.pos_conv.0."):
                new_key = key.replace("encoder.pos_conv.0.", "pos_conv.")

            # Map layer norm
            elif key == "encoder.layer_norm.weight":
                new_key = "layer_norm.weight"
            elif key == "encoder.layer_norm.bias":
                new_key = "layer_norm.bias"

            # Map transformer layers
            elif key.startswith("encoder.layers."):
                new_key = key.replace("encoder.layers.", "layers.")
                # Map attention
                new_key = new_key.replace(".self_attn.k_proj.", ".self_attn.in_proj_")
                new_key = new_key.replace(".self_attn.v_proj.", ".self_attn.in_proj_")
                new_key = new_key.replace(".self_attn.q_proj.", ".self_attn.in_proj_")
                new_key = new_key.replace(".self_attn.out_proj.", ".self_attn.out_proj.")

            # Map final proj
            elif key.startswith("final_proj."):
                new_key = key

            if new_key != key:
                logger.debug(f"Mapping: {key} -> {new_key}")

            new_state_dict[new_key] = value

        # Load with strict=False to handle any unmapped weights
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        if missing:
            logger.warning(f"Missing weights: {missing[:5]}...")
        if unexpected:
            logger.debug(f"Unexpected weights: {unexpected[:5]}...")

    @torch.no_grad()
    def extract(
        self,
        audio: torch.Tensor,
        output_layer: int = 12,
        output_dim: int = 768,
    ) -> torch.Tensor:
        """Extract HuBERT features."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device)
        if self.dtype == torch.float16:
            audio = audio.half()
        else:
            audio = audio.float()

        # Extract features
        features = self.model.extract_features(
            source=audio,
            padding_mask=None,
            output_layer=output_layer,
        )[0]

        # Apply final projection for v1 (256-dim)
        if output_dim == 256:
            features = self.model.final_proj(features)

        return features

    def forward(self, audio: torch.Tensor, output_dim: int = 768) -> torch.Tensor:
        return self.extract(audio, output_dim=output_dim)
