"""HuBERT/ContentVec feature extractor using transformers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import HubertModel

logger = logging.getLogger(__name__)


class HuBERTFeatureExtractor(nn.Module):
    """
    ContentVec feature extractor based on HuBERT.

    Uses the transformers library version for better compatibility.
    Supports both v1 (256-dim) and v2 (768-dim) output modes.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize HuBERT feature extractor.

        Args:
            model_path: Path to local model weights (optional, uses HF hub if None)
            device: Device to run on
            dtype: Data type for inference
        """
        super().__init__()
        self.device = device
        self.dtype = dtype

        # Load from HuggingFace Hub or local path
        if model_path and Path(model_path).exists():
            logger.info(f"Loading HuBERT from local path: {model_path}")
            # Load from local weights
            self.model = HubertModel.from_pretrained(
                "lengyue233/content-vec-best",
                local_files_only=False,
            )
            # If we have local weights, load them
            if Path(model_path).is_file():
                state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
                # Handle different state dict formats
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                self.model.load_state_dict(state_dict, strict=False)
        else:
            logger.info("Loading HuBERT from HuggingFace Hub: lengyue233/content-vec-best")
            self.model = HubertModel.from_pretrained("lengyue233/content-vec-best")

        self.model.to(device).to(dtype)
        self.model.eval()

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Final projection layer for v1 models (768 -> 256)
        # This mimics the final_proj in fairseq HuBERT
        self.final_proj = nn.Linear(768, 256)
        self._load_final_proj(model_path)
        self.final_proj.to(device).to(dtype)
        self.final_proj.eval()

        for param in self.final_proj.parameters():
            param.requires_grad = False

    def _load_final_proj(self, model_path: Optional[str]) -> None:
        """Load final_proj weights from checkpoint if available."""
        if model_path and Path(model_path).exists():
            try:
                state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

                # Try different key patterns
                proj_weight = None
                proj_bias = None

                # Pattern 1: Direct keys
                if "final_proj.weight" in state_dict:
                    proj_weight = state_dict["final_proj.weight"]
                    proj_bias = state_dict.get("final_proj.bias")
                # Pattern 2: Under "model" key
                elif "model" in state_dict:
                    model_dict = state_dict["model"]
                    if "final_proj.weight" in model_dict:
                        proj_weight = model_dict["final_proj.weight"]
                        proj_bias = model_dict.get("final_proj.bias")

                if proj_weight is not None:
                    self.final_proj.weight.data.copy_(proj_weight)
                    if proj_bias is not None:
                        self.final_proj.bias.data.copy_(proj_bias)
                    logger.info("Loaded final_proj weights from checkpoint")
                else:
                    logger.info("No final_proj in checkpoint, using random initialization")
                    # Initialize with a reasonable projection
                    nn.init.xavier_uniform_(self.final_proj.weight)
                    nn.init.zeros_(self.final_proj.bias)
            except Exception as e:
                logger.warning(f"Failed to load final_proj: {e}")
                nn.init.xavier_uniform_(self.final_proj.weight)
                nn.init.zeros_(self.final_proj.bias)
        else:
            # No local model, initialize randomly
            nn.init.xavier_uniform_(self.final_proj.weight)
            nn.init.zeros_(self.final_proj.bias)

    @torch.no_grad()
    def extract(
        self,
        audio: torch.Tensor,
        output_layer: int = 12,
        output_dim: int = 768,
    ) -> torch.Tensor:
        """
        Extract ContentVec features from audio.

        Args:
            audio: Input audio tensor [B, T] at 16kHz
            output_layer: Which transformer layer to extract features from
            output_dim: Output dimension (768 for v2, 256 for v1)

        Returns:
            features: Feature tensor [B, T', output_dim] where T' = T // 320
        """
        # Ensure correct shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # IMPORTANT: Keep audio as float32 for model input
        # Let torch.autocast (in calling code) handle dtype conversion internally
        # This ensures numerical stability in layer norm and convolutions
        audio = audio.to(self.device).float()

        # Extract features
        outputs = self.model(
            audio,
            output_hidden_states=True,
        )

        # Get features from specified layer
        hidden_states = outputs.hidden_states[output_layer]

        # Apply final projection for v1 (256-dim output)
        if output_dim == 256:
            hidden_states = self.final_proj(hidden_states)

        return hidden_states

    def forward(self, audio: torch.Tensor, output_dim: int = 768) -> torch.Tensor:
        """Forward pass - alias for extract."""
        return self.extract(audio, output_dim=output_dim)
