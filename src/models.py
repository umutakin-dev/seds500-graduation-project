"""
Neural network models for tabular diffusion.

Simplified MLP-based denoiser inspired by TabDDPM.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    This is the standard positional encoding from "Attention Is All You Need",
    adapted for timesteps in diffusion models.

    Args:
        timesteps: 1-D Tensor of N indices, one per batch element
        dim: Dimension of the output embedding
        max_period: Controls the minimum frequency of the embeddings

    Returns:
        Tensor of shape [N, dim] with positional embeddings
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class SiLU(nn.Module):
    """Sigmoid Linear Unit (SiLU) / Swish activation function."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class MLPBlock(nn.Module):
    """Basic MLP block: Linear -> Activation -> Dropout."""

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.linear(x)))


class MLPDenoiser(nn.Module):
    """
    MLP-based denoising network for tabular diffusion.

    Architecture:
        Input: x (noisy data) + timestep embedding + optional label embedding
        -> Linear projection to hidden dim
        -> Stack of MLP blocks
        -> Linear output to predict noise

    This is a simplified version of TabDDPM's MLPDiffusion.
    """

    def __init__(
        self,
        d_in: int,
        hidden_dims: List[int] = [256, 256, 256],
        dropout: float = 0.0,
        num_classes: int = 0,
        dim_t: int = 128,
    ):
        """
        Args:
            d_in: Input dimension (number of features)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            num_classes: Number of classes for conditional generation (0 = unconditional)
            dim_t: Dimension of timestep embedding
        """
        super().__init__()
        self.d_in = d_in
        self.dim_t = dim_t
        self.num_classes = num_classes

        # Time embedding network
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            SiLU(),
            nn.Linear(dim_t, dim_t),
        )

        # Optional class embedding
        if num_classes > 0:
            self.label_embed = nn.Embedding(num_classes, dim_t)
        else:
            self.label_embed = None

        # Input projection: project input + add time/label embeddings
        self.input_proj = nn.Linear(d_in, dim_t)

        # MLP blocks
        dims = [dim_t] + hidden_dims
        self.blocks = nn.ModuleList([
            MLPBlock(dims[i], dims[i + 1], dropout)
            for i in range(len(dims) - 1)
        ])

        # Output projection: predict noise (same dimension as input)
        self.output_proj = nn.Linear(hidden_dims[-1], d_in)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the denoiser.

        Args:
            x: Noisy input data [batch_size, d_in]
            t: Timesteps [batch_size]
            y: Optional class labels [batch_size]

        Returns:
            Predicted noise [batch_size, d_in]
        """
        # Get timestep embedding
        t_emb = self.time_embed(timestep_embedding(t, self.dim_t))

        # Add label embedding if conditional
        if self.label_embed is not None and y is not None:
            y_emb = self.label_embed(y)
            t_emb = t_emb + y_emb

        # Project input and add embeddings
        h = self.input_proj(x) + t_emb

        # Pass through MLP blocks
        for block in self.blocks:
            h = block(h)

        # Output projection
        out = self.output_proj(h)

        return out


class ResidualMLPBlock(nn.Module):
    """MLP block with residual connection."""

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x + residual


class ResidualMLPDenoiser(nn.Module):
    """
    Residual MLP denoiser with skip connections.

    Similar to MLPDenoiser but with residual blocks for better gradient flow.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = 256,
        d_hidden: int = 512,
        n_blocks: int = 4,
        dropout: float = 0.0,
        num_classes: int = 0,
        dim_t: int = 128,
    ):
        super().__init__()
        self.d_in = d_in
        self.dim_t = dim_t
        self.num_classes = num_classes

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            SiLU(),
            nn.Linear(dim_t, dim_t),
        )

        # Optional class embedding
        if num_classes > 0:
            self.label_embed = nn.Embedding(num_classes, dim_t)
        else:
            self.label_embed = None

        # Input projection
        self.input_proj = nn.Linear(d_in, d_model)
        self.time_proj = nn.Linear(dim_t, d_model)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(d_model, d_hidden, dropout)
            for _ in range(n_blocks)
        ])

        # Output
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_in)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get embeddings
        t_emb = self.time_embed(timestep_embedding(t, self.dim_t))

        if self.label_embed is not None and y is not None:
            t_emb = t_emb + self.label_embed(y)

        # Project and combine
        h = self.input_proj(x) + self.time_proj(t_emb)

        # Residual blocks
        for block in self.blocks:
            h = block(h)

        # Output
        h = self.output_norm(h)
        out = self.output_proj(h)

        return out
