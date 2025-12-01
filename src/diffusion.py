"""
Minimal Gaussian Diffusion implementation for tabular data.

This is a simplified version focusing on numerical features only.
Based on DDPM (Ho et al., 2020) and TabDDPM (Kotelnikov et al., 2023).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear noise schedule from beta_start to beta_end."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule as proposed in https://arxiv.org/abs/2102.09672
    Often works better than linear schedule.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion process for continuous/numerical features.

    Forward process: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    Reverse process: p(x_{t-1} | x_t) learned by neural network
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        # Define beta schedule
        if beta_schedule == "linear":
            betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Precompute diffusion parameters
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (not parameters, but should move to device with model)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract coefficients at specified timesteps and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: sample x_t given x_0.

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Args:
            x_0: Original data [batch_size, num_features]
            t: Timesteps [batch_size]
            noise: Optional pre-sampled noise

        Returns:
            x_t: Noised data at timestep t
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def q_posterior_mean_variance(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance of p(x_{t-1} | x_t) using the learned model.

        The model predicts the noise, and we use it to estimate x_0,
        then compute the posterior.
        """
        # Model predicts noise
        predicted_noise = model(x_t, t, y)

        # Estimate x_0 from x_t and predicted noise
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        x_0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t

        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

        # Get posterior mean and variance
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(
            x_0=x_0_pred, x_t=x_t, t=t
        )

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from p(x_{t-1} | x_t) - single denoising step.
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            model, x_t, t, y, clip_denoised
        )

        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        x_t_minus_1 = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return x_t_minus_1

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        y: Optional[torch.Tensor] = None,
        device: str = "cpu",
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples by running the full reverse diffusion process.

        Args:
            model: Trained denoising model
            shape: Shape of samples to generate (batch_size, num_features)
            y: Optional class labels for conditional generation
            device: Device to generate samples on
            clip_denoised: Whether to clip x_0 predictions to [-1, 1]

        Returns:
            Generated samples
        """
        batch_size = shape[0]

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Reverse diffusion: t = T-1, T-2, ..., 0
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, y, clip_denoised)

        return x

    def training_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the training loss (simplified DDPM loss).

        L_simple = E[||epsilon - epsilon_theta(x_t, t)||^2]

        Args:
            model: Denoising model
            x_0: Original data
            y: Optional labels for conditional generation

        Returns:
            MSE loss between true noise and predicted noise
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

        # Sample noise and create noisy data
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)

        # Predict noise
        predicted_noise = model(x_t, t, y)

        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)

        return loss
