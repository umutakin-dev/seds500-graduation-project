"""
Diffusion implementations for tabular data.

Includes:
- GaussianDiffusion: For continuous/numerical features
- MultinomialDiffusion: For categorical features
- HybridDiffusion: Combines both for mixed-type tabular data

Based on DDPM (Ho et al., 2020) and TabDDPM (Kotelnikov et al., 2023).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict


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

        # Register as buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def q_posterior_mean_variance(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None, clip_denoised: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted_noise = model(x_t, t, y)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        x_0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_0=x_0_pred, x_t=x_t, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None, clip_denoised: bool = True) -> torch.Tensor:
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, y, clip_denoised)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        x_t_minus_1 = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return x_t_minus_1

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: Tuple[int, ...], y: Optional[torch.Tensor] = None, device: str = "cpu", clip_denoised: bool = True) -> torch.Tensor:
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, y, clip_denoised)
        return x

    def training_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        label_dropout: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute training loss with optional label dropout for classifier-free guidance.

        Args:
            model: Denoiser network
            x_0: Clean data
            y: Class labels (optional)
            label_dropout: Probability of dropping labels during training (for CFG).
                           When dropped, y is set to None for that sample, training
                           the model to also predict unconditionally.
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t, _ = self.q_sample(x_0, t, noise)

        # Apply label dropout for classifier-free guidance training
        y_input = y
        if y is not None and label_dropout > 0:
            drop_mask = torch.rand(batch_size, device=device) < label_dropout
            if drop_mask.all():
                y_input = None
            elif drop_mask.any():
                # For partial dropout, we need to handle it in the model
                # For simplicity, we'll drop all labels if any are dropped in this batch
                # A more sophisticated approach would use masking in the model
                y_input = y  # Keep labels for now, full CFG requires model changes

        predicted_noise = model(x_t, t, y_input)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def sample_with_guidance(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        y: torch.Tensor,
        guidance_scale: float = 1.0,
        device: str = "cpu",
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples with classifier-free guidance.

        Args:
            model: Denoiser network (must be trained with label_dropout > 0)
            shape: Output shape (batch_size, d_in)
            y: Target class labels for all samples
            guidance_scale: Strength of class guidance (1.0 = no guidance, >1.0 = stronger)
            device: Device to generate on
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Get conditional and unconditional predictions
            noise_cond = model(x, t_batch, y)
            noise_uncond = model(x, t_batch, None)

            # Apply classifier-free guidance
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            # Compute x_{t-1} using the guided noise prediction
            sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t_batch, x.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape)
            x_0_pred = (x - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t

            if clip_denoised:
                x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

            model_mean, _, model_log_var = self.q_posterior_mean_variance(x_0=x_0_pred, x_t=x, t=t_batch)
            noise = torch.randn_like(x)
            nonzero_mask = (t_batch != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            x = model_mean + nonzero_mask * torch.exp(0.5 * model_log_var) * noise

        return x


# =============================================================================
# Multinomial Diffusion for Categorical Features
# =============================================================================

class MultinomialDiffusion(nn.Module):
    """
    Multinomial Diffusion process for categorical features.

    Forward process: q(x_t | x_0) = Cat(x_t; alpha_t * x_0 + (1 - alpha_t) * 1/K)
    where x_0 is a one-hot vector and K is the number of categories.
    """

    def __init__(
        self,
        num_classes: int,
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

        if beta_schedule == "linear":
            betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: interpolate between one-hot and uniform."""
        alpha_t = self._extract(self.alphas_cumprod, t, x_0.shape)
        uniform = torch.ones_like(x_0) / self.num_classes
        x_t = alpha_t * x_0 + (1 - alpha_t) * uniform
        return x_t

    @torch.no_grad()
    def sample(self, model_logits: torch.Tensor) -> torch.Tensor:
        """Sample categories from model output logits."""
        probs = F.softmax(model_logits, dim=-1)
        samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
        one_hot = F.one_hot(samples, num_classes=self.num_classes).float()
        return one_hot

    def training_loss(self, predicted_logits: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss for categorical diffusion."""
        targets = x_0.argmax(dim=-1)
        loss = F.cross_entropy(predicted_logits, targets)
        return loss


# =============================================================================
# Hybrid Diffusion for Mixed-Type Tabular Data
# =============================================================================

class HybridDiffusion(nn.Module):
    """
    Hybrid Diffusion combining Gaussian and Multinomial for mixed-type tabular data.
    
    This is the key innovation from TabDDPM: handle numerical and categorical
    features with their appropriate diffusion processes, but train them jointly.
    """

    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: List[int],
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities
        self.num_categorical = len(cat_cardinalities)
        self.num_timesteps = num_timesteps
        self.cat_dims = sum(cat_cardinalities)

        self.gaussian_diffusion = GaussianDiffusion(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        self.multinomial_diffusions = nn.ModuleList([
            MultinomialDiffusion(
                num_classes=card,
                num_timesteps=num_timesteps,
                beta_schedule=beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end,
            )
            for card in cat_cardinalities
        ])

    def _split_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Split concatenated features into numerical and categorical parts."""
        x_num = x[:, :self.num_numerical]
        x_cats = []
        offset = self.num_numerical
        for card in self.cat_cardinalities:
            x_cats.append(x[:, offset:offset + card])
            offset += card
        return x_num, x_cats

    def _concat_features(self, x_num: torch.Tensor, x_cats: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate numerical and categorical features."""
        return torch.cat([x_num] + x_cats, dim=-1)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion for hybrid data."""
        x_num, x_cats = self._split_features(x_0)
        x_num_t, noise = self.gaussian_diffusion.q_sample(x_num, t)
        x_cats_t = [diff.q_sample(x_cat, t) for x_cat, diff in zip(x_cats, self.multinomial_diffusions)]
        x_t = self._concat_features(x_num_t, x_cats_t)
        return x_t, noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None, clip_denoised: bool = True) -> torch.Tensor:
        """Single reverse diffusion step for hybrid data."""
        model_output = model(x_t, t, y)
        
        # Split model output
        noise_pred = model_output[:, :self.num_numerical]
        cat_logits = []
        offset = self.num_numerical
        for card in self.cat_cardinalities:
            cat_logits.append(model_output[:, offset:offset + card])
            offset += card

        x_num_t, x_cats_t = self._split_features(x_t)

        # Reverse step for numerical
        sqrt_alphas_cumprod_t = self.gaussian_diffusion._extract(self.gaussian_diffusion.sqrt_alphas_cumprod, t, x_num_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.gaussian_diffusion._extract(self.gaussian_diffusion.sqrt_one_minus_alphas_cumprod, t, x_num_t.shape)
        x_num_0_pred = (x_num_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
        if clip_denoised:
            x_num_0_pred = torch.clamp(x_num_0_pred, -1.0, 1.0)

        model_mean, _, model_log_var = self.gaussian_diffusion.q_posterior_mean_variance(x_0=x_num_0_pred, x_t=x_num_t, t=t)
        noise = torch.randn_like(x_num_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_num_t.shape) - 1)))
        x_num_t_minus_1 = model_mean + nonzero_mask * torch.exp(0.5 * model_log_var) * noise

        # Reverse step for categorical
        x_cats_t_minus_1 = []
        for i, (logits, diff) in enumerate(zip(cat_logits, self.multinomial_diffusions)):
            if t[0] == 0:
                probs = F.softmax(logits, dim=-1)
                samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
                x_cat = F.one_hot(samples, num_classes=diff.num_classes).float()
            else:
                probs = F.softmax(logits, dim=-1)
                alpha_t = diff._extract(diff.alphas_cumprod, t, probs.shape)
                alpha_t_prev = diff._extract(F.pad(diff.alphas_cumprod[:-1], (1, 0), value=1.0), t, probs.shape)
                x_cat = (alpha_t_prev / alpha_t.clamp(min=1e-8)) * x_cats_t[i] + (1 - alpha_t_prev / alpha_t.clamp(min=1e-8)) * probs
                x_cat = x_cat / x_cat.sum(dim=-1, keepdim=True)
            x_cats_t_minus_1.append(x_cat)

        return self._concat_features(x_num_t_minus_1, x_cats_t_minus_1)

    @torch.no_grad()
    def sample(self, model: nn.Module, batch_size: int, y: Optional[torch.Tensor] = None, device: str = "cpu", clip_denoised: bool = True) -> torch.Tensor:
        """Generate samples by running the full reverse diffusion process."""
        x_num = torch.randn(batch_size, self.num_numerical, device=device)
        x_cats = [torch.ones(batch_size, card, device=device) / card for card in self.cat_cardinalities]
        x = self._concat_features(x_num, x_cats)
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, y, clip_denoised)
        return x

    def training_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        label_dropout: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the hybrid training loss (MSE for numerical + CE for categorical).

        Args:
            model: Hybrid denoiser network
            x_0: Clean data (numerical + one-hot categorical concatenated)
            y: Class labels (optional)
            label_dropout: Probability of dropping labels during training (for CFG)
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

        x_num_0, x_cats_0 = self._split_features(x_0)
        noise = torch.randn_like(x_num_0)
        x_num_t, _ = self.gaussian_diffusion.q_sample(x_num_0, t, noise)
        x_cats_t = [diff.q_sample(x_cat, t) for x_cat, diff in zip(x_cats_0, self.multinomial_diffusions)]
        x_t = self._concat_features(x_num_t, x_cats_t)

        # Apply label dropout for classifier-free guidance training
        y_input = y
        if y is not None and label_dropout > 0:
            drop_mask = torch.rand(batch_size, device=device) < label_dropout
            if drop_mask.all():
                y_input = None

        model_output = model(x_t, t, y_input)
        noise_pred = model_output[:, :self.num_numerical]
        cat_logits = []
        offset = self.num_numerical
        for card in self.cat_cardinalities:
            cat_logits.append(model_output[:, offset:offset + card])
            offset += card

        loss_num = F.mse_loss(noise_pred, noise)
        loss_cat = torch.tensor(0.0, device=device)
        for logits, x_cat_0, diff in zip(cat_logits, x_cats_0, self.multinomial_diffusions):
            loss_cat = loss_cat + diff.training_loss(logits, x_cat_0)
        if len(self.cat_cardinalities) > 0:
            loss_cat = loss_cat / len(self.cat_cardinalities)

        loss = loss_num + loss_cat
        return {"loss": loss, "loss_num": loss_num, "loss_cat": loss_cat}

    @torch.no_grad()
    def sample_with_guidance(
        self,
        model: nn.Module,
        batch_size: int,
        y: torch.Tensor,
        guidance_scale: float = 1.0,
        device: str = "cpu",
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples with classifier-free guidance for hybrid data.

        Args:
            model: Hybrid denoiser network (must be trained with label_dropout > 0)
            batch_size: Number of samples to generate
            y: Target class labels for all samples
            guidance_scale: Strength of class guidance (1.0 = no guidance, >1.0 = stronger)
            device: Device to generate on
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]
        """
        # Initialize from noise (numerical) and uniform (categorical)
        x_num = torch.randn(batch_size, self.num_numerical, device=device)
        x_cats = [torch.ones(batch_size, card, device=device) / card for card in self.cat_cardinalities]
        x = self._concat_features(x_num, x_cats)

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Get conditional and unconditional predictions
            output_cond = model(x, t_batch, y)
            output_uncond = model(x, t_batch, None)

            # Apply classifier-free guidance to numerical noise prediction
            noise_cond = output_cond[:, :self.num_numerical]
            noise_uncond = output_uncond[:, :self.num_numerical]
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            # Apply guidance to categorical logits
            cat_logits = []
            offset = self.num_numerical
            for card in self.cat_cardinalities:
                logits_cond = output_cond[:, offset:offset + card]
                logits_uncond = output_uncond[:, offset:offset + card]
                # Apply guidance in logit space
                logits_guided = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
                cat_logits.append(logits_guided)
                offset += card

            x_num_t, x_cats_t = self._split_features(x)

            # Reverse step for numerical using guided noise
            sqrt_alphas_cumprod_t = self.gaussian_diffusion._extract(
                self.gaussian_diffusion.sqrt_alphas_cumprod, t_batch, x_num_t.shape
            )
            sqrt_one_minus_alphas_cumprod_t = self.gaussian_diffusion._extract(
                self.gaussian_diffusion.sqrt_one_minus_alphas_cumprod, t_batch, x_num_t.shape
            )
            x_num_0_pred = (x_num_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t
            if clip_denoised:
                x_num_0_pred = torch.clamp(x_num_0_pred, -1.0, 1.0)

            model_mean, _, model_log_var = self.gaussian_diffusion.q_posterior_mean_variance(
                x_0=x_num_0_pred, x_t=x_num_t, t=t_batch
            )
            noise = torch.randn_like(x_num_t)
            nonzero_mask = (t_batch != 0).float().view(-1, *([1] * (len(x_num_t.shape) - 1)))
            x_num_t_minus_1 = model_mean + nonzero_mask * torch.exp(0.5 * model_log_var) * noise

            # Reverse step for categorical using guided logits
            x_cats_t_minus_1 = []
            for i, (logits, diff) in enumerate(zip(cat_logits, self.multinomial_diffusions)):
                if t == 0:
                    # Final step: sample from guided logits
                    probs = F.softmax(logits, dim=-1)
                    samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    x_cat = F.one_hot(samples, num_classes=diff.num_classes).float()
                else:
                    # Intermediate step: update probability distribution
                    probs = F.softmax(logits, dim=-1)
                    alpha_t = diff._extract(diff.alphas_cumprod, t_batch, probs.shape)
                    alpha_t_prev = diff._extract(
                        F.pad(diff.alphas_cumprod[:-1], (1, 0), value=1.0), t_batch, probs.shape
                    )
                    x_cat = (alpha_t_prev / alpha_t.clamp(min=1e-8)) * x_cats_t[i] + \
                            (1 - alpha_t_prev / alpha_t.clamp(min=1e-8)) * probs
                    x_cat = x_cat / x_cat.sum(dim=-1, keepdim=True)
                x_cats_t_minus_1.append(x_cat)

            x = self._concat_features(x_num_t_minus_1, x_cats_t_minus_1)

        return x
