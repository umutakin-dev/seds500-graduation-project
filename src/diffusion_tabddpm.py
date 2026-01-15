"""
TabDDPM-style Diffusion for Tabular Data.

This implementation follows the TabDDPM paper more closely:
- Log-space operations for numerical stability
- KL divergence loss for categoricals (not cross-entropy)
- Gumbel-softmax sampling
- Proper posterior computation q(x_{t-1}|x_t, x_0)

Based on: https://github.com/rotot0/tab-ddpm

Experiment 018: Comparing this approach vs our simpler implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict


# =============================================================================
# Log-space Helper Functions (from TabDDPM utils.py)
# =============================================================================

def log_1_min_a(a: torch.Tensor) -> torch.Tensor:
    """Compute log(1 - exp(a)) in a numerically stable way."""
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute log(exp(a) + exp(b)) in a numerically stable way."""
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def log_sub_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute log(exp(a) - exp(b)) in a numerically stable way."""
    m = torch.maximum(a, b)
    return torch.log(torch.exp(a - m) - torch.exp(b - m) + 1e-40) + m


def index_to_log_onehot(x: torch.Tensor, num_classes: List[int]) -> torch.Tensor:
    """Convert category indices to log one-hot encoding.

    Args:
        x: Tensor of shape (batch, num_cat_features) with category indices
        num_classes: List of number of classes for each categorical feature

    Returns:
        Log one-hot tensor of shape (batch, sum(num_classes))
    """
    onehots = []
    for i, nc in enumerate(num_classes):
        onehots.append(F.one_hot(x[:, i].long(), nc))
    x_onehot = torch.cat(onehots, dim=1).float()
    # Convert to log space (log(1) = 0 for the selected class, log(0) = -inf for others)
    log_onehot = torch.log(x_onehot.clamp(min=1e-30))
    return log_onehot


def log_onehot_to_index(log_x: torch.Tensor, num_classes: List[int]) -> torch.Tensor:
    """Convert log one-hot encoding back to category indices."""
    indices = []
    offset = 0
    for nc in num_classes:
        indices.append(log_x[:, offset:offset + nc].argmax(dim=1))
        offset += nc
    return torch.stack(indices, dim=1)


def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    """Sum all dimensions except batch."""
    return x.reshape(x.shape[0], -1).sum(-1)


def mean_flat(x: torch.Tensor) -> torch.Tensor:
    """Mean over all dimensions except batch."""
    return x.mean(dim=list(range(1, len(x.shape))))


# =============================================================================
# Beta Schedules
# =============================================================================

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> np.ndarray:
    """Linear noise schedule."""
    scale = 1000 / timesteps
    return np.linspace(scale * beta_start, scale * beta_end, timesteps, dtype=np.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    """Cosine noise schedule as in improved DDPM."""
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0001, 0.9999)


# =============================================================================
# TabDDPM-style Gaussian Diffusion (unchanged from original)
# =============================================================================

class GaussianDiffusion(nn.Module):
    """Gaussian Diffusion for continuous/numerical features."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        if beta_schedule == "linear":
            betas = linear_beta_schedule(num_timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        # Register buffers
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float32))
        self.register_buffer("alphas_cumprod", torch.tensor(alphas_cumprod, dtype=torch.float32))
        self.register_buffer("alphas_cumprod_prev", torch.tensor(alphas_cumprod_prev, dtype=torch.float32))
        self.register_buffer("sqrt_alphas_cumprod", torch.tensor(np.sqrt(alphas_cumprod), dtype=torch.float32))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.tensor(np.sqrt(1.0 - alphas_cumprod), dtype=torch.float32))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.tensor(np.sqrt(1.0 / alphas_cumprod), dtype=torch.float32))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.tensor(np.sqrt(1.0 / alphas_cumprod - 1), dtype=torch.float32))

        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", torch.tensor(posterior_variance, dtype=torch.float32))
        self.register_buffer("posterior_log_variance_clipped",
                           torch.tensor(np.log(np.append(posterior_variance[1], posterior_variance[1:])), dtype=torch.float32))
        self.register_buffer("posterior_mean_coef1",
                           torch.tensor(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod), dtype=torch.float32))
        self.register_buffer("posterior_mean_coef2",
                           torch.tensor((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod), dtype=torch.float32))

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """Compute posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(self, model_output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """Compute p(x_{t-1} | x_t) using model output (predicted noise)."""
        pred_xstart = self.predict_xstart_from_eps(x_t, t, model_output)
        model_mean, model_var, model_log_var = self.q_posterior_mean_variance(pred_xstart, x_t, t)
        return model_mean, model_var, model_log_var, pred_xstart

    @torch.no_grad()
    def p_sample(self, model_output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Single reverse diffusion step."""
        model_mean, _, model_log_var, _ = self.p_mean_variance(model_output, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_var) * noise

    def training_loss(self, model_output: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """MSE loss between predicted and actual noise."""
        return mean_flat((model_output - noise) ** 2)


# =============================================================================
# TabDDPM-style Multinomial Diffusion (KEY DIFFERENCE)
# =============================================================================

class MultinomialDiffusionTabDDPM(nn.Module):
    """
    TabDDPM-style Multinomial Diffusion for categorical features.

    Key differences from our simple implementation:
    1. Works entirely in log-space for numerical stability
    2. Uses KL divergence loss on posteriors (not cross-entropy)
    3. Implements proper posterior computation q(x_{t-1}|x_t, x_0)
    4. Uses Gumbel-softmax for sampling
    """

    def __init__(
        self,
        num_classes: List[int],
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.total_cat_dims = sum(num_classes)

        # Compute slices for each categorical feature
        self.slices = []
        offset = 0
        for nc in num_classes:
            self.slices.append(slice(offset, offset + nc))
            offset += nc

        # Beta schedule
        if beta_schedule == "linear":
            betas = linear_beta_schedule(num_timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        # Register buffers in log-space
        self.register_buffer("log_alpha", torch.tensor(log_alpha, dtype=torch.float32))
        self.register_buffer("log_cumprod_alpha", torch.tensor(log_cumprod_alpha, dtype=torch.float32))
        self.register_buffer("log_1_min_alpha", torch.tensor(log_1_min_a(torch.tensor(log_alpha)), dtype=torch.float32))
        self.register_buffer("log_1_min_cumprod_alpha", torch.tensor(log_1_min_a(torch.tensor(log_cumprod_alpha)), dtype=torch.float32))

        # Also keep non-log versions for convenience
        self.register_buffer("alphas_cumprod", torch.tensor(np.exp(log_cumprod_alpha), dtype=torch.float32))
        self.register_buffer("alphas_cumprod_prev", torch.tensor(np.append(1.0, np.exp(log_cumprod_alpha[:-1])), dtype=torch.float32))

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_pred(self, log_x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0) in log-space.

        q(x_t | x_0) = alpha_bar_t * x_0 + (1 - alpha_bar_t) * 1/K
        In log-space: log_add_exp(log_x_0 + log_alpha_bar_t, log_1_min_alpha_bar_t - log(K))
        """
        log_cumprod_alpha_t = self._extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha_t = self._extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        # Compute log(1/K) for each categorical feature
        log_uniform = self._get_log_uniform(log_x_start)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha_t + log_uniform
        )
        return log_probs

    def q_pred_one_timestep(self, log_x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """One-step transition: q(x_t | x_{t-1})."""
        log_alpha_t = self._extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = self._extract(self.log_1_min_alpha, t, log_x_t.shape)
        log_uniform = self._get_log_uniform(log_x_t)

        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t + log_uniform
        )
        return log_probs

    def _get_log_uniform(self, log_x: torch.Tensor) -> torch.Tensor:
        """Get log(1/K) for each position, respecting categorical boundaries."""
        log_uniform = torch.zeros_like(log_x)
        for i, nc in enumerate(self.num_classes):
            log_uniform[:, self.slices[i]] = -math.log(nc)
        return log_uniform

    def q_posterior(self, log_x_start: torch.Tensor, log_x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute posterior q(x_{t-1} | x_t, x_0) in log-space.

        q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) * q(x_{t-1} | x_0)
        """
        # Handle t=0 case: posterior is just x_0
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)

        # q(x_{t-1} | x_0)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        # At t=0, return log_x_start directly
        t_broadcast = t.view(-1, *([1] * (len(log_x_start.shape) - 1))).expand_as(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)

        # Unnormalized log posterior: log q(x_{t-1}|x_0) + log q(x_t|x_{t-1})
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        # Normalize per categorical feature
        log_posterior = self._normalize_logprobs(unnormed_logprobs)
        return log_posterior

    def _normalize_logprobs(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Normalize log probabilities per categorical feature."""
        normalized = torch.zeros_like(log_probs)
        for i, nc in enumerate(self.num_classes):
            sl = self.slices[i]
            log_sum = torch.logsumexp(log_probs[:, sl], dim=1, keepdim=True)
            normalized[:, sl] = log_probs[:, sl] - log_sum
        return normalized

    def predict_start(self, model_output: torch.Tensor) -> torch.Tensor:
        """Convert model output (logits) to log probabilities for x_0."""
        log_pred = torch.zeros_like(model_output)
        for i, nc in enumerate(self.num_classes):
            sl = self.slices[i]
            log_pred[:, sl] = F.log_softmax(model_output[:, sl], dim=1)
        return log_pred

    def p_pred(self, model_output: torch.Tensor, log_x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Model prediction p(x_{t-1} | x_t).

        Model predicts x_0, then we compute the posterior.
        """
        log_x0_pred = self.predict_start(model_output)
        log_model_pred = self.q_posterior(log_x0_pred, log_x_t, t)
        return log_model_pred

    def log_sample_categorical(self, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Sample from categorical using Gumbel-softmax trick.

        This is more stable than direct multinomial sampling.
        """
        full_sample = []
        for i, nc in enumerate(self.num_classes):
            sl = self.slices[i]
            logits = log_probs[:, sl]

            # Gumbel-softmax sampling
            uniform = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sample_idx = (gumbel_noise + logits).argmax(dim=1)
            full_sample.append(sample_idx)

        # Convert back to log one-hot
        full_sample = torch.stack(full_sample, dim=1)
        return index_to_log_onehot(full_sample, self.num_classes).to(log_probs.device)

    def q_sample(self, log_x_start: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: sample x_t from q(x_t | x_0)."""
        log_probs = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_probs)
        return log_sample

    @torch.no_grad()
    def p_sample(self, model_output: torch.Tensor, log_x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Single reverse diffusion step."""
        log_model_prob = self.p_pred(model_output, log_x_t, t)
        log_sample = self.log_sample_categorical(log_model_prob)
        return log_sample

    def multinomial_kl(self, log_prob1: torch.Tensor, log_prob2: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence: KL(p1 || p2) where inputs are log probabilities."""
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def training_loss(
        self,
        model_output: torch.Tensor,
        log_x_start: torch.Tensor,
        log_x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence loss.

        L = KL(q(x_{t-1}|x_t, x_0) || p_θ(x_{t-1}|x_t))

        At t=0: use negative log-likelihood instead.
        """
        # True posterior q(x_{t-1} | x_t, x_0)
        log_true_prob = self.q_posterior(log_x_start, log_x_t, t)

        # Model prediction p(x_{t-1} | x_t)
        log_model_prob = self.p_pred(model_output, log_x_t, t)

        # KL divergence
        kl = self.multinomial_kl(log_true_prob, log_model_prob)

        # At t=0, use decoder NLL instead
        decoder_nll = -self._log_categorical(log_x_start, log_model_prob)

        # Select KL for t>0, NLL for t=0
        mask = (t == 0).float()
        loss = mask * decoder_nll + (1.0 - mask) * kl

        return loss

    def _log_categorical(self, log_x: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
        """Compute log p(x) under categorical distribution."""
        return (log_x.exp() * log_prob).sum(dim=1)


# =============================================================================
# TabDDPM-style Hybrid Diffusion
# =============================================================================

class HybridDiffusionTabDDPM(nn.Module):
    """
    TabDDPM-style Hybrid Diffusion combining Gaussian and Multinomial.

    Key differences from our simple implementation:
    1. Categorical features are processed in log-space
    2. Model receives log probabilities (not one-hot) for categoricals
    3. Uses KL divergence loss for categoricals
    4. Uses Gumbel-softmax for sampling
    """

    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: List[int],
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
    ):
        super().__init__()
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities
        self.num_timesteps = num_timesteps
        self.total_cat_dims = sum(cat_cardinalities)

        self.gaussian_diffusion = GaussianDiffusion(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
        )

        self.multinomial_diffusion = MultinomialDiffusionTabDDPM(
            num_classes=cat_cardinalities,
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
        ) if cat_cardinalities else None

    def _split_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split into numerical and categorical parts."""
        x_num = x[:, :self.num_numerical]
        x_cat = x[:, self.num_numerical:]
        return x_num, x_cat

    def _concat_features(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Concatenate numerical and categorical parts."""
        return torch.cat([x_num, x_cat], dim=-1)

    def q_sample(
        self,
        x_num: torch.Tensor,
        cat_indices: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward diffusion for hybrid data.

        Args:
            x_num: Numerical features (batch, num_numerical)
            cat_indices: Categorical indices (batch, num_cat_features)
            t: Timesteps (batch,)

        Returns:
            x_num_t: Noised numerical features
            log_x_cat_t: Noised categorical features in log-space
            noise: Noise added to numerical features
            log_x_cat_start: Original categorical features in log-space
        """
        # Numerical: standard Gaussian diffusion
        x_num_t, noise = self.gaussian_diffusion.q_sample(x_num, t)

        # Categorical: convert to log one-hot, then diffuse
        log_x_cat_start = index_to_log_onehot(cat_indices, self.cat_cardinalities).to(x_num.device)
        log_x_cat_t = self.multinomial_diffusion.q_sample(log_x_cat_start, t)

        return x_num_t, log_x_cat_t, noise, log_x_cat_start

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_num_t: torch.Tensor,
        log_x_cat_t: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single reverse diffusion step."""
        # Combine inputs for model
        x_in = self._concat_features(x_num_t, log_x_cat_t)
        model_output = model(x_in, t, y)

        # Split model output
        model_out_num = model_output[:, :self.num_numerical]
        model_out_cat = model_output[:, self.num_numerical:]

        # Reverse step for numerical
        x_num_t_minus_1 = self.gaussian_diffusion.p_sample(model_out_num, x_num_t, t)

        # Reverse step for categorical
        log_x_cat_t_minus_1 = self.multinomial_diffusion.p_sample(model_out_cat, log_x_cat_t, t)

        return x_num_t_minus_1, log_x_cat_t_minus_1

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        y: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate samples by running full reverse diffusion.

        Returns:
            x_num: Generated numerical features
            cat_indices: Generated categorical indices
        """
        # Initialize from noise
        x_num = torch.randn(batch_size, self.num_numerical, device=device)

        # Initialize categorical from uniform distribution in log-space
        uniform_logits = torch.zeros(batch_size, self.total_cat_dims, device=device)
        log_x_cat = self.multinomial_diffusion.log_sample_categorical(uniform_logits)

        # Reverse diffusion
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x_num, log_x_cat = self.p_sample(model, x_num, log_x_cat, t_batch, y)

        # Convert log_x_cat back to indices
        cat_indices = log_onehot_to_index(log_x_cat, self.cat_cardinalities)

        return x_num, cat_indices

    def training_loss(
        self,
        model: nn.Module,
        x_num: torch.Tensor,
        cat_indices: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            model: Denoiser network
            x_num: Numerical features (batch, num_numerical)
            cat_indices: Categorical indices (batch, num_cat_features)
            y: Optional conditioning labels

        Returns:
            Dictionary with total loss, numerical loss, and categorical loss
        """
        batch_size = x_num.shape[0]
        device = x_num.device

        # Sample timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

        # Forward diffusion
        x_num_t, log_x_cat_t, noise, log_x_cat_start = self.q_sample(x_num, cat_indices, t)

        # Model forward pass (receives log probabilities for categoricals)
        x_in = self._concat_features(x_num_t, log_x_cat_t)
        model_output = model(x_in, t, y)

        # Split model output
        model_out_num = model_output[:, :self.num_numerical]
        model_out_cat = model_output[:, self.num_numerical:]

        # Numerical loss (MSE on noise)
        loss_num = self.gaussian_diffusion.training_loss(model_out_num, noise)

        # Categorical loss (KL divergence)
        loss_cat = self.multinomial_diffusion.training_loss(
            model_out_cat, log_x_cat_start, log_x_cat_t, t
        )

        # Average categorical loss over number of categorical features
        loss_cat = loss_cat / len(self.cat_cardinalities) if self.cat_cardinalities else loss_cat

        # Total loss
        loss = loss_num.mean() + loss_cat.mean()

        return {
            "loss": loss,
            "loss_num": loss_num.mean(),
            "loss_cat": loss_cat.mean(),
        }
