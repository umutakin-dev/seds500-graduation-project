"""
Train HybridDiffusion on Ozel Rich data - V4 (Soft clipping approach).

Key insight from V3 failure:
- Without clipping: values explode to infinity
- With hard clip [-1,1]: boundary collapse
- Solution: Soft clip to [-3, 3] (covers 99.7% of Gaussian)

This approach:
1. Keep unbounded Gaussian preprocessing (v3 style)
2. Use soft clipping in sampling: clip to [-3, 3] instead of [-1, 1]
3. Scale model output to match wider range
"""

import argparse
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

from diffusion import HybridDiffusion, GaussianDiffusion
from models import MLPDenoiser


# Monkey-patch the p_sample method to use wider clipping bounds
def patched_p_sample(self, model, x_t, t, y=None, clip_denoised=True, clip_range=3.0):
    """Single reverse diffusion step with configurable clip range."""
    model_output = model(x_t, t, y)

    noise_pred = model_output[:, :self.num_numerical]
    cat_logits = []
    offset = self.num_numerical
    for card in self.cat_cardinalities:
        cat_logits.append(model_output[:, offset:offset + card])
        offset += card

    x_num_t, x_cats_t = self._split_features(x_t)

    # Reverse step for numerical
    sqrt_alphas_cumprod_t = self.gaussian_diffusion._extract(
        self.gaussian_diffusion.sqrt_alphas_cumprod, t, x_num_t.shape)
    sqrt_one_minus_alphas_cumprod_t = self.gaussian_diffusion._extract(
        self.gaussian_diffusion.sqrt_one_minus_alphas_cumprod, t, x_num_t.shape)

    x_num_0_pred = (x_num_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t

    if clip_denoised:
        # WIDER CLIPPING: [-3, 3] instead of [-1, 1]
        x_num_0_pred = torch.clamp(x_num_0_pred, -clip_range, clip_range)

    model_mean, _, model_log_var = self.gaussian_diffusion.q_posterior_mean_variance(
        x_0=x_num_0_pred, x_t=x_num_t, t=t)

    noise = torch.randn_like(x_num_t)
    nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_num_t.shape) - 1)))
    model_log_var_clamped = torch.clamp(model_log_var, min=-20, max=2)
    x_num_t_minus_1 = model_mean + nonzero_mask * torch.exp(0.5 * model_log_var_clamped) * noise
    x_num_t_minus_1 = torch.where(torch.isnan(x_num_t_minus_1), x_num_t, x_num_t_minus_1)

    # Categorical sampling (same as original)
    x_cats_t_minus_1 = []
    for i, (logits, diff) in enumerate(zip(cat_logits, self.multinomial_diffusions)):
        logits_clamped = torch.clamp(logits, min=-20, max=20)
        logits_clamped = torch.where(torch.isnan(logits_clamped), torch.zeros_like(logits_clamped), logits_clamped)

        probs = F.softmax(logits_clamped, dim=-1)
        bad_mask = torch.isnan(probs) | torch.isinf(probs) | (probs < 0)
        if bad_mask.any():
            uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = torch.where(bad_mask, uniform, probs)

        probs = probs + 1e-6
        probs = probs / probs.sum(dim=-1, keepdim=True)

        if t[0] == 0:
            try:
                samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
            except RuntimeError:
                samples = probs.argmax(dim=-1)
            x_cat = F.one_hot(samples, num_classes=diff.num_classes).float()
        else:
            alpha_t = diff._extract(diff.alphas_cumprod, t, probs.shape)
            alpha_t_prev = diff._extract(F.pad(diff.alphas_cumprod[:-1], (1, 0), value=1.0), t, probs.shape)
            ratio = alpha_t_prev / alpha_t.clamp(min=1e-8)
            x_cat = ratio * x_cats_t[i] + (1 - ratio) * probs
            x_cat = torch.where(torch.isnan(x_cat), probs, x_cat)
            x_cat = x_cat / x_cat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        x_cats_t_minus_1.append(x_cat)

    return self._concat_features(x_num_t_minus_1, x_cats_t_minus_1)


def patched_sample(self, model, batch_size, y=None, device="cpu", clip_denoised=True, clip_range=3.0):
    """Generate samples with configurable clip range."""
    x_num = torch.randn(batch_size, self.num_numerical, device=device)
    x_cats = [torch.ones(batch_size, card, device=device) / card for card in self.cat_cardinalities]
    x = self._concat_features(x_num, x_cats)

    for t in reversed(range(self.num_timesteps)):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        x = patched_p_sample(self, model, x, t_batch, y, clip_denoised, clip_range)
    return x


def load_v3_data():
    """Load v3 prepared data (unbounded Gaussian)."""
    data_path = Path("data/ozel_rich_v3/prepared.pt")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found.")

    data = torch.load(data_path, weights_only=False)
    with open(data_path.parent / "preprocessors.pkl", "rb") as f:
        preprocessors = pickle.load(f)
    return data, preprocessors


def create_training_tensor(data):
    """Create combined tensor for diffusion training."""
    X_train_num = data["X_train_num"]
    cat_train = data["cat_train"]
    cat_cardinalities = data["cat_cardinalities"]

    cat_onehot_list = []
    for i, card in enumerate(cat_cardinalities):
        cat_onehot_list.append(torch.eye(card)[torch.tensor(cat_train[:, i])])
    cat_onehot = torch.cat(cat_onehot_list, dim=1).float()

    # IMPORTANT: Scale training data to [-3, 3] range to match our clip bounds
    # This is the key insight: training data should match the sampling bounds
    X_train_num_scaled = torch.clamp(X_train_num, -3, 3)

    X_train = torch.cat([X_train_num_scaled, cat_onehot], dim=1)
    return X_train


def train_epoch(model, diffusion, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_loss_num = 0
    total_loss_cat = 0
    n_batches = 0

    for batch in dataloader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        losses = diffusion.training_loss(model, x)
        loss = losses["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_loss_num += losses["loss_num"].item()
        total_loss_cat += losses["loss_cat"].item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "loss_num": total_loss_num / n_batches,
        "loss_cat": total_loss_cat / n_batches,
    }


def validate_samples(model, diffusion, data, preprocessors, device, n_samples=100):
    """Validate by generating samples."""
    model.eval()

    with torch.no_grad():
        samples = patched_sample(diffusion, model, n_samples, device=device,
                                  clip_denoised=True, clip_range=3.0)

    samples = samples.cpu().numpy()
    num_numerical = data["num_numerical"]

    train_num = data["X_train_num"].numpy()
    train_num_clipped = np.clip(train_num, -3, 3)
    train_mean = train_num_clipped.mean(axis=0)
    train_std = train_num_clipped.std(axis=0)

    gen_num = samples[:, :num_numerical]
    gen_mean = gen_num.mean(axis=0)
    gen_std = gen_num.std(axis=0)

    # Check for boundary collapse at ±3
    at_boundary = (np.abs(gen_num) > 2.9).mean(axis=0)

    # Inverse transform
    scaler = preprocessors["scaler"]
    target_scaler = preprocessors["target_scaler"]

    gen_features_scaled = gen_num[:, :2]
    gen_target_scaled = gen_num[:, 2:3]

    gen_features = scaler.inverse_transform(gen_features_scaled)
    gen_target = target_scaler.inverse_transform(gen_target_scaled).flatten()

    y_train = data["y_train"]

    return {
        "train_mean": train_mean,
        "train_std": train_std,
        "gen_mean": gen_mean,
        "gen_std": gen_std,
        "at_boundary": at_boundary,
        "gen_target_mean": gen_target.mean(),
        "gen_target_std": gen_target.std(),
        "orig_target_mean": y_train.mean(),
        "orig_target_std": y_train.std(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", type=str, default="256,256,256")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", type=str, default="cosine")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="checkpoints/ozel_rich_v4")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("\nLoading v3 data...")
    data, preprocessors = load_v3_data()

    X_train = create_training_tensor(data)
    num_numerical = data["num_numerical"]
    cat_cardinalities = data["cat_cardinalities"]

    print(f"Training data shape: {X_train.shape}")

    train_num = X_train[:, :num_numerical].numpy()
    print(f"\nClipped numeric stats (should be in [-3, 3]):")
    print(f"  Mean: {train_num.mean(axis=0)}")
    print(f"  Std:  {train_num.std(axis=0)}")
    print(f"  Min:  {train_num.min(axis=0)}")
    print(f"  Max:  {train_num.max(axis=0)}")

    dataset = TensorDataset(X_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    d_in = X_train.shape[1]

    model = MLPDenoiser(
        d_in=d_in,
        hidden_dims=hidden_dims,
        dropout=0.1,
    ).to(device)

    diffusion = HybridDiffusion(
        num_numerical=num_numerical,
        cat_cardinalities=cat_cardinalities,
        num_timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    print(f"\n{'='*60}")
    print("TRAINING V4 MODEL (Soft clip to [-3, 3])")
    print(f"{'='*60}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    history = []

    for epoch in range(args.epochs):
        metrics = train_epoch(model, diffusion, dataloader, optimizer, device)
        scheduler.step()
        history.append(metrics)

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {"d_in": d_in, "hidden_dims": hidden_dims, "dropout": 0.1},
                "diffusion_config": {
                    "num_numerical": num_numerical,
                    "cat_cardinalities": cat_cardinalities,
                    "num_timesteps": args.timesteps,
                    "beta_schedule": args.beta_schedule,
                },
                "epoch": epoch,
                "loss": best_loss,
                "clip_range": 3.0,  # Store clip range in checkpoint
            }, output_dir / "best_model.pt")

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{args.epochs} | "
                  f"Loss: {metrics['loss']:.4f} | "
                  f"Num: {metrics['loss_num']:.4f} | "
                  f"Cat: {metrics['loss_cat']:.4f}")

            if args.device == "cuda":
                model_cpu = model.to("cpu")
                diffusion_cpu = diffusion.to("cpu")
                val_stats = validate_samples(model_cpu, diffusion_cpu, data, preprocessors, "cpu")
                model.to(device)
                diffusion.to(device)
            else:
                val_stats = validate_samples(model, diffusion, data, preprocessors, device)

            print(f"  Gen: mean={val_stats['gen_mean']}, std={val_stats['gen_std']}")
            print(f"  At boundary (>2.9): {val_stats['at_boundary']}")
            print(f"  Target: gen={val_stats['gen_target_mean']:.1f}±{val_stats['gen_target_std']:.1f}, "
                  f"orig={val_stats['orig_target_mean']:.1f}±{val_stats['orig_target_std']:.1f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {"d_in": d_in, "hidden_dims": hidden_dims, "dropout": 0.1},
        "diffusion_config": {
            "num_numerical": num_numerical,
            "cat_cardinalities": cat_cardinalities,
            "num_timesteps": args.timesteps,
            "beta_schedule": args.beta_schedule,
        },
        "epoch": args.epochs,
        "loss": metrics["loss"],
        "clip_range": 3.0,
    }, output_dir / "final_model.pt")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
