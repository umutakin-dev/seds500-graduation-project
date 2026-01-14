"""
Train HybridDiffusion on Ozel Rich data - V3 (TabDDPM-style).

Key changes:
- Uses v3 data preparation (unbounded Gaussian, no /3 clipping)
- Uses clip_denoised=False during sampling
- Proper inverse transform for generated samples

This should fix the model collapse issue observed in v1/v2.
"""

import argparse
import json
import pickle
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

from diffusion import HybridDiffusion
from models import MLPDenoiser


def load_v3_data():
    """Load v3 prepared data (unbounded Gaussian)."""
    data_path = Path("data/ozel_rich_v3/prepared.pt")
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run prepare_ozel_rich_data_v3.py first."
        )

    data = torch.load(data_path, weights_only=False)

    with open(data_path.parent / "preprocessors.pkl", "rb") as f:
        preprocessors = pickle.load(f)

    return data, preprocessors


def create_training_tensor(data):
    """Create combined tensor for diffusion training (numeric + one-hot categorical)."""
    X_train_num = data["X_train_num"]  # Already scaled (unbounded Gaussian)
    cat_train = data["cat_train"]
    cat_cardinalities = data["cat_cardinalities"]

    # One-hot encode categoricals
    cat_onehot_list = []
    for i, card in enumerate(cat_cardinalities):
        cat_onehot_list.append(
            torch.eye(card)[torch.tensor(cat_train[:, i])]
        )

    cat_onehot = torch.cat(cat_onehot_list, dim=1).float()

    # Combine: [numeric] + [one-hot categorical]
    X_train = torch.cat([X_train_num, cat_onehot], dim=1)

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

        # Gradient clipping for stability
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
    """
    Validate by generating samples and checking statistics.

    For v3: Generate in unbounded Gaussian space, then inverse transform.
    """
    model.eval()

    with torch.no_grad():
        # Generate with clip_denoised=FALSE (key change for v3)
        samples = diffusion.sample(
            model,
            batch_size=n_samples,
            device=device,
            clip_denoised=False  # No hard clipping!
        )

    samples = samples.cpu().numpy()
    num_numerical = data["num_numerical"]

    # Get training stats for comparison
    train_num = data["X_train_num"].numpy()
    train_mean = train_num.mean(axis=0)
    train_std = train_num.std(axis=0)

    # Generated numeric stats
    gen_num = samples[:, :num_numerical]
    gen_mean = gen_num.mean(axis=0)
    gen_std = gen_num.std(axis=0)

    # Check for boundary collapse (v1/v2 issue)
    at_boundary = (np.abs(gen_num) > 2.5).mean(axis=0)  # >2.5 sigma is rare

    # Inverse transform to original scale
    scaler = preprocessors["scaler"]
    target_scaler = preprocessors["target_scaler"]

    # Separate features and target
    gen_features_scaled = gen_num[:, :2]  # Cap, Boy
    gen_target_scaled = gen_num[:, 2:3]  # Target

    gen_features = scaler.inverse_transform(gen_features_scaled)
    gen_target = target_scaler.inverse_transform(gen_target_scaled).flatten()

    # Original training stats
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
    parser.add_argument("--output-dir", type=str, default="checkpoints/ozel_rich_v3")
    args = parser.parse_args()

    # Device setup
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load data
    print("\nLoading v3 data (unbounded Gaussian)...")
    data, preprocessors = load_v3_data()

    X_train = create_training_tensor(data)
    num_numerical = data["num_numerical"]
    cat_cardinalities = data["cat_cardinalities"]

    print(f"Training data shape: {X_train.shape}")
    print(f"Numerical features: {num_numerical}")
    print(f"Categorical cardinalities: {cat_cardinalities}")

    # Check data stats
    train_num = X_train[:, :num_numerical].numpy()
    print(f"\nNumeric data stats (should be ~N(0,1)):")
    print(f"  Mean: {train_num.mean(axis=0)}")
    print(f"  Std:  {train_num.std(axis=0)}")
    print(f"  Min:  {train_num.min(axis=0)}")
    print(f"  Max:  {train_num.max(axis=0)}")

    # Create dataloader
    dataset = TensorDataset(X_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create model
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

    # Training
    print(f"\n{'='*60}")
    print("TRAINING V3 MODEL (TabDDPM-style, unbounded)")
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
            # Save best model
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "d_in": d_in,
                    "hidden_dims": hidden_dims,
                    "dropout": 0.1,
                },
                "diffusion_config": {
                    "num_numerical": num_numerical,
                    "cat_cardinalities": cat_cardinalities,
                    "num_timesteps": args.timesteps,
                    "beta_schedule": args.beta_schedule,
                },
                "epoch": epoch,
                "loss": best_loss,
            }, output_dir / "best_model.pt")

        # Log progress
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{args.epochs} | "
                  f"Loss: {metrics['loss']:.4f} | "
                  f"Num: {metrics['loss_num']:.4f} | "
                  f"Cat: {metrics['loss_cat']:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

            # Validate samples (on CPU to avoid CUDA issues)
            if args.device == "cuda":
                model_cpu = model.to("cpu")
                diffusion_cpu = diffusion.to("cpu")
                val_stats = validate_samples(model_cpu, diffusion_cpu, data, preprocessors, "cpu", n_samples=100)
                model.to(device)
                diffusion.to(device)
            else:
                val_stats = validate_samples(model, diffusion, data, preprocessors, device, n_samples=100)

            print(f"  Generated: mean={val_stats['gen_mean']}, std={val_stats['gen_std']}")
            print(f"  Training:  mean={val_stats['train_mean']}, std={val_stats['train_std']}")
            print(f"  At boundary (>2.5σ): {val_stats['at_boundary']}")
            print(f"  Target: gen={val_stats['gen_target_mean']:.1f}±{val_stats['gen_target_std']:.1f}, "
                  f"orig={val_stats['orig_target_mean']:.1f}±{val_stats['orig_target_std']:.1f}")

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "d_in": d_in,
            "hidden_dims": hidden_dims,
            "dropout": 0.1,
        },
        "diffusion_config": {
            "num_numerical": num_numerical,
            "cat_cardinalities": cat_cardinalities,
            "num_timesteps": args.timesteps,
            "beta_schedule": args.beta_schedule,
        },
        "epoch": args.epochs,
        "loss": metrics["loss"],
    }, output_dir / "final_model.pt")

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
