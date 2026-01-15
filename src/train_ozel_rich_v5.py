"""
Train HybridDiffusion on Ozel Rich data - V5 (MinMax scaling).

Key insight:
- Use MinMax scaling to [-1, 1] (not QuantileTransformer)
- Diffusion clip range [-1, 1] matches data range exactly
- Inverse transform is simple linear (no distribution mismatch)

This should fix both:
- Boundary collapse (data fills the range properly)
- Sampling divergence (bounded by design)
"""

import argparse
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from diffusion import HybridDiffusion
from models import MLPDenoiser


def load_v5_data():
    """Load v5 prepared data (MinMax scaled)."""
    data_path = Path("data/ozel_rich_v5/prepared.pt")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Run prepare_ozel_rich_data_v5.py first.")

    data = torch.load(data_path, weights_only=False)
    with open(data_path.parent / "preprocessors.pkl", "rb") as f:
        preprocessors = pickle.load(f)
    return data, preprocessors


def create_training_tensor(data):
    """Create combined tensor for diffusion training."""
    X_train_num = data["X_train_num"]  # Already in [-1, 1]
    cat_train = data["cat_train"]
    cat_cardinalities = data["cat_cardinalities"]

    cat_onehot_list = []
    for i, card in enumerate(cat_cardinalities):
        cat_onehot_list.append(torch.eye(card)[torch.tensor(cat_train[:, i])])
    cat_onehot = torch.cat(cat_onehot_list, dim=1).float()

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
        # Use standard clip_denoised=True since data is in [-1, 1]
        samples = diffusion.sample(model, batch_size=n_samples, device=device, clip_denoised=True)

    samples = samples.cpu().numpy()
    num_numerical = data["num_numerical"]

    train_num = data["X_train_num"].numpy()
    train_mean = train_num.mean(axis=0)
    train_std = train_num.std(axis=0)

    gen_num = samples[:, :num_numerical]
    gen_mean = gen_num.mean(axis=0)
    gen_std = gen_num.std(axis=0)

    # Check for boundary collapse
    at_boundary = (np.abs(gen_num) > 0.95).mean(axis=0)

    # Inverse transform using MinMax scaler
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
    parser.add_argument("--output-dir", type=str, default="checkpoints/ozel_rich_v5")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("\nLoading v5 data (MinMax scaled to [-1, 1])...")
    data, preprocessors = load_v5_data()

    X_train = create_training_tensor(data)
    num_numerical = data["num_numerical"]
    cat_cardinalities = data["cat_cardinalities"]

    print(f"Training data shape: {X_train.shape}")

    train_num = X_train[:, :num_numerical].numpy()
    print(f"\nNumeric data stats (should be in [-1, 1]):")
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
    print("TRAINING V5 MODEL (MinMax scaling)")
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
            print(f"  Train: mean={val_stats['train_mean']}, std={val_stats['train_std']}")
            print(f"  At boundary (>0.95): {val_stats['at_boundary']}")
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
    }, output_dir / "final_model.pt")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
