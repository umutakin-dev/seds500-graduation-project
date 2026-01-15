"""
Train TabDDPM-style HybridDiffusion on Ozel Rich data.

Experiment 018: Testing TabDDPM-style multinomial diffusion with:
- Log-space operations
- KL divergence loss
- Gumbel-softmax sampling
- Proper posterior computation
"""

import argparse
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from diffusion_tabddpm import HybridDiffusionTabDDPM
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


def create_training_data(data):
    """Create tensors for training."""
    X_train_num = data["X_train_num"]  # Already in [-1, 1]
    cat_train = torch.tensor(data["cat_train"], dtype=torch.long)
    num_numerical = data["num_numerical"]
    cat_cardinalities = data["cat_cardinalities"]

    return X_train_num, cat_train, num_numerical, cat_cardinalities


def train_epoch(model, diffusion, X_num, cat_indices, optimizer, device, batch_size=128):
    """Train for one epoch."""
    model.train()
    n_samples = X_num.shape[0]
    indices = torch.randperm(n_samples)

    total_loss = 0
    total_loss_num = 0
    total_loss_cat = 0
    n_batches = 0

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]

        x_num = X_num[batch_idx].to(device)
        cat_idx = cat_indices[batch_idx].to(device)

        optimizer.zero_grad()
        losses = diffusion.training_loss(model, x_num, cat_idx)
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
    """Generate samples and compute statistics."""
    model.eval()

    with torch.no_grad():
        x_num, cat_indices = diffusion.sample(model, batch_size=n_samples, device=device)

    x_num = x_num.cpu().numpy()
    cat_indices = cat_indices.cpu().numpy()

    # Training data stats
    train_num = data["X_train_num"].numpy()
    train_mean = train_num.mean(axis=0)
    train_std = train_num.std(axis=0)

    gen_mean = x_num.mean(axis=0)
    gen_std = x_num.std(axis=0)

    # Check for boundary collapse
    at_boundary = (np.abs(x_num) > 0.95).mean(axis=0)

    # Inverse transform target
    scaler = preprocessors["scaler"]
    target_scaler = preprocessors["target_scaler"]

    gen_features_scaled = x_num[:, :2]
    gen_target_scaled = x_num[:, 2:3]

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
    parser.add_argument("--output-dir", type=str, default="checkpoints/experiment_018")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("\nLoading v5 data...")
    data, preprocessors = load_v5_data()

    X_train_num, cat_train, num_numerical, cat_cardinalities = create_training_data(data)

    print(f"Training data: {X_train_num.shape[0]} samples")
    print(f"Numerical features: {num_numerical}")
    print(f"Categorical features: {len(cat_cardinalities)} with cardinalities {cat_cardinalities}")

    # Total input dimension: numerical + log_cat (same as one-hot)
    d_in = num_numerical + sum(cat_cardinalities)
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]

    model = MLPDenoiser(
        d_in=d_in,
        hidden_dims=hidden_dims,
        dropout=0.1,
    ).to(device)

    diffusion = HybridDiffusionTabDDPM(
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
    print("TRAINING EXPERIMENT 018: TabDDPM-style Diffusion")
    print(f"{'='*60}")
    print("Features: Log-space ops, KL loss, Gumbel-softmax sampling")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    history = []

    for epoch in range(args.epochs):
        metrics = train_epoch(
            model, diffusion, X_train_num, cat_train,
            optimizer, device, args.batch_size
        )
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

            print(f"  Gen: mean={val_stats['gen_mean'].round(3)}, std={val_stats['gen_std'].round(3)}")
            print(f"  Train: mean={val_stats['train_mean'].round(3)}, std={val_stats['train_std'].round(3)}")
            print(f"  At boundary (>0.95): {val_stats['at_boundary'].round(3)}")
            print(f"  Target: gen={val_stats['gen_target_mean']:.1f}+/-{val_stats['gen_target_std']:.1f}, "
                  f"orig={val_stats['orig_target_mean']:.1f}+/-{val_stats['orig_target_std']:.1f}")

    # Save final model
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
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
