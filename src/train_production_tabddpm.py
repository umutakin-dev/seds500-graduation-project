"""
Train TabDDPM-style HybridDiffusion on Production data.

Experiment 019: Apply TabDDPM-style multinomial diffusion to Production dataset
with the same improvements as Experiment 018:
- Log-space operations
- KL divergence loss
- Gumbel-softmax sampling
- Proper posterior computation

This allows fair comparison between Production and Ozel Rich using the same methodology.
"""

import argparse
import json
import pickle
import torch
import numpy as np
from pathlib import Path

from diffusion_tabddpm import HybridDiffusionTabDDPM
from models import MLPDenoiser


def load_production_data():
    """Load Production full data and convert to v5-compatible format."""
    data_path = Path("data/production/full_minmax.pt")
    preprocessor_path = Path("data/production/preprocessors_full.pkl")

    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Run prepare_production_data.py --mode full first.")

    data = torch.load(data_path, weights_only=False)
    with open(preprocessor_path, "rb") as f:
        preprocessors = pickle.load(f)

    # Data is in format: [numerical (including target), categorical_onehot]
    num_numerical = data["num_numerical"]  # 7 (includes target as last numerical column)
    cat_cardinalities = data["cat_cardinalities"]

    X_train = data["X_train"]  # Shape: [N, 117]
    X_test = data["X_test"]

    # Split into numerical and categorical parts
    X_train_num = X_train[:, :num_numerical].float()  # First 7 columns
    X_test_num = X_test[:, :num_numerical].float()

    # Convert one-hot categorical back to indices
    cat_onehot_train = X_train[:, num_numerical:].numpy()
    cat_onehot_test = X_test[:, num_numerical:].numpy()

    cat_train = onehot_to_indices(cat_onehot_train, cat_cardinalities)
    cat_test = onehot_to_indices(cat_onehot_test, cat_cardinalities)

    return {
        "X_train_num": X_train_num,
        "X_test_num": X_test_num,
        "cat_train": cat_train,
        "cat_test": cat_test,
        "y_train": data["y_train"],
        "y_test": data["y_test"],
        "num_numerical": num_numerical,
        "cat_cardinalities": cat_cardinalities,
    }, preprocessors


def onehot_to_indices(onehot, cardinalities):
    """Convert one-hot encoded categorical back to indices."""
    n_samples = onehot.shape[0]
    n_cats = len(cardinalities)
    indices = np.zeros((n_samples, n_cats), dtype=np.int64)

    offset = 0
    for i, card in enumerate(cardinalities):
        cat_onehot = onehot[:, offset:offset + card]
        indices[:, i] = cat_onehot.argmax(axis=1)
        offset += card

    return indices


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

    # Inverse transform target (last numerical column)
    target_scaler = preprocessors["target_scaler"]
    gen_target_scaled = x_num[:, -1:]
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
    parser.add_argument("--hidden-dims", type=str, default="512,512,512,512")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", type=str, default="cosine")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="checkpoints/experiment_019")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("\nLoading Production data...")
    data, preprocessors = load_production_data()

    X_train_num = data["X_train_num"]
    cat_train = torch.tensor(data["cat_train"], dtype=torch.long)
    num_numerical = data["num_numerical"]
    cat_cardinalities = data["cat_cardinalities"]

    print(f"Training data: {X_train_num.shape[0]} samples")
    print(f"Numerical features: {num_numerical}")
    print(f"Categorical features: {len(cat_cardinalities)} with cardinalities {cat_cardinalities}")

    # Total input dimension: numerical + sum of cardinalities (for log probabilities)
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
    print("TRAINING EXPERIMENT 019: TabDDPM on Production Data")
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

        # Print progress every epoch (so user knows it's not stuck)
        print(f"\rEpoch {epoch+1:4d}/{args.epochs} | Loss: {metrics['loss']:.4f} | Num: {metrics['loss_num']:.4f} | Cat: {metrics['loss_cat']:.4f}", end="", flush=True)

        # Print detailed validation every 50 epochs
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print()  # New line after the progress

            if args.device == "cuda":
                model_cpu = model.to("cpu")
                diffusion_cpu = diffusion.to("cpu")
                val_stats = validate_samples(model_cpu, diffusion_cpu, data, preprocessors, "cpu")
                model.to(device)
                diffusion.to(device)
            else:
                val_stats = validate_samples(model, diffusion, data, preprocessors, device)

            print(f"  Gen: mean={val_stats['gen_mean'][:3].round(3)}, std={val_stats['gen_std'][:3].round(3)}")
            print(f"  Train: mean={val_stats['train_mean'][:3].round(3)}, std={val_stats['train_std'][:3].round(3)}")
            print(f"  At boundary (>0.95): {val_stats['at_boundary'][:3].round(3)}")
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
