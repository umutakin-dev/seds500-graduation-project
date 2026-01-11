"""
Train HybridDiffusion model on production data (numeric + categorical features).

This uses both Gaussian diffusion for numeric features and Multinomial diffusion
for categorical features, following the TabDDPM approach.

Usage:
    python src/train_production_full.py --epochs 500 --device cuda
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from diffusion import HybridDiffusion
from models import HybridMLPDenoiser


def load_data(data_path="data/production/full.pt"):
    """Load prepared production data with categorical features."""
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)

    print(f"Train shape: {data['X_train'].shape}")
    print(f"Test shape: {data['X_test'].shape}")
    print(f"Numerical dims: {data['num_numerical']} (including target)")
    print(f"Categorical features: {len(data['cat_cardinalities'])}")
    print(f"Categorical cardinalities: {data['cat_cardinalities']}")
    print(f"Target: {data['target_col']}")

    return data


def train(args):
    # Load data
    data = load_data(args.data_path)
    X_train = data["X_train"]

    num_numerical = data["num_numerical"]
    cat_cardinalities = data["cat_cardinalities"]

    # Create model and diffusion
    model = HybridMLPDenoiser(
        num_numerical=num_numerical,
        cat_cardinalities=cat_cardinalities,
        hidden_dims=[256, 256, 256],
        dropout=0.1,
    ).to(args.device)

    diffusion = HybridDiffusion(
        num_numerical=num_numerical,
        cat_cardinalities=cat_cardinalities,
        num_timesteps=1000,
        beta_schedule="linear",
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Training loop
    print(f"\nTraining HybridDiffusion on {args.device}...")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Total input dim: {num_numerical} + {sum(cat_cardinalities)} = {num_numerical + sum(cat_cardinalities)}")

    dataset = torch.utils.data.TensorDataset(X_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    best_loss = float("inf")
    losses = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_loss_num = 0
        total_loss_cat = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            x_0 = batch[0].to(args.device)

            loss_dict = diffusion.training_loss(model, x_0)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_loss_num += loss_dict["loss_num"].item()
            total_loss_cat += loss_dict["loss_cat"].item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        avg_loss_num = total_loss_num / len(dataloader)
        avg_loss_cat = total_loss_cat / len(dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f} (num={avg_loss_num:.4f}, cat={avg_loss_cat:.4f})")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Save checkpoint
    checkpoint_dir = Path("checkpoints/production_full")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "num_numerical": num_numerical,
            "cat_cardinalities": cat_cardinalities,
            "hidden_dims": [256, 256, 256],
            "dropout": 0.1,
        },
        "diffusion_config": {
            "num_timesteps": 1000,
            "beta_schedule": "linear",
        },
        "data_config": {
            "numerical_cols": data["numerical_cols"],
            "categorical_cols": data["categorical_cols"],
            "target_col": data["target_col"],
        },
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
        "final_loss": best_loss,
        "losses": losses,
    }
    torch.save(checkpoint, checkpoint_dir / "final_model.pt")
    print(f"\nModel saved to {checkpoint_dir / 'final_model.pt'}")
    print(f"Final loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/production/full.pt")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train(args)
