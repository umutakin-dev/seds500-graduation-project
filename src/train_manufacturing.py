"""
Train GaussianDiffusion model on manufacturing duration data.

Uses only Çap (diameter) and Boy (length) as features - no categorical.
This matches the harder prediction problem where baseline R² ~ 0.75.

Usage:
    python src/train_manufacturing.py --epochs 500 --device cuda
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm

from diffusion import GaussianDiffusion
from models import MLPDenoiser


def load_data(data_path="data/manufacturing/prepared.pt"):
    """Load prepared manufacturing data (numeric only)."""
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)

    print(f"Train shape: {data['X_train'].shape}")
    print(f"Test shape: {data['X_test'].shape}")
    print(f"Features: {data['feature_names']}")

    return data


def train(args):
    # Load data
    data = load_data(args.data_path)
    X_train = data["X_train"]

    num_features = data["num_features"]  # 3: Çap, Boy, target

    # Create model and diffusion (numeric only)
    model = MLPDenoiser(
        d_in=num_features,
        hidden_dims=[256, 256, 256],
        dropout=0.1,
    ).to(args.device)

    diffusion = GaussianDiffusion(
        num_timesteps=1000,
        beta_schedule="linear",
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Training loop
    print(f"\nTraining GaussianDiffusion on {args.device}...")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Input dim: {num_features} (Çap, Boy, target)")

    dataset = torch.utils.data.TensorDataset(X_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    best_loss = float("inf")
    losses = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            x_0 = batch[0].to(args.device)

            loss = diffusion.training_loss(model, x_0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Save checkpoint
    checkpoint_dir = Path("checkpoints/manufacturing")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "d_in": num_features,
            "hidden_dims": [256, 256, 256],
            "dropout": 0.1,
        },
        "diffusion_config": {
            "num_timesteps": 1000,
            "beta_schedule": "linear",
        },
        "data_config": {
            "num_features": num_features,
            "feature_names": data["feature_names"],
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
    parser.add_argument("--data_path", type=str, default="data/manufacturing/prepared.pt")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train(args)
