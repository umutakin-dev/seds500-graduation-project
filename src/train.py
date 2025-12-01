"""
Training script for tabular diffusion model.

Usage:
    python src/train.py --dataset iris --epochs 1000
    python src/train.py --dataset california --epochs 2000
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

from diffusion import GaussianDiffusion
from models import MLPDenoiser, ResidualMLPDenoiser


def load_dataset(name: str, use_quantile: bool = True):
    """
    Load and preprocess a dataset.

    Args:
        name: Dataset name ('iris', 'california')
        use_quantile: If True, use quantile transformer (recommended for diffusion)

    Returns:
        X_train, X_test, scaler
    """
    if name == "iris":
        data = load_iris()
        X, y = data.data, data.target
    elif name == "california":
        data = fetch_california_housing()
        X, y = data.data, data.target.reshape(-1, 1)
        # For california, include target as feature (regression task)
        X = np.hstack([X, y])
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Split
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # Scale to approximately [-1, 1] range
    if use_quantile:
        # Quantile transform to Gaussian, then scale
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Clip to [-3, 3] (covers 99.7% of normal distribution)
        X_train = np.clip(X_train, -3, 3) / 3  # Now in [-1, 1]
        X_test = np.clip(X_test, -3, 3) / 3
    else:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Clip outliers
        X_train = np.clip(X_train, -3, 3) / 3
        X_test = np.clip(X_test, -3, 3) / 3

    return X_train, X_test, scaler


def train(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int,
    checkpoint_dir: Path = None,
):
    """
    Train the diffusion model.

    Args:
        model: Denoising network
        diffusion: Diffusion process
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epochs: Number of epochs
        checkpoint_dir: Directory to save checkpoints
    """
    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch in pbar:
            x = batch[0].to(device)

            optimizer.zero_grad()
            loss = diffusion.training_loss(model, x)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

            # Save checkpoint
            if checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_dir / f"checkpoint_{epoch + 1}.pt")

    return losses


@torch.no_grad()
def evaluate(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    X_real: np.ndarray,
    n_samples: int,
    device: str,
):
    """
    Generate samples and compute basic statistics.

    Args:
        model: Trained denoising network
        diffusion: Diffusion process
        X_real: Real data for comparison
        n_samples: Number of samples to generate
        device: Device
    """
    model.eval()

    # Generate samples
    print(f"Generating {n_samples} samples...")
    shape = (n_samples, X_real.shape[1])
    X_synthetic = diffusion.sample(model, shape, device=device)
    X_synthetic = X_synthetic.cpu().numpy()

    # Compare statistics
    print("\n=== Statistics Comparison ===")
    print(f"{'Feature':<10} {'Real Mean':>12} {'Syn Mean':>12} {'Real Std':>12} {'Syn Std':>12}")
    print("-" * 60)

    for i in range(X_real.shape[1]):
        real_mean = X_real[:, i].mean()
        syn_mean = X_synthetic[:, i].mean()
        real_std = X_real[:, i].std()
        syn_std = X_synthetic[:, i].std()
        print(f"{'Feature ' + str(i):<10} {real_mean:>12.4f} {syn_mean:>12.4f} {real_std:>12.4f} {syn_std:>12.4f}")

    # Correlation comparison
    print("\n=== Correlation Matrix Comparison ===")
    real_corr = np.corrcoef(X_real.T)
    syn_corr = np.corrcoef(X_synthetic.T)
    corr_diff = np.abs(real_corr - syn_corr).mean()
    print(f"Mean absolute correlation difference: {corr_diff:.4f}")

    return X_synthetic


def main():
    parser = argparse.ArgumentParser(description="Train tabular diffusion model")
    parser.add_argument("--dataset", type=str, default="iris", choices=["iris", "california"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "residual"])
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Using device: {args.device}")
    print(f"Dataset: {args.dataset}")

    # Load data
    X_train, X_test, scaler = load_dataset(args.dataset)
    print(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")

    # Create data loader
    train_tensor = torch.FloatTensor(X_train)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Create diffusion process
    diffusion = GaussianDiffusion(
        num_timesteps=args.timesteps,
        beta_schedule="cosine",
    ).to(args.device)

    # Create model
    d_in = X_train.shape[1]
    if args.model_type == "mlp":
        model = MLPDenoiser(
            d_in=d_in,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
        ).to(args.device)
    else:
        model = ResidualMLPDenoiser(
            d_in=d_in,
            d_model=256,
            d_hidden=512,
            n_blocks=4,
            dropout=args.dropout,
        ).to(args.device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    checkpoint_dir = Path(args.checkpoint_dir) / args.dataset
    losses = train(
        model=model,
        diffusion=diffusion,
        train_loader=train_loader,
        optimizer=optimizer,
        device=args.device,
        epochs=args.epochs,
        checkpoint_dir=checkpoint_dir,
    )

    # Save final model
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'diffusion_config': {
            'num_timesteps': args.timesteps,
            'beta_schedule': 'cosine',
        },
        'model_config': {
            'd_in': d_in,
            'hidden_dims': args.hidden_dims,
            'dropout': args.dropout,
        },
        'scaler': scaler,
    }, checkpoint_dir / "final_model.pt")
    print(f"\nModel saved to {checkpoint_dir / 'final_model.pt'}")

    # Evaluate
    X_synthetic = evaluate(
        model=model,
        diffusion=diffusion,
        X_real=X_train,
        n_samples=len(X_train),
        device=args.device,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
