"""
Train HybridDiffusion model on ozel rich dataset (from manufacturing data).

Uses Cap, Boy (numeric) and IslemTipi, AnmaOlcusu, SartnameSimple, UY (categorical).
This is Experiment 013.

Usage:
    python src/train_ozel_rich.py --epochs 500 --device cuda
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm

from diffusion import HybridDiffusion
from models import MLPDenoiser


def load_data(data_path="data/ozel_rich/prepared.pt"):
    """Load prepared ozel rich data."""
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)

    print(f"Train shape: {data['X_train'].shape}")
    print(f"Test shape: {data['X_test'].shape}")
    print(f"Features: {data['feature_names']}")
    print(f"Num numerical: {data['num_numerical']}")
    print(f"Cat cardinalities: {data['cat_cardinalities']}")

    return data


def train(args):
    # Load data
    data = load_data(args.data_path)
    X_train = data["X_train"]

    num_numerical = data["num_numerical"]  # 3: Cap, Boy, target
    cat_cardinalities = data["cat_cardinalities"]  # [3, 16, 5, 2]
    total_dims = num_numerical + sum(cat_cardinalities)  # 3 + 26 = 29

    # Create model and diffusion
    model = MLPDenoiser(
        d_in=total_dims,
        hidden_dims=[512, 512, 512],  # Larger network for more features
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
    print(f"Input dim: {total_dims} (3 numeric + {sum(cat_cardinalities)} one-hot)")

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
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "num": f"{loss_dict['loss_num'].item():.4f}",
                "cat": f"{loss_dict['loss_cat'].item():.4f}"
            })

        avg_loss = total_loss / len(dataloader)
        avg_loss_num = total_loss_num / len(dataloader)
        avg_loss_cat = total_loss_cat / len(dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f} (num={avg_loss_num:.4f}, cat={avg_loss_cat:.4f})")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Save checkpoint
    checkpoint_dir = Path("checkpoints/ozel_rich")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "d_in": total_dims,
            "hidden_dims": [512, 512, 512],
            "dropout": 0.1,
        },
        "diffusion_config": {
            "num_numerical": num_numerical,
            "cat_cardinalities": cat_cardinalities,
            "num_timesteps": 1000,
            "beta_schedule": "linear",
        },
        "data_config": {
            "num_numerical": num_numerical,
            "cat_cardinalities": cat_cardinalities,
            "cat_columns": data["cat_columns"],
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
    parser.add_argument("--data_path", type=str, default="data/ozel_rich/prepared.pt")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train(args)
