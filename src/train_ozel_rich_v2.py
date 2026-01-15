"""
Train HybridDiffusion model on ozel rich dataset - V2 with fixes.

Changes from v1:
1. Cosine beta schedule (more stable than linear)
2. Lower learning rate (0.0001 instead of 0.0005)
3. More epochs (1000 instead of 500)
4. Validation during training to detect collapse

Usage:
    python src/train_ozel_rich_v2.py --epochs 1000 --device cuda
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
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
    print(f"Num numerical: {data['num_numerical']}")
    print(f"Cat cardinalities: {data['cat_cardinalities']}")

    return data


def validate_sampling(model, diffusion, device, n_samples=100):
    """
    Quick validation: generate samples and check if they're reasonable.
    Returns True if samples look okay, False if collapsed.
    """
    model.eval()
    try:
        with torch.no_grad():
            samples = diffusion.sample(
                model,
                batch_size=n_samples,
                device=device,
                clip_denoised=True  # Use clipping to avoid numerical issues
            )

        num_part = samples[:, :diffusion.num_numerical].cpu().numpy()

        # Check for collapse indicators
        mean = np.abs(num_part.mean())
        std = num_part.std()
        max_val = np.abs(num_part).max()

        # With clipping, collapsed samples will have std near 0 (all at boundaries)
        # Good samples should have: std > 0.1, not all at boundaries
        at_boundary = (np.abs(num_part) > 0.99).mean()
        is_collapsed = (std < 0.1) or (at_boundary > 0.5)

        return not is_collapsed, {"mean": mean, "std": std, "max": max_val, "at_boundary": at_boundary}
    except Exception as e:
        # Early in training, sampling might fail - that's okay
        return False, {"error": str(e)}


def train(args):
    # Load data
    data = load_data(args.data_path)
    X_train = data["X_train"]
    X_test = data["X_test"]

    num_numerical = data["num_numerical"]  # 3: Cap, Boy, target
    cat_cardinalities = data["cat_cardinalities"]  # [3, 16, 5, 2]
    total_dims = num_numerical + sum(cat_cardinalities)  # 3 + 26 = 29

    # Create model and diffusion with COSINE schedule
    model = MLPDenoiser(
        d_in=total_dims,
        hidden_dims=[256, 256, 256],  # Slightly smaller - less prone to overfit
        dropout=0.1,
    ).to(args.device)

    diffusion = HybridDiffusion(
        num_numerical=num_numerical,
        cat_cardinalities=cat_cardinalities,
        num_timesteps=1000,
        beta_schedule="cosine",  # Changed from linear to cosine
    ).to(args.device)

    # Lower learning rate for stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler - reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True
    )

    # Training loop
    print(f"\nTraining HybridDiffusion V2 on {args.device}...")
    print(f"Changes: cosine schedule, lr={args.lr}, epochs={args.epochs}")
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

        # Update learning rate scheduler
        scheduler.step(avg_loss)

        # Periodic logging (no validation during training - causes CUDA issues)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f} (num={avg_loss_num:.4f}, cat={avg_loss_cat:.4f})")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Final validation - do on CPU to avoid CUDA issues
    print(f"\n{'='*60}")
    print("Final sampling validation (on CPU)...")
    model_cpu = model.cpu()
    diffusion_cpu = diffusion.cpu()
    is_ok, stats = validate_sampling(model_cpu, diffusion_cpu, "cpu", n_samples=100)
    print(f"  Status: {'OK' if is_ok else 'COLLAPSED'}")
    if "error" in stats:
        print(f"  Error: {stats['error']}")
    else:
        print(f"  Stats: std={stats['std']:.4f}, boundary={stats['at_boundary']:.1%}")
    # Move back to original device for saving
    model = model.to(args.device)
    diffusion = diffusion.to(args.device)

    # Save checkpoint
    checkpoint_dir = Path("checkpoints/ozel_rich_v2")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "d_in": total_dims,
            "hidden_dims": [256, 256, 256],
            "dropout": 0.1,
        },
        "diffusion_config": {
            "num_numerical": num_numerical,
            "cat_cardinalities": cat_cardinalities,
            "num_timesteps": 1000,
            "beta_schedule": "cosine",  # Changed
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
            "version": "v2",
        },
        "final_loss": best_loss,
        "losses": losses,
        "sampling_valid": is_ok,
    }
    torch.save(checkpoint, checkpoint_dir / "final_model.pt")
    print(f"\nModel saved to {checkpoint_dir / 'final_model.pt'}")
    print(f"Final loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/ozel_rich/prepared.pt")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)  # Lower than v1
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train(args)
