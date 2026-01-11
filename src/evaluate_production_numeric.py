"""
Evaluate Gaussian diffusion model on production data (numeric features only).

Compares augmentation methods for regression:
- Original data only
- Diffusion augmentation
- Noise augmentation (baseline)
- SMOGN (if available)

Usage:
    python src/evaluate_production_numeric.py --device cuda --n_synthetic 1000
"""

import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from diffusion import GaussianDiffusion
from models import MLPDenoiser


def load_data(data_path="data/production/numeric.pt"):
    """Load prepared production data."""
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    return data


def load_model(checkpoint_path, device):
    """Load trained diffusion model."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint["model_config"]
    diffusion_config = checkpoint["diffusion_config"]

    model = MLPDenoiser(
        d_in=model_config["d_in"],
        hidden_dims=model_config["hidden_dims"],
        dropout=model_config.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    diffusion = GaussianDiffusion(
        num_timesteps=diffusion_config["num_timesteps"],
        beta_schedule=diffusion_config["beta_schedule"],
    ).to(device)

    return model, diffusion, checkpoint


def generate_samples(model, diffusion, n_samples, d_in, device):
    """Generate synthetic samples using diffusion."""
    print(f"Generating {n_samples} synthetic samples...")
    with torch.no_grad():
        samples = diffusion.sample(
            model,
            shape=(n_samples, d_in),
            device=device,
            clip_denoised=True,
        )
    return samples.cpu().numpy()


def add_noise_augmentation(X, n_samples, noise_scale=0.1):
    """Simple noise augmentation baseline."""
    indices = np.random.choice(len(X), n_samples, replace=True)
    X_noisy = X[indices] + np.random.normal(0, noise_scale, (n_samples, X.shape[1]))
    return X_noisy.astype(np.float32)


def evaluate_regressor(name, model, X_train, y_train, X_test, y_test):
    """Train and evaluate a regressor."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {"rmse": rmse, "mae": mae, "r2": r2}


def main(args):
    # Load data
    data = load_data(args.data_path)

    X_train = data["X_train"].numpy()
    X_test = data["X_test"].numpy()

    # Split features and target (target is last column)
    X_train_features = X_train[:, :-1]
    y_train = X_train[:, -1]
    X_test_features = X_test[:, :-1]
    y_test = X_test[:, -1]

    print(f"\nData shapes:")
    print(f"  Train: {X_train_features.shape} features, {y_train.shape} target")
    print(f"  Test: {X_test_features.shape} features, {y_test.shape} target")

    # Load model
    checkpoint_path = Path("checkpoints/production_numeric/final_model.pt")
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please run train_production_numeric.py first.")
        return

    model, diffusion, checkpoint = load_model(checkpoint_path, args.device)
    d_in = checkpoint["model_config"]["d_in"]

    # Generate synthetic samples
    n_synthetic = args.n_synthetic
    synthetic = generate_samples(model, diffusion, n_synthetic, d_in, args.device)
    synthetic_features = synthetic[:, :-1]
    synthetic_target = synthetic[:, -1]

    # Noise augmentation baseline
    noise_aug = add_noise_augmentation(X_train, n_synthetic, noise_scale=0.1)
    noise_features = noise_aug[:, :-1]
    noise_target = noise_aug[:, -1]

    # Prepare datasets
    datasets = {
        "Original": (X_train_features, y_train),
        "+ Diffusion": (
            np.vstack([X_train_features, synthetic_features]),
            np.concatenate([y_train, synthetic_target])
        ),
        "+ Noise": (
            np.vstack([X_train_features, noise_features]),
            np.concatenate([y_train, noise_target])
        ),
    }

    print(f"\nDataset sizes:")
    for name, (X, y) in datasets.items():
        print(f"  {name}: {len(X)} samples")

    # Evaluate regressors
    print("\n" + "=" * 70)
    print("ML EFFICIENCY EVALUATION (Regression)")
    print("=" * 70)

    regressors = [
        ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("Ridge", Ridge(alpha=1.0)),
    ]

    results = {}

    for reg_name, reg_class in regressors:
        print(f"\n{reg_name}:")
        print("-" * 50)

        results[reg_name] = {}

        for data_name, (X, y) in datasets.items():
            # Create fresh regressor instance
            if reg_name == "Random Forest":
                reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            elif reg_name == "Gradient Boosting":
                reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
            else:
                reg = Ridge(alpha=1.0)

            metrics = evaluate_regressor(reg_name, reg, X, y, X_test_features, y_test)
            results[reg_name][data_name] = metrics

            if data_name == "Original":
                print(f"  {data_name:12}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
                baseline_rmse = metrics['rmse']
            else:
                delta = metrics['rmse'] - baseline_rmse
                print(f"  {data_name:12}: RMSE={metrics['rmse']:.4f} ({delta:+.4f}), MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for reg_name in results:
        orig_rmse = results[reg_name]["Original"]["rmse"]
        diff_rmse = results[reg_name]["+ Diffusion"]["rmse"]
        noise_rmse = results[reg_name]["+ Noise"]["rmse"]

        diff_better = diff_rmse < noise_rmse
        status = "BETTER" if diff_better else "WORSE"

        print(f"\n{reg_name}: Diffusion vs Noise = {status}")
        print(f"  Original:  RMSE={orig_rmse:.4f}")
        print(f"  Diffusion: RMSE={diff_rmse:.4f} ({diff_rmse - orig_rmse:+.4f})")
        print(f"  Noise:     RMSE={noise_rmse:.4f} ({noise_rmse - orig_rmse:+.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/production/numeric.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_synthetic", type=int, default=1000,
                        help="Number of synthetic samples to generate")
    args = parser.parse_args()

    main(args)
