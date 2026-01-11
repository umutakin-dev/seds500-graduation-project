"""
Evaluate GaussianDiffusion model on manufacturing duration data.

Compares augmentation methods for regression:
- Original data only (baseline)
- Gaussian Diffusion augmentation
- Noise augmentation (random baseline)
- SMOGN augmentation (traditional method)

Uses only Çap and Boy as features (no categorical).
This is Experiment 011: Compare diffusion vs traditional augmentation.

Usage:
    python src/evaluate_manufacturing.py --device cuda --n_synthetic 1000
"""

import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import smogn
    SMOGN_AVAILABLE = True
except ImportError:
    SMOGN_AVAILABLE = False
    print("Warning: smogn not installed. Run: pip install smogn")

from diffusion import GaussianDiffusion
from models import MLPDenoiser


def load_data(data_path="data/manufacturing/prepared.pt"):
    """Load prepared manufacturing data."""
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    return data


def load_preprocessors(preprocessor_path="data/manufacturing/preprocessors.pkl"):
    """Load preprocessors for inverse transform."""
    with open(preprocessor_path, "rb") as f:
        return pickle.load(f)


def load_model(checkpoint_path, device):
    """Load trained Gaussian diffusion model."""
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


def generate_samples(model, diffusion, n_samples, input_dim, device):
    """Generate synthetic samples using Gaussian diffusion."""
    print(f"Generating {n_samples} synthetic samples...")
    with torch.no_grad():
        samples = diffusion.sample(
            model,
            shape=(n_samples, input_dim),
            device=device,
            clip_denoised=True,
        )
    return samples.cpu().numpy()


def add_noise_augmentation(X, y, n_samples, noise_scale=0.1):
    """
    Simple noise augmentation baseline.
    Samples from training data and adds Gaussian noise.
    """
    indices = np.random.choice(len(X), n_samples, replace=True)
    X_aug = X[indices].copy()
    y_aug = y[indices].copy()

    # Add noise to features
    X_noise = np.random.normal(0, noise_scale * X.std(axis=0), X_aug.shape)
    X_aug += X_noise

    # Add noise to target
    y_noise = np.random.normal(0, noise_scale * y.std(), y_aug.shape)
    y_aug += y_noise

    return X_aug.astype(np.float32), y_aug.astype(np.float32)


def apply_smogn(X, y, target_col="target"):
    """
    Apply SMOGN (SMOTE for regression) to augment the dataset.
    Returns augmented features and target.
    """
    if not SMOGN_AVAILABLE:
        print("SMOGN not available, skipping...")
        return None, None

    import pandas as pd

    # Create DataFrame for SMOGN
    df = pd.DataFrame(X, columns=["Çap", "Boy"])
    df[target_col] = y

    try:
        # Apply SMOGN
        df_smogn = smogn.smoter(
            data=df,
            y=target_col,
            samp_method="extreme",
        )

        X_aug = df_smogn[["Çap", "Boy"]].values.astype(np.float32)
        y_aug = df_smogn[target_col].values.astype(np.float32)

        return X_aug, y_aug
    except Exception as e:
        print(f"SMOGN failed: {e}")
        return None, None


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
    preprocessors = load_preprocessors()

    # Get unscaled features and targets
    X_train = data["X_train_unscaled"]  # [Çap, Boy]
    X_test = data["X_test_unscaled"]
    y_train = data["y_train"]  # Original target in minutes
    y_test = data["y_test"]

    # Also get scaled data for diffusion
    X_train_scaled = data["X_train"].numpy()  # [Çap_scaled, Boy_scaled, target_scaled]

    print(f"\nData shapes:")
    print(f"  Train: {X_train.shape} features, {y_train.shape} target")
    print(f"  Test: {X_test.shape} features, {y_test.shape} target")

    # Load model
    checkpoint_path = Path("checkpoints/manufacturing/final_model.pt")
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please run train_manufacturing.py first.")
        return

    model, diffusion, checkpoint = load_model(checkpoint_path, args.device)
    num_features = checkpoint["data_config"]["num_features"]

    # Generate synthetic samples (scaled)
    n_synthetic = args.n_synthetic
    synthetic_scaled = generate_samples(model, diffusion, n_synthetic, num_features, args.device)

    # Inverse transform synthetic samples
    scaler = preprocessors["scaler"]
    target_scaler = preprocessors["target_scaler"]

    # Unscale: first undo the /3 clipping, then inverse transform
    synthetic_unclipped = synthetic_scaled * 3
    synthetic_features_unscaled = scaler.inverse_transform(synthetic_unclipped[:, :2])
    synthetic_target_unscaled = target_scaler.inverse_transform(
        synthetic_unclipped[:, 2:3]
    ).flatten()

    # Noise augmentation baseline (on unscaled data)
    noise_features, noise_target = add_noise_augmentation(X_train, y_train, n_synthetic, noise_scale=0.1)

    # Apply SMOGN
    print("\nApplying SMOGN...")
    X_smogn, y_smogn = apply_smogn(X_train, y_train)

    # Prepare datasets
    datasets = {
        "Original": (X_train, y_train),
        "+ Diffusion": (
            np.vstack([X_train, synthetic_features_unscaled]),
            np.concatenate([y_train, synthetic_target_unscaled])
        ),
        "+ Noise": (
            np.vstack([X_train, noise_features]),
            np.concatenate([y_train, noise_target])
        ),
    }

    if X_smogn is not None:
        datasets["+ SMOGN"] = (X_smogn, y_smogn)

    print(f"\nDataset sizes:")
    for name, (X, y) in datasets.items():
        print(f"  {name}: {len(X)} samples")

    # Evaluate regressors
    print("\n" + "=" * 70)
    print("EXPERIMENT 011: AUGMENTATION COMPARISON")
    print("=" * 70)
    print("\nPredicting: Makine Süre (100.000 ADET) DK - Machine duration")
    print("Features: Çap, Boy only (no categorical)")

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

            metrics = evaluate_regressor(reg_name, reg, X, y, X_test, y_test)
            results[reg_name][data_name] = metrics

            if data_name == "Original":
                print(f"  {data_name:12}: RMSE={metrics['rmse']:.1f} min, R²={metrics['r2']:.4f}")
                baseline_rmse = metrics['rmse']
            else:
                delta = metrics['rmse'] - baseline_rmse
                pct = delta / baseline_rmse * 100
                print(f"  {data_name:12}: RMSE={metrics['rmse']:.1f} min ({pct:+.1f}%), R²={metrics['r2']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Diffusion vs SMOGN")
    print("=" * 70)

    for reg_name in results:
        orig_rmse = results[reg_name]["Original"]["rmse"]
        diff_rmse = results[reg_name]["+ Diffusion"]["rmse"]
        noise_rmse = results[reg_name]["+ Noise"]["rmse"]
        smogn_rmse = results[reg_name].get("+ SMOGN", {}).get("rmse")

        diff_vs_orig = (diff_rmse - orig_rmse) / orig_rmse * 100
        noise_vs_orig = (noise_rmse - orig_rmse) / orig_rmse * 100

        print(f"\n{reg_name}:")
        print(f"  Original:  RMSE={orig_rmse:.1f} min")
        print(f"  Diffusion: RMSE={diff_rmse:.1f} min ({diff_vs_orig:+.1f}%)")
        print(f"  Noise:     RMSE={noise_rmse:.1f} min ({noise_vs_orig:+.1f}%)")

        if smogn_rmse:
            smogn_vs_orig = (smogn_rmse - orig_rmse) / orig_rmse * 100
            diff_vs_smogn = (smogn_rmse - diff_rmse) / smogn_rmse * 100
            print(f"  SMOGN:     RMSE={smogn_rmse:.1f} min ({smogn_vs_orig:+.1f}%)")
            print(f"  -> Diffusion beats SMOGN by {diff_vs_smogn:.1f}%")

    # Final verdict
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    diffusion_wins = 0
    smogn_wins = 0

    for reg_name in results:
        diff_rmse = results[reg_name]["+ Diffusion"]["rmse"]
        smogn_rmse = results[reg_name].get("+ SMOGN", {}).get("rmse")
        if smogn_rmse:
            if diff_rmse < smogn_rmse:
                diffusion_wins += 1
            else:
                smogn_wins += 1

    print(f"\nDiffusion beats SMOGN: {diffusion_wins}/{diffusion_wins + smogn_wins} models")
    if diffusion_wins > smogn_wins:
        print("-> Diffusion-based augmentation outperforms traditional SMOGN!")
    elif diffusion_wins == smogn_wins:
        print("-> Results are mixed between Diffusion and SMOGN")
    else:
        print("-> SMOGN outperforms Diffusion on this dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/manufacturing/prepared.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_synthetic", type=int, default=1000,
                        help="Number of synthetic samples to generate")
    args = parser.parse_args()

    main(args)
