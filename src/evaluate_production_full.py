"""
Evaluate HybridDiffusion model on production data (numeric + categorical).

Compares augmentation methods for regression:
- Original data only
- Hybrid Diffusion augmentation
- Noise augmentation (baseline)

Usage:
    python src/evaluate_production_full.py --device cuda --n_synthetic 1000
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

from diffusion import HybridDiffusion
from models import HybridMLPDenoiser


def load_data(data_path="data/production/full.pt"):
    """Load prepared production data."""
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    return data


def load_model(checkpoint_path, device):
    """Load trained hybrid diffusion model."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint["model_config"]
    diffusion_config = checkpoint["diffusion_config"]

    model = HybridMLPDenoiser(
        num_numerical=model_config["num_numerical"],
        cat_cardinalities=model_config["cat_cardinalities"],
        hidden_dims=model_config["hidden_dims"],
        dropout=model_config.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    diffusion = HybridDiffusion(
        num_numerical=model_config["num_numerical"],
        cat_cardinalities=model_config["cat_cardinalities"],
        num_timesteps=diffusion_config["num_timesteps"],
        beta_schedule=diffusion_config["beta_schedule"],
    ).to(device)

    return model, diffusion, checkpoint


def generate_samples(model, diffusion, n_samples, device):
    """Generate synthetic samples using hybrid diffusion."""
    print(f"Generating {n_samples} synthetic samples...")
    with torch.no_grad():
        samples = diffusion.sample(
            model,
            batch_size=n_samples,
            device=device,
            clip_denoised=True,
        )
    return samples.cpu().numpy()


def add_noise_augmentation(X, n_samples, num_numerical, noise_scale=0.1):
    """
    Simple noise augmentation baseline.
    Only adds noise to numerical features, keeps categorical as-is.
    """
    indices = np.random.choice(len(X), n_samples, replace=True)
    X_aug = X[indices].copy()

    # Add noise only to numerical part
    noise = np.random.normal(0, noise_scale, (n_samples, num_numerical))
    X_aug[:, :num_numerical] += noise

    return X_aug.astype(np.float32)


def apply_smogn(X_features, y, target_col="target"):
    """
    Apply SMOGN (SMOTE for regression) to augment the dataset.
    Returns augmented features and target.
    """
    if not SMOGN_AVAILABLE:
        print("SMOGN not available, skipping...")
        return None, None

    import pandas as pd

    # Create DataFrame for SMOGN
    df = pd.DataFrame(X_features, columns=[f"f{i}" for i in range(X_features.shape[1])])
    df[target_col] = y

    try:
        # Apply SMOGN
        df_smogn = smogn.smoter(
            data=df,
            y=target_col,
            samp_method="extreme",
        )

        X_aug = df_smogn.drop(columns=[target_col]).values.astype(np.float32)
        y_aug = df_smogn[target_col].values

        return X_aug, y_aug
    except Exception as e:
        print(f"SMOGN failed: {e}")
        return None, None


def decode_for_ml(X, num_numerical, cat_cardinalities):
    """
    Convert hybrid format to ML-ready format.
    Numerical stays as-is, categorical becomes argmax indices.
    """
    X_num = X[:, :num_numerical]

    X_cat = []
    offset = num_numerical
    for card in cat_cardinalities:
        onehot = X[:, offset:offset + card]
        cat_idx = np.argmax(onehot, axis=1)
        X_cat.append(cat_idx)
        offset += card

    X_cat = np.column_stack(X_cat) if X_cat else np.zeros((len(X), 0))

    return np.hstack([X_num, X_cat.astype(np.float32)])


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

    X_train_raw = data["X_train"].numpy()
    X_test_raw = data["X_test"].numpy()
    num_numerical = data["num_numerical"]
    cat_cardinalities = data["cat_cardinalities"]

    # Decode to ML format (target is last numerical column)
    X_train_ml = decode_for_ml(X_train_raw, num_numerical, cat_cardinalities)
    X_test_ml = decode_for_ml(X_test_raw, num_numerical, cat_cardinalities)

    # Split features and target (target is last numerical column, before categorical)
    X_train_features = np.hstack([X_train_ml[:, :num_numerical-1], X_train_ml[:, num_numerical:]])
    y_train = X_train_ml[:, num_numerical-1]
    X_test_features = np.hstack([X_test_ml[:, :num_numerical-1], X_test_ml[:, num_numerical:]])
    y_test = X_test_ml[:, num_numerical-1]

    print(f"\nData shapes:")
    print(f"  Train: {X_train_features.shape} features, {y_train.shape} target")
    print(f"  Test: {X_test_features.shape} features, {y_test.shape} target")

    # Load model
    checkpoint_path = Path("checkpoints/production_full/final_model.pt")
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please run train_production_full.py first.")
        return

    model, diffusion, checkpoint = load_model(checkpoint_path, args.device)

    # Generate synthetic samples
    n_synthetic = args.n_synthetic
    synthetic_raw = generate_samples(model, diffusion, n_synthetic, args.device)
    synthetic_ml = decode_for_ml(synthetic_raw, num_numerical, cat_cardinalities)
    synthetic_features = np.hstack([synthetic_ml[:, :num_numerical-1], synthetic_ml[:, num_numerical:]])
    synthetic_target = synthetic_ml[:, num_numerical-1]

    # Noise augmentation baseline
    noise_raw = add_noise_augmentation(X_train_raw, n_synthetic, num_numerical, noise_scale=0.1)
    noise_ml = decode_for_ml(noise_raw, num_numerical, cat_cardinalities)
    noise_features = np.hstack([noise_ml[:, :num_numerical-1], noise_ml[:, num_numerical:]])
    noise_target = noise_ml[:, num_numerical-1]

    # Apply SMOGN
    print("Applying SMOGN...")
    X_smogn, y_smogn = apply_smogn(X_train_features, y_train)

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

    if X_smogn is not None:
        datasets["+ SMOGN"] = (X_smogn, y_smogn)

    print(f"\nDataset sizes:")
    for name, (X, y) in datasets.items():
        print(f"  {name}: {len(X)} samples")

    # Evaluate regressors
    print("\n" + "=" * 70)
    print("ML EFFICIENCY EVALUATION (Regression with Categorical Features)")
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
        smogn_rmse = results[reg_name].get("+ SMOGN", {}).get("rmse")

        # Compare diffusion vs best baseline
        baselines = {"Noise": noise_rmse}
        if smogn_rmse:
            baselines["SMOGN"] = smogn_rmse

        best_baseline_name = min(baselines, key=baselines.get)
        best_baseline_rmse = baselines[best_baseline_name]

        diff_better = diff_rmse < best_baseline_rmse
        status = "BETTER" if diff_better else "WORSE"

        print(f"\n{reg_name}: Diffusion vs {best_baseline_name} = {status}")
        print(f"  Original:  RMSE={orig_rmse:.4f}")
        print(f"  Diffusion: RMSE={diff_rmse:.4f} ({diff_rmse - orig_rmse:+.4f})")
        print(f"  Noise:     RMSE={noise_rmse:.4f} ({noise_rmse - orig_rmse:+.4f})")
        if smogn_rmse:
            print(f"  SMOGN:     RMSE={smogn_rmse:.4f} ({smogn_rmse - orig_rmse:+.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/production/full.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_synthetic", type=int, default=1000,
                        help="Number of synthetic samples to generate")
    args = parser.parse_args()

    main(args)
