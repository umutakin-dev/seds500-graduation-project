"""
Evaluate HybridDiffusion model on ozel rich dataset (from manufacturing data).

Compares augmentation methods for regression:
- Original data only (baseline)
- Hybrid Diffusion augmentation
- Noise augmentation (random baseline)
- SMOGN augmentation (traditional method)

Uses Cap, Boy (numeric) and IslemTipi, AnmaOlcusu, SartnameSimple, UY (categorical).
This is Experiment 013.

Usage:
    python src/evaluate_ozel_rich.py --device cuda --n_synthetic 500
"""

import argparse
import pickle
import torch
import torch.nn.functional as F
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

from diffusion import HybridDiffusion
from models import MLPDenoiser


def load_data(data_path="data/ozel_rich/prepared.pt"):
    """Load prepared ozel rich data."""
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    return data


def load_preprocessors(preprocessor_path="data/ozel_rich/preprocessors.pkl"):
    """Load preprocessors for inverse transform."""
    with open(preprocessor_path, "rb") as f:
        return pickle.load(f)


def load_model(checkpoint_path, device):
    """Load trained Hybrid diffusion model."""
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

    diffusion = HybridDiffusion(
        num_numerical=diffusion_config["num_numerical"],
        cat_cardinalities=diffusion_config["cat_cardinalities"],
        num_timesteps=diffusion_config["num_timesteps"],
        beta_schedule=diffusion_config["beta_schedule"],
    ).to(device)

    return model, diffusion, checkpoint


def generate_samples(model, diffusion, n_samples, device):
    """Generate synthetic samples using Hybrid diffusion."""
    print(f"Generating {n_samples} synthetic samples...")
    with torch.no_grad():
        samples = diffusion.sample(
            model,
            batch_size=n_samples,
            device=device,
            clip_denoised=True,
        )
    return samples.cpu().numpy()


def add_noise_augmentation(X_num, cat_idx, y, n_samples, noise_scale=0.1):
    """Simple noise augmentation baseline."""
    indices = np.random.choice(len(X_num), n_samples, replace=True)
    X_num_aug = X_num[indices].copy()
    cat_aug = cat_idx[indices].copy()
    y_aug = y[indices].copy()

    # Add noise to numeric features only
    X_noise = np.random.normal(0, noise_scale * X_num.std(axis=0), X_num_aug.shape)
    X_num_aug += X_noise

    # Add noise to target
    y_noise = np.random.normal(0, noise_scale * y.std(), y_aug.shape)
    y_aug += y_noise

    return X_num_aug.astype(np.float32), cat_aug, y_aug.astype(np.float32)


def apply_smogn(X_num, y, target_col="target"):
    """
    Apply SMOGN (SMOTE for regression) to augment numeric features only.
    Note: SMOGN doesn't handle high-cardinality categoricals well, so we only use numeric.
    """
    if not SMOGN_AVAILABLE:
        print("SMOGN not available, skipping...")
        return None, None

    import pandas as pd

    # Create DataFrame for SMOGN (numeric only)
    df = pd.DataFrame(X_num, columns=["Cap", "Boy"])
    df[target_col] = y

    try:
        # Apply SMOGN
        df_smogn = smogn.smoter(
            data=df,
            y=target_col,
            samp_method="extreme",
        )

        X_num_aug = df_smogn[["Cap", "Boy"]].values.astype(np.float32)
        y_aug = df_smogn[target_col].values.astype(np.float32)

        return X_num_aug, y_aug
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

    # Get unscaled numeric features and categorical indices
    X_train_num = data["X_train_num_unscaled"]  # [Cap, Boy]
    X_test_num = data["X_test_num_unscaled"]
    cat_train = data["cat_train"]  # [IslemTipi, AnmaOlcusu, SartnameSimple, UY]
    cat_test = data["cat_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    cat_cardinalities = data["cat_cardinalities"]
    cat_columns = data["cat_columns"]
    num_numerical = data["num_numerical"]

    # One-hot encode categoricals
    def onehot_encode(cat_idx, cardinalities):
        result = []
        for i, card in enumerate(cardinalities):
            result.append(np.eye(card)[cat_idx[:, i]])
        return np.hstack(result).astype(np.float32)

    cat_train_onehot = onehot_encode(cat_train, cat_cardinalities)
    cat_test_onehot = onehot_encode(cat_test, cat_cardinalities)

    # Combine features for ML
    X_train_ml = np.column_stack([X_train_num, cat_train_onehot])
    X_test_ml = np.column_stack([X_test_num, cat_test_onehot])

    print(f"\nData shapes:")
    print(f"  Train: {X_train_ml.shape} features, {y_train.shape} target")
    print(f"  Test: {X_test_ml.shape} features, {y_test.shape} target")
    print(f"  Features: Cap, Boy + {cat_columns}")

    # Load model
    checkpoint_path = Path("checkpoints/ozel_rich/final_model.pt")
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please run train_ozel_rich.py first.")
        return

    model, diffusion, checkpoint = load_model(checkpoint_path, args.device)

    # Generate synthetic samples (scaled)
    n_synthetic = args.n_synthetic
    synthetic_scaled = generate_samples(model, diffusion, n_synthetic, args.device)

    # Inverse transform synthetic samples
    scaler = preprocessors["scaler"]
    target_scaler = preprocessors["target_scaler"]

    # Split synthetic into numeric and categorical
    synthetic_num_scaled = synthetic_scaled[:, :num_numerical]  # Cap, Boy, target
    synthetic_cat_probs = synthetic_scaled[:, num_numerical:]   # All categorical probs

    # Unscale numeric
    synthetic_num_unclipped = synthetic_num_scaled * 3
    synthetic_features_unscaled = scaler.inverse_transform(synthetic_num_unclipped[:, :2])
    synthetic_target_unscaled = target_scaler.inverse_transform(
        synthetic_num_unclipped[:, 2:3]
    ).flatten()

    # Convert categorical probabilities to indices and one-hot
    offset = 0
    synthetic_cat_onehot_list = []
    for card in cat_cardinalities:
        probs = synthetic_cat_probs[:, offset:offset + card]
        indices = probs.argmax(axis=1)
        synthetic_cat_onehot_list.append(np.eye(card)[indices])
        offset += card
    synthetic_cat_onehot = np.hstack(synthetic_cat_onehot_list).astype(np.float32)

    # Noise augmentation
    noise_features, noise_cat, noise_target = add_noise_augmentation(
        X_train_num, cat_train, y_train, n_synthetic, noise_scale=0.1
    )
    noise_cat_onehot = onehot_encode(noise_cat, cat_cardinalities)

    # Apply SMOGN (numeric only, then sample categoricals from original)
    print("\nApplying SMOGN...")
    X_smogn_num, y_smogn = apply_smogn(X_train_num, y_train)

    # Prepare datasets
    datasets = {
        "Original": (X_train_ml, y_train),
        "+ Diffusion": (
            np.vstack([X_train_ml, np.column_stack([synthetic_features_unscaled, synthetic_cat_onehot])]),
            np.concatenate([y_train, synthetic_target_unscaled])
        ),
        "+ Noise": (
            np.vstack([X_train_ml, np.column_stack([noise_features, noise_cat_onehot])]),
            np.concatenate([y_train, noise_target])
        ),
    }

    if X_smogn_num is not None:
        # For SMOGN, sample categoricals from original distribution
        smogn_cat_indices = np.random.choice(len(cat_train), len(X_smogn_num), replace=True)
        smogn_cat_onehot = onehot_encode(cat_train[smogn_cat_indices], cat_cardinalities)
        datasets["+ SMOGN"] = (
            np.column_stack([X_smogn_num, smogn_cat_onehot]),
            y_smogn
        )

    print(f"\nDataset sizes:")
    for name, (X, y) in datasets.items():
        print(f"  {name}: {len(X)} samples")

    # Evaluate regressors
    print("\n" + "=" * 70)
    print("EXPERIMENT 013: OZEL RICH DATASET AUGMENTATION COMPARISON")
    print("=" * 70)
    print("\nPredicting: MakineSure - Machine duration (minutes for 100k units)")
    print(f"Features: Cap, Boy + {cat_columns}")

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

            metrics = evaluate_regressor(reg_name, reg, X, y, X_test_ml, y_test)
            results[reg_name][data_name] = metrics

            if data_name == "Original":
                print(f"  {data_name:12}: RMSE={metrics['rmse']:.1f} min, R2={metrics['r2']:.4f}")
                baseline_rmse = metrics['rmse']
            else:
                delta = metrics['rmse'] - baseline_rmse
                pct = delta / baseline_rmse * 100
                print(f"  {data_name:12}: RMSE={metrics['rmse']:.1f} min ({pct:+.1f}%), R2={metrics['r2']:.4f}")

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
    parser.add_argument("--data_path", type=str, default="data/ozel_rich/prepared.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_synthetic", type=int, default=500,
                        help="Number of synthetic samples to generate")
    args = parser.parse_args()

    main(args)
