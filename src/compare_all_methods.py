"""
Compare all synthetic data generation methods on Ozel Rich dataset.

Methods compared:
1. Diffusion (HybridDiffusion)
2. CTGAN (GAN-based)
3. SMOGN (Interpolation-based)

This creates the final comparison table for the thesis.

Usage:
    python src/compare_all_methods.py --device cuda
"""

import argparse
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

try:
    from ctgan import CTGAN
    CTGAN_AVAILABLE = True
except ImportError:
    CTGAN_AVAILABLE = False

try:
    import smogn
    SMOGN_AVAILABLE = True
except ImportError:
    SMOGN_AVAILABLE = False

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


def load_diffusion_model(checkpoint_path, device):
    """Load trained Hybrid diffusion model."""
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


def generate_diffusion_samples(model, diffusion, n_samples, device, preprocessors, data):
    """Generate and process diffusion samples."""
    with torch.no_grad():
        samples = diffusion.sample(
            model,
            batch_size=n_samples,
            device=device,
            clip_denoised=True,
        )
    samples = samples.cpu().numpy()

    # Process samples
    scaler = preprocessors["scaler"]
    target_scaler = preprocessors["target_scaler"]
    num_numerical = data["num_numerical"]
    cat_cardinalities = data["cat_cardinalities"]

    num_scaled = samples[:, :num_numerical]
    cat_probs = samples[:, num_numerical:]

    # Unscale numeric
    num_unclipped = num_scaled * 3
    features_unscaled = scaler.inverse_transform(num_unclipped[:, :2])
    target_unscaled = target_scaler.inverse_transform(num_unclipped[:, 2:3]).flatten()

    # Convert categorical probabilities to one-hot
    offset = 0
    cat_onehot_list = []
    for card in cat_cardinalities:
        probs = cat_probs[:, offset:offset + card]
        indices = probs.argmax(axis=1)
        cat_onehot_list.append(np.eye(card)[indices])
        offset += card
    cat_onehot = np.hstack(cat_onehot_list).astype(np.float32)

    X = np.column_stack([features_unscaled, cat_onehot])
    y = target_unscaled

    return X, y


def generate_ctgan_samples(n_samples, cat_columns, cat_cardinalities):
    """Load CTGAN model and generate samples."""
    if not CTGAN_AVAILABLE:
        return None, None

    model_path = Path("checkpoints/ctgan/ozel_rich_model.pkl")
    if not model_path.exists():
        print(f"CTGAN model not found: {model_path}")
        return None, None

    ctgan = CTGAN.load(model_path)
    synthetic_df = ctgan.sample(n_samples)

    # Convert to ML features
    X_num = synthetic_df[["Cap", "Boy"]].values.astype(np.float32)
    y = synthetic_df["MakineSure"].values.astype(np.float32)

    onehot_list = []
    for col_name in cat_columns:
        col_idx = list(cat_columns).index(col_name)
        card = cat_cardinalities[col_idx]

        # Map string categories to indices
        unique_vals = sorted(synthetic_df[col_name].unique())
        val_to_idx = {v: i for i, v in enumerate(unique_vals)}
        indices = synthetic_df[col_name].map(val_to_idx).values
        indices = np.clip(indices, 0, card - 1)
        onehot_list.append(np.eye(card)[indices])

    cat_onehot = np.hstack(onehot_list).astype(np.float32)
    X = np.column_stack([X_num, cat_onehot])

    return X, y


def generate_smogn_samples(X_train_num, y_train, cat_train, cat_cardinalities):
    """Generate SMOGN samples."""
    if not SMOGN_AVAILABLE:
        return None, None

    df = pd.DataFrame(X_train_num, columns=["Cap", "Boy"])
    df["target"] = y_train

    try:
        df_smogn = smogn.smoter(
            data=df,
            y="target",
            samp_method="extreme",
        )
        X_smogn = df_smogn[["Cap", "Boy"]].values.astype(np.float32)
        y_smogn = df_smogn["target"].values.astype(np.float32)

        # Sample categoricals from original distribution
        smogn_cat_indices = np.random.choice(len(cat_train), len(X_smogn), replace=True)
        cat_onehot_list = []
        for i, card in enumerate(cat_cardinalities):
            cat_onehot_list.append(np.eye(card)[cat_train[smogn_cat_indices, i]])
        cat_onehot = np.hstack(cat_onehot_list).astype(np.float32)

        X = np.column_stack([X_smogn, cat_onehot])

        return X, y_smogn
    except Exception as e:
        print(f"SMOGN failed: {e}")
        return None, None


def evaluate_regressor(model, X_train, y_train, X_test, y_test):
    """Train and evaluate a regressor."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {"rmse": rmse, "r2": r2}


def main(args):
    # Load data
    data = load_data(args.data_path)
    preprocessors = load_preprocessors()

    # Get data
    X_train_num = data["X_train_num_unscaled"]
    X_test_num = data["X_test_num_unscaled"]
    cat_train = data["cat_train"]
    cat_test = data["cat_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    cat_cardinalities = data["cat_cardinalities"]
    cat_columns = data["cat_columns"]

    # One-hot encode
    def onehot_encode(cat_idx, cardinalities):
        result = []
        for i, card in enumerate(cardinalities):
            result.append(np.eye(card)[cat_idx[:, i]])
        return np.hstack(result).astype(np.float32)

    cat_train_onehot = onehot_encode(cat_train, cat_cardinalities)
    cat_test_onehot = onehot_encode(cat_test, cat_cardinalities)

    X_train_ml = np.column_stack([X_train_num, cat_train_onehot])
    X_test_ml = np.column_stack([X_test_num, cat_test_onehot])

    print(f"\nData summary:")
    print(f"  Training: {len(X_train_ml)} samples, {X_train_ml.shape[1]} features")
    print(f"  Test: {len(X_test_ml)} samples")

    # Generate synthetic data from each method
    n_synthetic = len(X_train_ml)

    print(f"\n{'='*70}")
    print(f"GENERATING SYNTHETIC DATA")
    print(f"{'='*70}")

    # 1. Diffusion
    print(f"\n1. Diffusion...")
    checkpoint_path = Path("checkpoints/ozel_rich/final_model.pt")
    if checkpoint_path.exists():
        model, diffusion, _ = load_diffusion_model(checkpoint_path, args.device)
        X_diffusion, y_diffusion = generate_diffusion_samples(
            model, diffusion, n_synthetic, args.device, preprocessors, data
        )
        print(f"   Generated {len(X_diffusion)} samples")
    else:
        print(f"   Model not found: {checkpoint_path}")
        X_diffusion, y_diffusion = None, None

    # 2. CTGAN
    print(f"\n2. CTGAN...")
    X_ctgan, y_ctgan = generate_ctgan_samples(n_synthetic, cat_columns, cat_cardinalities)
    if X_ctgan is not None:
        print(f"   Generated {len(X_ctgan)} samples")
    else:
        print(f"   Not available")

    # 3. SMOGN
    print(f"\n3. SMOGN...")
    X_smogn, y_smogn = generate_smogn_samples(X_train_num, y_train, cat_train, cat_cardinalities)
    if X_smogn is not None:
        print(f"   Generated {len(X_smogn)} samples")
    else:
        print(f"   Not available")

    # Prepare datasets
    datasets = {"Original": (X_train_ml, y_train)}

    if X_diffusion is not None:
        datasets["Diffusion"] = (X_diffusion, y_diffusion)

    if X_ctgan is not None:
        datasets["CTGAN"] = (X_ctgan, y_ctgan)

    if X_smogn is not None:
        datasets["SMOGN"] = (X_smogn, y_smogn)

    # Evaluate
    print(f"\n{'='*70}")
    print(f"EVALUATION: Train on Synthetic, Test on Real")
    print(f"{'='*70}")

    regressors = [
        ("Random Forest", lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ("Gradient Boosting", lambda: GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("Ridge", lambda: Ridge(alpha=1.0)),
    ]

    results = {}

    for reg_name, reg_factory in regressors:
        print(f"\n{reg_name}:")
        print("-" * 60)

        results[reg_name] = {}
        baseline_r2 = None

        for data_name, (X, y) in datasets.items():
            reg = reg_factory()
            metrics = evaluate_regressor(reg, X, y, X_test_ml, y_test)
            results[reg_name][data_name] = metrics

            if data_name == "Original":
                baseline_r2 = metrics['r2']
                print(f"  {data_name:12}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.1f}")
            else:
                delta = metrics['r2'] - baseline_r2
                print(f"  {data_name:12}: R²={metrics['r2']:.4f} ({delta:+.4f}), RMSE={metrics['rmse']:.1f}")

    # Final summary table
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON: Train on Synthetic, Test on Real")
    print(f"{'='*70}")

    print("\n| Method | RF R² | GB R² | Ridge R² | Avg R² |")
    print("|--------|-------|-------|----------|--------|")

    for data_name in datasets.keys():
        rf_r2 = results["Random Forest"][data_name]["r2"]
        gb_r2 = results["Gradient Boosting"][data_name]["r2"]
        ridge_r2 = results["Ridge"][data_name]["r2"]
        avg_r2 = (rf_r2 + gb_r2 + ridge_r2) / 3
        print(f"| {data_name:8} | {rf_r2:.4f} | {gb_r2:.4f} | {ridge_r2:.4f} | {avg_r2:.4f} |")

    # Interpretation
    print(f"\n{'='*70}")
    print(f"INTERPRETATION")
    print(f"{'='*70}")

    orig_avg = np.mean([results[r]["Original"]["r2"] for r in results])

    if "Diffusion" in datasets:
        diff_avg = np.mean([results[r]["Diffusion"]["r2"] for r in results])
        print(f"\nDiffusion vs Original: {diff_avg - orig_avg:+.4f} R²")

    if "CTGAN" in datasets:
        ctgan_avg = np.mean([results[r]["CTGAN"]["r2"] for r in results])
        print(f"CTGAN vs Original: {ctgan_avg - orig_avg:+.4f} R²")

    if "SMOGN" in datasets:
        smogn_avg = np.mean([results[r]["SMOGN"]["r2"] for r in results])
        print(f"SMOGN vs Original: {smogn_avg - orig_avg:+.4f} R²")

    # Winner
    print(f"\n{'='*70}")
    print(f"CONCLUSION")
    print(f"{'='*70}")

    method_avgs = {}
    for data_name in datasets.keys():
        if data_name != "Original":
            avg = np.mean([results[r][data_name]["r2"] for r in results])
            method_avgs[data_name] = avg

    if method_avgs:
        best_method = max(method_avgs, key=method_avgs.get)
        worst_method = min(method_avgs, key=method_avgs.get)

        print(f"\nBest synthetic method: {best_method} (avg R² = {method_avgs[best_method]:.4f})")
        print(f"Worst synthetic method: {worst_method} (avg R² = {method_avgs[worst_method]:.4f})")

        if method_avgs[best_method] >= orig_avg * 0.95:
            print(f"\n-> {best_method} generates realistic synthetic data!")
            print(f"   Models trained on {best_method} synthetic data work on real data.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/ozel_rich/prepared.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    main(args)
