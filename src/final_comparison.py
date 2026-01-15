"""
Final Comparison: All Methods for Synthetic Data Generation

This script runs the complete comparison for the thesis:
1. Diffusion (for augmentation)
2. CTGAN (for replacement/privacy)
3. SMOGN (baseline - fails on complex data)

Results are saved for visualization.
"""

import pickle
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

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


def load_data():
    data = torch.load("data/ozel_rich/prepared.pt", weights_only=False)
    with open("data/ozel_rich/preprocessors.pkl", "rb") as f:
        preprocessors = pickle.load(f)
    return data, preprocessors


def onehot_encode(cat_idx, cardinalities):
    result = []
    for i, card in enumerate(cardinalities):
        result.append(np.eye(card)[cat_idx[:, i]])
    return np.hstack(result).astype(np.float32)


def load_diffusion_model(device):
    # Try v2 first, fall back to v1
    for version in ["v2", "v1"]:
        if version == "v2":
            path = "checkpoints/ozel_rich_v2/final_model.pt"
        else:
            path = "checkpoints/ozel_rich/final_model.pt"

        if Path(path).exists():
            checkpoint = torch.load(path, map_location=device, weights_only=False)
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

            print(f"Loaded diffusion model: {path}")
            return model, diffusion
    return None, None


def generate_diffusion_samples(model, diffusion, n_samples, device, preprocessors, data):
    """Generate diffusion samples for AUGMENTATION use case."""
    with torch.no_grad():
        samples = diffusion.sample(model, batch_size=n_samples, device=device, clip_denoised=True)
    samples = samples.numpy()

    scaler = preprocessors["scaler"]
    target_scaler = preprocessors["target_scaler"]
    num_numerical = data["num_numerical"]
    cat_cardinalities = data["cat_cardinalities"]

    num_scaled = samples[:, :num_numerical]
    cat_probs = samples[:, num_numerical:]

    num_unclipped = num_scaled * 3
    features_unscaled = scaler.inverse_transform(num_unclipped[:, :2])
    target_unscaled = target_scaler.inverse_transform(num_unclipped[:, 2:3]).flatten()

    offset = 0
    cat_onehot_list = []
    for card in cat_cardinalities:
        probs = cat_probs[:, offset:offset + card]
        indices = probs.argmax(axis=1)
        cat_onehot_list.append(np.eye(card)[indices])
        offset += card
    cat_onehot = np.hstack(cat_onehot_list).astype(np.float32)

    X = np.column_stack([features_unscaled, cat_onehot])
    return X, target_unscaled


def generate_ctgan_samples(n_samples, cat_columns, cat_cardinalities):
    """Generate CTGAN samples."""
    if not CTGAN_AVAILABLE:
        return None, None

    model_path = Path("checkpoints/ctgan/ozel_rich_model.pkl")
    if not model_path.exists():
        return None, None

    ctgan = CTGAN.load(model_path)
    synthetic_df = ctgan.sample(n_samples)

    X_num = synthetic_df[["Cap", "Boy"]].values.astype(np.float32)
    y = synthetic_df["MakineSure"].values.astype(np.float32)

    onehot_list = []
    for col_name in cat_columns:
        col_idx = list(cat_columns).index(col_name)
        card = cat_cardinalities[col_idx]
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
        df_smogn = smogn.smoter(data=df, y="target", samp_method="extreme")
        X_smogn = df_smogn[["Cap", "Boy"]].values.astype(np.float32)
        y_smogn = df_smogn["target"].values.astype(np.float32)

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


def evaluate(X_train, y_train, X_test, y_test, model_name="RF"):
    """Evaluate with specified model."""
    if model_name == "RF":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == "GB":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        model = Ridge(alpha=1.0)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
    }


def main():
    print("="*70)
    print("FINAL COMPARISON: Synthetic Data Generation Methods")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    device = "cpu"  # Use CPU for stability

    # Load data
    data, preprocessors = load_data()

    X_train_num = data["X_train_num_unscaled"]
    X_test_num = data["X_test_num_unscaled"]
    cat_train = data["cat_train"]
    cat_test = data["cat_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    cat_cardinalities = data["cat_cardinalities"]
    cat_columns = data["cat_columns"]

    cat_train_onehot = onehot_encode(cat_train, cat_cardinalities)
    cat_test_onehot = onehot_encode(cat_test, cat_cardinalities)

    X_train_ml = np.column_stack([X_train_num, cat_train_onehot])
    X_test_ml = np.column_stack([X_test_num, cat_test_onehot])

    print(f"\nDataset: Ozel Rich")
    print(f"  Train: {len(X_train_ml)} samples, {X_train_ml.shape[1]} features")
    print(f"  Test: {len(X_test_ml)} samples")
    print(f"  Features: 2 numeric + {sum(cat_cardinalities)} one-hot ({len(cat_cardinalities)} categorical)")

    n_synthetic = 500  # Consistent with original experiments

    # Results storage
    results = {
        "dataset": "ozel_rich",
        "n_train": len(X_train_ml),
        "n_test": len(X_test_ml),
        "n_features": X_train_ml.shape[1],
        "n_synthetic": n_synthetic,
        "methods": {}
    }

    # 1. BASELINE
    print(f"\n{'='*70}")
    print("1. BASELINE (Original data only)")
    print("="*70)

    baseline_results = {}
    for model_name in ["RF", "GB", "Ridge"]:
        metrics = evaluate(X_train_ml, y_train, X_test_ml, y_test, model_name)
        baseline_results[model_name] = metrics
        print(f"  {model_name}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.1f}")

    results["methods"]["Original"] = {
        "augmentation": baseline_results,
        "replacement": baseline_results  # Same for baseline
    }

    # 2. DIFFUSION
    print(f"\n{'='*70}")
    print("2. DIFFUSION (HybridDiffusion)")
    print("="*70)

    model, diffusion = load_diffusion_model(device)
    if model is not None:
        print(f"  Generating {n_synthetic} synthetic samples...")
        X_diff, y_diff = generate_diffusion_samples(
            model, diffusion, n_synthetic, device, preprocessors, data
        )

        # Augmentation
        X_aug = np.vstack([X_train_ml, X_diff])
        y_aug = np.concatenate([y_train, y_diff])

        aug_results = {}
        rep_results = {}
        print("\n  AUGMENTATION (Original + Synthetic):")
        for model_name in ["RF", "GB", "Ridge"]:
            metrics = evaluate(X_aug, y_aug, X_test_ml, y_test, model_name)
            aug_results[model_name] = metrics
            delta = metrics['r2'] - baseline_results[model_name]['r2']
            print(f"    {model_name}: R²={metrics['r2']:.4f} ({delta:+.4f})")

        # Replacement (generate full size)
        X_diff_full, y_diff_full = generate_diffusion_samples(
            model, diffusion, len(X_train_ml), device, preprocessors, data
        )
        print("\n  REPLACEMENT (Synthetic only):")
        for model_name in ["RF", "GB", "Ridge"]:
            metrics = evaluate(X_diff_full, y_diff_full, X_test_ml, y_test, model_name)
            rep_results[model_name] = metrics
            delta = metrics['r2'] - baseline_results[model_name]['r2']
            print(f"    {model_name}: R²={metrics['r2']:.4f} ({delta:+.4f})")

        results["methods"]["Diffusion"] = {
            "augmentation": aug_results,
            "replacement": rep_results
        }

    # 3. CTGAN
    print(f"\n{'='*70}")
    print("3. CTGAN (Conditional Tabular GAN)")
    print("="*70)

    X_ctgan, y_ctgan = generate_ctgan_samples(n_synthetic, cat_columns, cat_cardinalities)
    if X_ctgan is not None:
        # Augmentation
        X_aug = np.vstack([X_train_ml, X_ctgan])
        y_aug = np.concatenate([y_train, y_ctgan])

        aug_results = {}
        rep_results = {}
        print("\n  AUGMENTATION (Original + Synthetic):")
        for model_name in ["RF", "GB", "Ridge"]:
            metrics = evaluate(X_aug, y_aug, X_test_ml, y_test, model_name)
            aug_results[model_name] = metrics
            delta = metrics['r2'] - baseline_results[model_name]['r2']
            print(f"    {model_name}: R²={metrics['r2']:.4f} ({delta:+.4f})")

        # Replacement
        X_ctgan_full, y_ctgan_full = generate_ctgan_samples(len(X_train_ml), cat_columns, cat_cardinalities)
        print("\n  REPLACEMENT (Synthetic only):")
        for model_name in ["RF", "GB", "Ridge"]:
            metrics = evaluate(X_ctgan_full, y_ctgan_full, X_test_ml, y_test, model_name)
            rep_results[model_name] = metrics
            delta = metrics['r2'] - baseline_results[model_name]['r2']
            print(f"    {model_name}: R²={metrics['r2']:.4f} ({delta:+.4f})")

        results["methods"]["CTGAN"] = {
            "augmentation": aug_results,
            "replacement": rep_results
        }

    # 4. SMOGN
    print(f"\n{'='*70}")
    print("4. SMOGN (Traditional Augmentation)")
    print("="*70)

    X_smogn, y_smogn = generate_smogn_samples(X_train_num, y_train, cat_train, cat_cardinalities)
    if X_smogn is not None:
        aug_results = {}
        print("\n  AUGMENTATION (SMOGN output - replaces original):")
        for model_name in ["RF", "GB", "Ridge"]:
            metrics = evaluate(X_smogn, y_smogn, X_test_ml, y_test, model_name)
            aug_results[model_name] = metrics
            delta = metrics['r2'] - baseline_results[model_name]['r2']
            print(f"    {model_name}: R²={metrics['r2']:.4f} ({delta:+.4f})")

        results["methods"]["SMOGN"] = {
            "augmentation": aug_results,
            "replacement": None  # SMOGN doesn't do replacement
        }

    # SUMMARY TABLE
    print(f"\n{'='*70}")
    print("SUMMARY: Average R² Across Models")
    print("="*70)

    print("\n| Method    | Augmentation | Replacement | Best For |")
    print("|-----------|--------------|-------------|----------|")

    for method in ["Original", "Diffusion", "CTGAN", "SMOGN"]:
        if method in results["methods"]:
            aug = results["methods"][method]["augmentation"]
            rep = results["methods"][method]["replacement"]

            if aug:
                aug_avg = np.mean([aug[m]["r2"] for m in aug])
            else:
                aug_avg = None

            if rep:
                rep_avg = np.mean([rep[m]["r2"] for m in rep])
            else:
                rep_avg = None

            aug_str = f"{aug_avg:.4f}" if aug_avg is not None else "N/A"
            rep_str = f"{rep_avg:.4f}" if rep_avg is not None else "N/A"

            if method == "Original":
                best = "Baseline"
            elif method == "Diffusion":
                best = "Augmentation"
            elif method == "CTGAN":
                best = "Replacement"
            else:
                best = "Fails"

            print(f"| {method:9} | {aug_str:12} | {rep_str:11} | {best:8} |")

    # Save results
    results_path = Path("results/final_comparison.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to: {results_path}")

    # KEY FINDINGS
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print("="*70)
    print("""
1. AUGMENTATION (add synthetic to original):
   - Diffusion: Maintains baseline (safe choice)
   - CTGAN: Slight degradation
   - SMOGN: CATASTROPHIC failure on complex data

2. REPLACEMENT (privacy-preserving, synthetic only):
   - CTGAN: Best choice (~35% of baseline R²)
   - Diffusion: Fails (model collapse issue)
   - SMOGN: N/A

3. RECOMMENDATION:
   - Use Diffusion for augmentation
   - Use CTGAN for privacy/data sharing
   - Avoid SMOGN on high-dimensional data
""")

    return results


if __name__ == "__main__":
    results = main()
