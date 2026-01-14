"""
Test V3 model for replacement scenario.

This tests the TabDDPM-style preprocessing fix:
- Unbounded Gaussian (no /3 clipping)
- clip_denoised=False during sampling
- Proper inverse transform

Expected: R² > 0 for replacement (vs -14 for v1/v2)
"""

import pickle
import torch
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

from diffusion import HybridDiffusion
from models import MLPDenoiser


def load_v3_data():
    """Load v3 data and preprocessors."""
    data_path = Path("data/ozel_rich_v3/prepared.pt")
    data = torch.load(data_path, weights_only=False)

    with open(data_path.parent / "preprocessors.pkl", "rb") as f:
        preprocessors = pickle.load(f)

    return data, preprocessors


def load_v3_model(device):
    """Load trained v3 model."""
    # Try best model first, then final
    for name in ["best_model.pt", "final_model.pt"]:
        path = Path(f"checkpoints/ozel_rich_v3/{name}")
        if path.exists():
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

            print(f"Loaded model from: {path}")
            return model, diffusion

    raise FileNotFoundError("No v3 model found. Run train_ozel_rich_v3.py first.")


def onehot_encode(cat_idx, cardinalities):
    """Convert categorical indices to one-hot."""
    result = []
    for i, card in enumerate(cardinalities):
        result.append(np.eye(card)[cat_idx[:, i]])
    return np.hstack(result).astype(np.float32)


def generate_v3_samples(model, diffusion, n_samples, device, data, preprocessors):
    """
    Generate synthetic samples using v3 model.

    Key differences from v1/v2:
    - clip_denoised=False (no hard clipping)
    - Direct inverse_transform (no *3 scaling)
    """
    model.eval()
    num_numerical = data["num_numerical"]
    cat_cardinalities = data["cat_cardinalities"]
    scaler = preprocessors["scaler"]
    target_scaler = preprocessors["target_scaler"]

    with torch.no_grad():
        # Generate with NO clipping (key fix!)
        samples = diffusion.sample(
            model,
            batch_size=n_samples,
            device=device,
            clip_denoised=False
        )

    samples = samples.cpu().numpy()

    # Split numeric and categorical
    gen_num = samples[:, :num_numerical]  # [Cap, Boy, Target] in Gaussian space
    gen_cat_probs = samples[:, num_numerical:]

    # Print generation stats
    print(f"\nGenerated numeric stats (should match training ~N(0,1)):")
    print(f"  Mean: {gen_num.mean(axis=0)}")
    print(f"  Std:  {gen_num.std(axis=0)}")
    print(f"  Min:  {gen_num.min(axis=0)}")
    print(f"  Max:  {gen_num.max(axis=0)}")

    # Check for boundary collapse
    at_boundary = (np.abs(gen_num) > 2.5).mean(axis=0)
    print(f"  At boundary (>2.5σ): {at_boundary}")

    # Inverse transform numeric features (NO *3 scaling!)
    gen_features_scaled = gen_num[:, :2]  # Cap, Boy
    gen_target_scaled = gen_num[:, 2:3]  # Target

    # Direct inverse transform (TabDDPM style)
    gen_features = scaler.inverse_transform(gen_features_scaled)
    gen_target = target_scaler.inverse_transform(gen_target_scaled).flatten()

    # Convert categorical probabilities to one-hot
    offset = 0
    cat_onehot_list = []
    for card in cat_cardinalities:
        probs = gen_cat_probs[:, offset:offset + card]
        indices = probs.argmax(axis=1)
        cat_onehot_list.append(np.eye(card)[indices])
        offset += card
    cat_onehot = np.hstack(cat_onehot_list).astype(np.float32)

    # Combine features for ML
    X_synthetic = np.column_stack([gen_features, cat_onehot])
    y_synthetic = gen_target

    return X_synthetic, y_synthetic


def evaluate_model(X_train, y_train, X_test, y_test, model_name="RF"):
    """Train and evaluate a model."""
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
    device = "cpu"  # Use CPU to avoid CUDA multinomial issues
    print(f"Using device: {device}")

    # Load data
    print("\nLoading v3 data...")
    data, preprocessors = load_v3_data()

    X_train_num = data["X_train_num_unscaled"]  # Original features
    X_test_num = data["X_test_num_unscaled"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    cat_train = data["cat_train"]
    cat_test = data["cat_test"]
    cat_cardinalities = data["cat_cardinalities"]

    # Prepare ML data
    cat_train_onehot = onehot_encode(cat_train, cat_cardinalities)
    cat_test_onehot = onehot_encode(cat_test, cat_cardinalities)

    X_train_ml = np.column_stack([X_train_num, cat_train_onehot])
    X_test_ml = np.column_stack([X_test_num, cat_test_onehot])

    print(f"\nData shapes:")
    print(f"  Train: {X_train_ml.shape}")
    print(f"  Test: {X_test_ml.shape}")

    # Baseline
    print(f"\n{'='*60}")
    print("BASELINE (Original data only)")
    print(f"{'='*60}")

    baseline = {}
    for model_name in ["RF", "GB", "Ridge"]:
        metrics = evaluate_model(X_train_ml, y_train, X_test_ml, y_test, model_name)
        baseline[model_name] = metrics
        print(f"  {model_name}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.1f}")

    # Load v3 model
    print(f"\n{'='*60}")
    print("V3 MODEL (TabDDPM-style, unbounded)")
    print(f"{'='*60}")

    model, diffusion = load_v3_model(device)

    # Generate synthetic data
    n_synthetic = len(X_train_ml)
    print(f"\nGenerating {n_synthetic} synthetic samples...")
    X_synthetic, y_synthetic = generate_v3_samples(
        model, diffusion, n_synthetic, device, data, preprocessors
    )

    print(f"\nSynthetic target stats:")
    print(f"  Mean: {y_synthetic.mean():.1f} (original: {y_train.mean():.1f})")
    print(f"  Std:  {y_synthetic.std():.1f} (original: {y_train.std():.1f})")

    # Test: REPLACEMENT (synthetic only)
    print(f"\n{'='*60}")
    print("TEST: REPLACEMENT (Synthetic only)")
    print(f"{'='*60}")

    replacement = {}
    for model_name in ["RF", "GB", "Ridge"]:
        metrics = evaluate_model(X_synthetic, y_synthetic, X_test_ml, y_test, model_name)
        replacement[model_name] = metrics
        delta = metrics['r2'] - baseline[model_name]['r2']
        print(f"  {model_name}: R²={metrics['r2']:.4f} (delta: {delta:+.4f})")

    # Test: AUGMENTATION (original + synthetic)
    print(f"\n{'='*60}")
    print("TEST: AUGMENTATION (Original + Synthetic)")
    print(f"{'='*60}")

    X_aug = np.vstack([X_train_ml, X_synthetic])
    y_aug = np.concatenate([y_train, y_synthetic])

    augmentation = {}
    for model_name in ["RF", "GB", "Ridge"]:
        metrics = evaluate_model(X_aug, y_aug, X_test_ml, y_test, model_name)
        augmentation[model_name] = metrics
        delta = metrics['r2'] - baseline[model_name]['r2']
        print(f"  {model_name}: R²={metrics['r2']:.4f} (delta: {delta:+.4f})")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: V3 vs Previous Versions")
    print(f"{'='*60}")

    print("\n| Scenario      | RF R²   | vs Baseline | vs V2     |")
    print("|---------------|---------|-------------|-----------|")

    rf_baseline = baseline["RF"]["r2"]
    rf_replacement = replacement["RF"]["r2"]
    rf_augmentation = augmentation["RF"]["r2"]

    # V2 replacement was around -14 (model collapse)
    v2_replacement = -14.0

    print(f"| Baseline      | {rf_baseline:.4f}  | -           | -         |")
    print(f"| Augmentation  | {rf_augmentation:.4f}  | {rf_augmentation - rf_baseline:+.4f}      | -         |")
    print(f"| Replacement   | {rf_replacement:.4f}  | {rf_replacement - rf_baseline:+.4f}      | {rf_replacement - v2_replacement:+.4f}    |")

    if rf_replacement > 0:
        print(f"\n SUCCESS! V3 fixes the model collapse issue.")
        print(f"          Replacement R² improved from {v2_replacement:.2f} to {rf_replacement:.4f}")
        print(f"          Achieves {rf_replacement/rf_baseline*100:.1f}% of baseline performance")
    elif rf_replacement > v2_replacement:
        print(f"\n PARTIAL SUCCESS: Improvement from {v2_replacement:.2f} to {rf_replacement:.4f}")
        print("          But still not usable for replacement.")
    else:
        print(f"\n NO IMPROVEMENT: Still at {rf_replacement:.4f}")


if __name__ == "__main__":
    main()
