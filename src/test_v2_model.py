"""
Test the v2 diffusion model for synthetic data generation.

Runs on CPU to avoid CUDA multinomial issues.
"""

import pickle
import torch
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from diffusion import HybridDiffusion
from models import MLPDenoiser


def load_data():
    data = torch.load("data/ozel_rich/prepared.pt", weights_only=False)
    with open("data/ozel_rich/preprocessors.pkl", "rb") as f:
        preprocessors = pickle.load(f)
    return data, preprocessors


def load_v2_model(device):
    checkpoint = torch.load(
        "checkpoints/ozel_rich_v2/final_model.pt",
        map_location=device,
        weights_only=False
    )

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

    return model, diffusion


def onehot_encode(cat_idx, cardinalities):
    result = []
    for i, card in enumerate(cardinalities):
        result.append(np.eye(card)[cat_idx[:, i]])
    return np.hstack(result).astype(np.float32)


def main():
    device = "cpu"  # Use CPU to avoid CUDA multinomial issues
    print(f"Using device: {device}")
    print("(Using CPU to avoid CUDA numerical issues with multinomial)")

    # Load data
    data, preprocessors = load_data()

    X_train_num = data["X_train_num_unscaled"]
    X_test_num = data["X_test_num_unscaled"]
    cat_train = data["cat_train"]
    cat_test = data["cat_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    cat_cardinalities = data["cat_cardinalities"]

    cat_train_onehot = onehot_encode(cat_train, cat_cardinalities)
    cat_test_onehot = onehot_encode(cat_test, cat_cardinalities)

    X_train_ml = np.column_stack([X_train_num, cat_train_onehot])
    X_test_ml = np.column_stack([X_test_num, cat_test_onehot])

    print(f"\nData: {len(X_train_ml)} train, {len(X_test_ml)} test")

    # Baseline
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_ml, y_train)
    baseline_r2 = r2_score(y_test, rf.predict(X_test_ml))
    print(f"\nBaseline (Original only): R² = {baseline_r2:.4f}")

    # Load v2 model
    print("\nLoading v2 diffusion model...")
    model, diffusion = load_v2_model(device)

    # Generate synthetic samples
    n_synthetic = len(X_train_ml)
    print(f"Generating {n_synthetic} synthetic samples (this may take a few minutes on CPU)...")

    with torch.no_grad():
        samples = diffusion.sample(
            model,
            batch_size=n_synthetic,
            device=device,
            clip_denoised=True
        )
    samples = samples.numpy()

    # Process samples
    scaler = preprocessors["scaler"]
    target_scaler = preprocessors["target_scaler"]
    num_numerical = data["num_numerical"]

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

    X_synthetic = np.column_stack([features_unscaled, cat_onehot])
    y_synthetic = target_unscaled

    # Check synthetic data quality
    print(f"\nSynthetic data stats:")
    print(f"  Target - mean: {y_synthetic.mean():.1f}, std: {y_synthetic.std():.1f}")
    print(f"  Original target - mean: {y_train.mean():.1f}, std: {y_train.std():.1f}")

    # Test 1: AUGMENTATION (Original + Synthetic)
    print(f"\n{'='*60}")
    print("TEST 1: AUGMENTATION (Original + Synthetic)")
    print(f"{'='*60}")

    X_aug = np.vstack([X_train_ml, X_synthetic])
    y_aug = np.concatenate([y_train, y_synthetic])

    rf_aug = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_aug.fit(X_aug, y_aug)
    aug_r2 = r2_score(y_test, rf_aug.predict(X_test_ml))

    print(f"  R² = {aug_r2:.4f} (baseline: {baseline_r2:.4f}, delta: {aug_r2 - baseline_r2:+.4f})")

    # Test 2: REPLACEMENT (Synthetic only)
    print(f"\n{'='*60}")
    print("TEST 2: REPLACEMENT (Synthetic only)")
    print(f"{'='*60}")

    rf_syn = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_syn.fit(X_synthetic, y_synthetic)
    syn_r2 = r2_score(y_test, rf_syn.predict(X_test_ml))

    print(f"  R² = {syn_r2:.4f} (baseline: {baseline_r2:.4f}, delta: {syn_r2 - baseline_r2:+.4f})")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: V2 Model Results")
    print(f"{'='*60}")
    print(f"\n| Scenario      | R²     | vs Baseline |")
    print(f"|---------------|--------|-------------|")
    print(f"| Baseline      | {baseline_r2:.4f} | -           |")
    print(f"| Augmentation  | {aug_r2:.4f} | {aug_r2 - baseline_r2:+.4f}      |")
    print(f"| Replacement   | {syn_r2:.4f} | {syn_r2 - baseline_r2:+.4f}      |")

    if syn_r2 > 0:
        print(f"\n SUCCESS: V2 model produces USABLE synthetic data for replacement!")
        print(f"          Models trained on synthetic data achieve {syn_r2/baseline_r2*100:.1f}% of original R²")
    else:
        print(f"\n ISSUE: V2 model still has problems with replacement scenario")


if __name__ == "__main__":
    main()
