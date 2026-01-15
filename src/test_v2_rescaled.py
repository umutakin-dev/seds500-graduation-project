"""
Test v2 model with post-hoc rescaling of generated samples.

The issue: Training data has std ~0.32, but diffusion generates std ~0.88
Fix: Rescale generated samples to match training distribution.
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
    device = "cpu"
    print(f"Using device: {device}")

    # Load data
    data, preprocessors = load_data()

    X_train_num = data["X_train_num_unscaled"]
    X_test_num = data["X_test_num_unscaled"]
    cat_train = data["cat_train"]
    cat_test = data["cat_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    cat_cardinalities = data["cat_cardinalities"]

    # Get training data stats for rescaling
    X_train_scaled = data["X_train"][:, :3].numpy()
    train_mean = X_train_scaled.mean(axis=0)
    train_std = X_train_scaled.std(axis=0)
    print(f"Training data: mean={train_mean}, std={train_std}")

    cat_train_onehot = onehot_encode(cat_train, cat_cardinalities)
    cat_test_onehot = onehot_encode(cat_test, cat_cardinalities)

    X_train_ml = np.column_stack([X_train_num, cat_train_onehot])
    X_test_ml = np.column_stack([X_test_num, cat_test_onehot])

    # Baseline
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_ml, y_train)
    baseline_r2 = r2_score(y_test, rf.predict(X_test_ml))
    print(f"\nBaseline (Original only): R² = {baseline_r2:.4f}")

    # Load model and generate
    print("\nGenerating synthetic samples...")
    model, diffusion = load_v2_model(device)

    n_synthetic = len(X_train_ml)
    with torch.no_grad():
        samples = diffusion.sample(model, batch_size=n_synthetic, device=device, clip_denoised=True)
    samples = samples.numpy()

    # Get raw generated stats
    gen_num = samples[:, :3]
    gen_mean = gen_num.mean(axis=0)
    gen_std = gen_num.std(axis=0)
    print(f"Generated (raw): mean={gen_mean}, std={gen_std}")

    # RESCALE: Transform generated samples to match training distribution
    # Formula: x_rescaled = (x - gen_mean) / gen_std * train_std + train_mean
    gen_num_rescaled = (gen_num - gen_mean) / (gen_std + 1e-8) * train_std + train_mean
    gen_num_rescaled = np.clip(gen_num_rescaled, -1, 1)

    rescaled_mean = gen_num_rescaled.mean(axis=0)
    rescaled_std = gen_num_rescaled.std(axis=0)
    print(f"Generated (rescaled): mean={rescaled_mean}, std={rescaled_std}")

    # Process samples with rescaled numeric
    scaler = preprocessors["scaler"]
    target_scaler = preprocessors["target_scaler"]
    num_numerical = data["num_numerical"]
    cat_probs = samples[:, num_numerical:]

    # Unscale numeric (use rescaled values)
    num_unclipped = gen_num_rescaled * 3
    features_unscaled = scaler.inverse_transform(num_unclipped[:, :2])
    target_unscaled = target_scaler.inverse_transform(num_unclipped[:, 2:3]).flatten()

    # Convert categorical
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

    print(f"\nSynthetic target: mean={y_synthetic.mean():.1f}, std={y_synthetic.std():.1f}")
    print(f"Original target:  mean={y_train.mean():.1f}, std={y_train.std():.1f}")

    # Test REPLACEMENT
    print(f"\n{'='*60}")
    print("TEST: REPLACEMENT (Synthetic only) - WITH RESCALING")
    print(f"{'='*60}")

    rf_syn = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_syn.fit(X_synthetic, y_synthetic)
    syn_r2 = r2_score(y_test, rf_syn.predict(X_test_ml))

    print(f"  R² = {syn_r2:.4f} (baseline: {baseline_r2:.4f}, delta: {syn_r2 - baseline_r2:+.4f})")

    if syn_r2 > 0:
        print(f"\n  SUCCESS! Rescaling helps. Achieves {syn_r2/baseline_r2*100:.1f}% of baseline.")
    else:
        print(f"\n  Still not working. Rescaling alone doesn't fix the issue.")


if __name__ == "__main__":
    main()
