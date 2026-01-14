"""
Test V5 model for replacement scenario.

V5 uses MinMax scaling to [-1, 1]:
- No distribution mismatch
- Linear inverse transform
- Should generate valid samples without boundary collapse
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


def load_v5_data():
    """Load v5 data and preprocessors."""
    data_path = Path("data/ozel_rich_v5/prepared.pt")
    data = torch.load(data_path, weights_only=False)
    with open(data_path.parent / "preprocessors.pkl", "rb") as f:
        preprocessors = pickle.load(f)
    return data, preprocessors


def load_v5_model(device):
    """Load trained v5 model."""
    for name in ["best_model.pt", "final_model.pt"]:
        path = Path(f"checkpoints/ozel_rich_v5/{name}")
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

    raise FileNotFoundError("No v5 model found. Run train_ozel_rich_v5.py first.")


def onehot_encode(cat_idx, cardinalities):
    """Convert categorical indices to one-hot."""
    result = []
    for i, card in enumerate(cardinalities):
        result.append(np.eye(card)[cat_idx[:, i]])
    return np.hstack(result).astype(np.float32)


def generate_v5_samples(model, diffusion, n_samples, device, data, preprocessors):
    """Generate synthetic samples using v5 model (MinMax scaling)."""
    model.eval()
    num_numerical = data["num_numerical"]
    cat_cardinalities = data["cat_cardinalities"]
    scaler = preprocessors["scaler"]
    target_scaler = preprocessors["target_scaler"]

    with torch.no_grad():
        samples = diffusion.sample(model, batch_size=n_samples, device=device, clip_denoised=True)

    samples = samples.cpu().numpy()

    gen_num = samples[:, :num_numerical]
    gen_cat_probs = samples[:, num_numerical:]

    print(f"\nGenerated numeric stats (should match training [-1, 1]):")
    print(f"  Mean: {gen_num.mean(axis=0)}")
    print(f"  Std:  {gen_num.std(axis=0)}")
    print(f"  Min:  {gen_num.min(axis=0)}")
    print(f"  Max:  {gen_num.max(axis=0)}")

    at_boundary = (np.abs(gen_num) > 0.95).mean(axis=0)
    print(f"  At boundary (>0.95): {at_boundary}")

    # Inverse transform using MinMax scaler (simple linear)
    gen_features_scaled = gen_num[:, :2]  # Cap, Boy
    gen_target_scaled = gen_num[:, 2:3]  # Target

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
    device = "cpu"
    print(f"Using device: {device}")

    print("\nLoading v5 data (MinMax scaled)...")
    data, preprocessors = load_v5_data()

    X_train_num = data["X_train_num_unscaled"]
    X_test_num = data["X_test_num_unscaled"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    cat_train = data["cat_train"]
    cat_test = data["cat_test"]
    cat_cardinalities = data["cat_cardinalities"]

    cat_train_onehot = onehot_encode(cat_train, cat_cardinalities)
    cat_test_onehot = onehot_encode(cat_test, cat_cardinalities)

    X_train_ml = np.column_stack([X_train_num, cat_train_onehot])
    X_test_ml = np.column_stack([X_test_num, cat_test_onehot])

    print(f"\nData shapes: Train: {X_train_ml.shape}, Test: {X_test_ml.shape}")

    # Baseline
    print(f"\n{'='*60}")
    print("BASELINE (Original data only)")
    print(f"{'='*60}")

    baseline = {}
    for model_name in ["RF", "GB", "Ridge"]:
        metrics = evaluate_model(X_train_ml, y_train, X_test_ml, y_test, model_name)
        baseline[model_name] = metrics
        print(f"  {model_name}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.1f}")

    # Load v5 model
    print(f"\n{'='*60}")
    print("V5 MODEL (MinMax scaling)")
    print(f"{'='*60}")

    model, diffusion = load_v5_model(device)

    n_synthetic = len(X_train_ml)
    print(f"\nGenerating {n_synthetic} synthetic samples...")
    X_synthetic, y_synthetic = generate_v5_samples(
        model, diffusion, n_synthetic, device, data, preprocessors
    )

    print(f"\nSynthetic target stats:")
    print(f"  Mean: {y_synthetic.mean():.1f} (original: {y_train.mean():.1f})")
    print(f"  Std:  {y_synthetic.std():.1f} (original: {y_train.std():.1f})")

    # Test: REPLACEMENT
    print(f"\n{'='*60}")
    print("TEST: REPLACEMENT (Synthetic only)")
    print(f"{'='*60}")

    replacement = {}
    for model_name in ["RF", "GB", "Ridge"]:
        metrics = evaluate_model(X_synthetic, y_synthetic, X_test_ml, y_test, model_name)
        replacement[model_name] = metrics
        delta = metrics['r2'] - baseline[model_name]['r2']
        print(f"  {model_name}: R²={metrics['r2']:.4f} (delta: {delta:+.4f})")

    # Test: AUGMENTATION
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
    print("SUMMARY: V5 Results")
    print(f"{'='*60}")

    rf_baseline = baseline["RF"]["r2"]
    rf_replacement = replacement["RF"]["r2"]
    rf_augmentation = augmentation["RF"]["r2"]

    print("\n| Scenario      | RF R²   | vs Baseline |")
    print("|---------------|---------|-------------|")
    print(f"| Baseline      | {rf_baseline:.4f}  | -           |")
    print(f"| Augmentation  | {rf_augmentation:.4f}  | {rf_augmentation - rf_baseline:+.4f}      |")
    print(f"| Replacement   | {rf_replacement:.4f}  | {rf_replacement - rf_baseline:+.4f}      |")

    if rf_replacement > 0:
        print(f"\n SUCCESS! V5 works for replacement.")
        print(f"          Achieves {rf_replacement/rf_baseline*100:.1f}% of baseline")
    else:
        print(f"\n V5 still has issues with replacement.")


if __name__ == "__main__":
    main()
