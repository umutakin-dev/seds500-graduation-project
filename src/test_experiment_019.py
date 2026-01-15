"""
Test Experiment 019 model for replacement and augmentation scenarios.

Evaluates TabDDPM-style diffusion on Production data.
"""

import pickle
import torch
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

from diffusion_tabddpm import HybridDiffusionTabDDPM
from models import MLPDenoiser


def load_production_data():
    """Load Production data and preprocessors."""
    data_path = Path("data/production/full_minmax.pt")
    data = torch.load(data_path, weights_only=False)
    with open(data_path.parent / "preprocessors_full.pkl", "rb") as f:
        preprocessors = pickle.load(f)
    return data, preprocessors


def load_experiment_019_model(device, checkpoint_dir="checkpoints/experiment_019_v2"):
    """Load trained experiment 019 model."""
    for name in ["best_model.pt", "final_model.pt"]:
        path = Path(f"{checkpoint_dir}/{name}")
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

            diffusion = HybridDiffusionTabDDPM(
                num_numerical=diffusion_config["num_numerical"],
                cat_cardinalities=diffusion_config["cat_cardinalities"],
                num_timesteps=diffusion_config["num_timesteps"],
                beta_schedule=diffusion_config["beta_schedule"],
            ).to(device)

            print(f"Loaded model from: {path}")
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A'):.4f}")
            return model, diffusion

    raise FileNotFoundError(f"No experiment 019 model found in {checkpoint_dir}.")


def onehot_encode(cat_idx, cardinalities):
    """Convert categorical indices to one-hot."""
    result = []
    for i, card in enumerate(cardinalities):
        result.append(np.eye(card)[cat_idx[:, i]])
    return np.hstack(result).astype(np.float32)


def onehot_to_indices(onehot, cardinalities):
    """Convert one-hot encoded categorical back to indices."""
    n_samples = onehot.shape[0]
    n_cats = len(cardinalities)
    indices = np.zeros((n_samples, n_cats), dtype=np.int64)

    offset = 0
    for i, card in enumerate(cardinalities):
        cat_onehot = onehot[:, offset:offset + card]
        indices[:, i] = cat_onehot.argmax(axis=1)
        offset += card

    return indices


def generate_samples(model, diffusion, n_samples, device, data, preprocessors):
    """Generate synthetic samples using experiment 019 model."""
    model.eval()
    num_numerical = diffusion.num_numerical
    cat_cardinalities = diffusion.cat_cardinalities
    scaler = preprocessors["scaler"]
    target_scaler = preprocessors["target_scaler"]

    with torch.no_grad():
        x_num, cat_indices = diffusion.sample(model, batch_size=n_samples, device=device)

    x_num = x_num.cpu().numpy()
    cat_indices = cat_indices.cpu().numpy()

    print(f"\nGenerated numeric stats (should match training [-1, 1]):")
    print(f"  Mean: {x_num.mean(axis=0)[:3].round(4)}...")
    print(f"  Std:  {x_num.std(axis=0)[:3].round(4)}...")

    at_boundary = (np.abs(x_num) > 0.95).mean(axis=0)
    print(f"  At boundary (>0.95): {at_boundary[:3].round(4)}...")

    # Inverse transform numerical features (excluding target)
    gen_features_scaled = x_num[:, :num_numerical-1]  # All but last (target)
    gen_target_scaled = x_num[:, num_numerical-1:num_numerical]  # Last column is target

    gen_features = scaler.inverse_transform(gen_features_scaled)
    gen_target = target_scaler.inverse_transform(gen_target_scaled).flatten()

    # Convert categorical indices to one-hot
    cat_onehot = onehot_encode(cat_indices, cat_cardinalities)

    X_synthetic = np.column_stack([gen_features, cat_onehot])
    y_synthetic = gen_target

    return X_synthetic, y_synthetic


def get_original_features(data, preprocessors):
    """Extract original (unscaled) features from Production data."""
    X_train = data["X_train"].numpy()
    X_test = data["X_test"].numpy()
    y_train = data["y_train"]
    y_test = data["y_test"]

    num_numerical = data["num_numerical"]
    cat_cardinalities = data["cat_cardinalities"]

    scaler = preprocessors["scaler"]
    target_scaler = preprocessors["target_scaler"]

    # Extract scaled numerical features (excluding target)
    X_train_num_scaled = X_train[:, :num_numerical-1]
    X_test_num_scaled = X_test[:, :num_numerical-1]

    # Inverse transform to get original numerical features
    X_train_num = scaler.inverse_transform(X_train_num_scaled)
    X_test_num = scaler.inverse_transform(X_test_num_scaled)

    # Extract categorical one-hot
    X_train_cat_onehot = X_train[:, num_numerical:]
    X_test_cat_onehot = X_test[:, num_numerical:]

    # Combine for ML
    X_train_ml = np.column_stack([X_train_num, X_train_cat_onehot])
    X_test_ml = np.column_stack([X_test_num, X_test_cat_onehot])

    return X_train_ml, X_test_ml, y_train, y_test, cat_cardinalities


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

    print("\nLoading Production data...")
    data, preprocessors = load_production_data()

    X_train_ml, X_test_ml, y_train, y_test, cat_cardinalities = get_original_features(data, preprocessors)

    print(f"\nData shapes: Train: {X_train_ml.shape}, Test: {X_test_ml.shape}")
    print(f"Target stats: Train mean={y_train.mean():.1f}, std={y_train.std():.1f}")

    # Baseline
    print(f"\n{'='*60}")
    print("BASELINE (Original data only)")
    print(f"{'='*60}")

    baseline = {}
    for model_name in ["RF", "GB", "Ridge"]:
        metrics = evaluate_model(X_train_ml, y_train, X_test_ml, y_test, model_name)
        baseline[model_name] = metrics
        print(f"  {model_name}: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.1f}")

    # Load experiment 019 model
    print(f"\n{'='*60}")
    print("EXPERIMENT 019 MODEL (TabDDPM-style on Production)")
    print(f"{'='*60}")

    model, diffusion = load_experiment_019_model(device)

    n_synthetic = len(X_train_ml)
    print(f"\nGenerating {n_synthetic} synthetic samples...")
    X_synthetic, y_synthetic = generate_samples(
        model, diffusion, n_synthetic, device, data, preprocessors
    )

    print(f"\nSynthetic target stats:")
    print(f"  Mean: {y_synthetic.mean():.1f} (original: {y_train.mean():.1f})")
    print(f"  Std:  {y_synthetic.std():.1f} (original: {y_train.std():.1f})")
    print(f"  Min:  {y_synthetic.min():.1f} (original: {y_train.min():.1f})")
    print(f"  Max:  {y_synthetic.max():.1f} (original: {y_train.max():.1f})")

    # Test: REPLACEMENT
    print(f"\n{'='*60}")
    print("TEST: REPLACEMENT (Synthetic only)")
    print(f"{'='*60}")

    replacement = {}
    for model_name in ["RF", "GB", "Ridge"]:
        metrics = evaluate_model(X_synthetic, y_synthetic, X_test_ml, y_test, model_name)
        replacement[model_name] = metrics
        delta = metrics['r2'] - baseline[model_name]['r2']
        print(f"  {model_name}: R2={metrics['r2']:.4f} (delta: {delta:+.4f})")

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
        print(f"  {model_name}: R2={metrics['r2']:.4f} (delta: {delta:+.4f})")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Experiment 019 Results (Production Data)")
    print(f"{'='*60}")

    rf_baseline = baseline["RF"]["r2"]
    rf_replacement = replacement["RF"]["r2"]
    rf_augmentation = augmentation["RF"]["r2"]

    print("\n| Scenario      | RF R2   | vs Baseline | % of Baseline |")
    print("|---------------|---------|-------------|---------------|")
    print(f"| Baseline      | {rf_baseline:.4f}  | -           | 100%          |")
    print(f"| Augmentation  | {rf_augmentation:.4f}  | {rf_augmentation - rf_baseline:+.4f}      | {rf_augmentation/rf_baseline*100:.1f}%         |")
    print(f"| Replacement   | {rf_replacement:.4f}  | {rf_replacement - rf_baseline:+.4f}      | {max(0, rf_replacement/rf_baseline*100):.1f}%         |")

    print("\n--- Comparison with Ozel Rich (Experiment 018) ---")
    print("| Dataset       | Baseline R2 | Replacement % | Augmentation % |")
    print("|---------------|-------------|---------------|----------------|")
    print(f"| Ozel Rich     | 0.6451      | 87.3%         | 99.1%          |")
    print(f"| Production    | {rf_baseline:.4f}      | {max(0, rf_replacement/rf_baseline*100):.1f}%         | {rf_augmentation/rf_baseline*100:.1f}%          |")


if __name__ == "__main__":
    main()
