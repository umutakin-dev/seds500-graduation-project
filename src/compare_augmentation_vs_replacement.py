"""
Compare synthetic data methods in two scenarios:
1. AUGMENTATION: Original + Synthetic (add synthetic to original)
2. REPLACEMENT: Synthetic only (replace original entirely)

This reveals that different methods excel in different use cases.

Usage:
    python src/compare_augmentation_vs_replacement.py --device cuda
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
    data = torch.load(data_path, weights_only=False)
    return data


def load_preprocessors(preprocessor_path="data/ozel_rich/preprocessors.pkl"):
    """Load preprocessors for inverse transform."""
    with open(preprocessor_path, "rb") as f:
        return pickle.load(f)


def onehot_encode(cat_idx, cardinalities):
    result = []
    for i, card in enumerate(cardinalities):
        result.append(np.eye(card)[cat_idx[:, i]])
    return np.hstack(result).astype(np.float32)


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


def generate_diffusion_augmentation(model, diffusion, X_train_num, cat_train, y_train,
                                     n_samples, device, preprocessors, data):
    """
    Generate diffusion samples for AUGMENTATION.
    Uses the approach from evaluate_ozel_rich.py that worked.
    """
    cat_cardinalities = data["cat_cardinalities"]

    with torch.no_grad():
        samples = diffusion.sample(model, batch_size=n_samples, device=device, clip_denoised=True)
    samples = samples.cpu().numpy()

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

    # AUGMENTATION: combine with original
    cat_train_onehot = onehot_encode(cat_train, cat_cardinalities)
    X_original = np.column_stack([X_train_num, cat_train_onehot])

    X_aug = np.vstack([X_original, X_synthetic])
    y_aug = np.concatenate([y_train, y_synthetic])

    return X_aug, y_aug, X_synthetic, y_synthetic


def generate_ctgan_data(n_samples, cat_columns, cat_cardinalities):
    """Load CTGAN model and generate samples."""
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


def generate_smogn_data(X_train_num, y_train, cat_train, cat_cardinalities):
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


def evaluate(X_train, y_train, X_test, y_test):
    """Evaluate with Random Forest."""
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return r2_score(y_test, y_pred)


def main(args):
    # Load data
    data = load_data(args.data_path)
    preprocessors = load_preprocessors()

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

    n_synthetic = 500  # Same as original experiment

    print(f"{'='*70}")
    print("COMPARING AUGMENTATION VS REPLACEMENT SCENARIOS")
    print(f"{'='*70}")
    print(f"\nOriginal train size: {len(X_train_ml)}")
    print(f"Synthetic samples: {n_synthetic}")

    # Baseline
    baseline_r2 = evaluate(X_train_ml, y_train, X_test_ml, y_test)
    print(f"\nBaseline (Original only): R² = {baseline_r2:.4f}")

    results = {"Original": {"augmentation": baseline_r2, "replacement": baseline_r2}}

    # 1. DIFFUSION
    print(f"\n--- DIFFUSION ---")
    checkpoint_path = Path("checkpoints/ozel_rich/final_model.pt")
    if checkpoint_path.exists():
        model, diffusion, _ = load_diffusion_model(checkpoint_path, args.device)

        X_aug, y_aug, X_syn, y_syn = generate_diffusion_augmentation(
            model, diffusion, X_train_num, cat_train, y_train,
            n_synthetic, args.device, preprocessors, data
        )

        aug_r2 = evaluate(X_aug, y_aug, X_test_ml, y_test)
        rep_r2 = evaluate(X_syn, y_syn, X_test_ml, y_test)

        print(f"  Augmentation (Original + {n_synthetic} synthetic): R² = {aug_r2:.4f} ({aug_r2 - baseline_r2:+.4f})")
        print(f"  Replacement ({len(X_syn)} synthetic only): R² = {rep_r2:.4f} ({rep_r2 - baseline_r2:+.4f})")

        results["Diffusion"] = {"augmentation": aug_r2, "replacement": rep_r2}

    # 2. CTGAN
    print(f"\n--- CTGAN ---")
    X_ctgan, y_ctgan = generate_ctgan_data(n_synthetic, cat_columns, cat_cardinalities)
    if X_ctgan is not None:
        # Augmentation
        X_aug = np.vstack([X_train_ml, X_ctgan])
        y_aug = np.concatenate([y_train, y_ctgan])
        aug_r2 = evaluate(X_aug, y_aug, X_test_ml, y_test)

        # Replacement (generate more for fair comparison)
        X_rep, y_rep = generate_ctgan_data(len(X_train_ml), cat_columns, cat_cardinalities)
        rep_r2 = evaluate(X_rep, y_rep, X_test_ml, y_test)

        print(f"  Augmentation (Original + {n_synthetic} synthetic): R² = {aug_r2:.4f} ({aug_r2 - baseline_r2:+.4f})")
        print(f"  Replacement ({len(X_rep)} synthetic only): R² = {rep_r2:.4f} ({rep_r2 - baseline_r2:+.4f})")

        results["CTGAN"] = {"augmentation": aug_r2, "replacement": rep_r2}

    # 3. SMOGN
    print(f"\n--- SMOGN ---")
    X_smogn, y_smogn = generate_smogn_data(X_train_num, y_train, cat_train, cat_cardinalities)
    if X_smogn is not None:
        # SMOGN generates augmented data (not replacement)
        aug_r2 = evaluate(X_smogn, y_smogn, X_test_ml, y_test)
        print(f"  Augmentation (SMOGN output): R² = {aug_r2:.4f} ({aug_r2 - baseline_r2:+.4f})")
        print(f"  Replacement: N/A (SMOGN generates augmented data, not new samples)")

        results["SMOGN"] = {"augmentation": aug_r2, "replacement": None}

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"\n| Method    | Augmentation R² | Replacement R² |")
    print(f"|-----------|-----------------|----------------|")
    print(f"| Original  | {results['Original']['augmentation']:.4f} (baseline) | {results['Original']['replacement']:.4f} (baseline) |")

    for method in ["Diffusion", "CTGAN", "SMOGN"]:
        if method in results:
            aug = results[method]["augmentation"]
            rep = results[method]["replacement"]
            aug_str = f"{aug:.4f}" if aug is not None else "N/A"
            rep_str = f"{rep:.4f}" if rep is not None else "N/A"
            print(f"| {method:9} | {aug_str:15} | {rep_str:14} |")

    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    print("""
USE CASE 1: Data Augmentation (add synthetic to existing data)
  - Goal: Improve model by adding more training examples
  - Best method: Check which has highest augmentation R²

USE CASE 2: Data Replacement (share synthetic instead of real)
  - Goal: Privacy - share data without revealing real records
  - Best method: Check which has highest replacement R²
  - Key: Models trained on synthetic should work on real data
""")

    # Recommendations
    if "CTGAN" in results and "Diffusion" in results:
        if results["CTGAN"]["replacement"] > results["Diffusion"]["replacement"]:
            print("RECOMMENDATION for Privacy/Replacement: CTGAN")
            print("  - CTGAN generates more realistic standalone synthetic data")
        else:
            print("RECOMMENDATION for Privacy/Replacement: Diffusion")

        diff_aug = results["Diffusion"]["augmentation"]
        ctgan_aug = results["CTGAN"]["augmentation"]
        if diff_aug > ctgan_aug:
            print("\nRECOMMENDATION for Augmentation: Diffusion")
        else:
            print("\nRECOMMENDATION for Augmentation: CTGAN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/ozel_rich/prepared.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    main(args)
