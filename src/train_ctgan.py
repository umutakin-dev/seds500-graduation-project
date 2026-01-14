"""
Train CTGAN on Ozel Rich dataset for baseline comparison.

CTGAN (Conditional Tabular GAN) is a popular GAN-based method for synthetic
tabular data generation. We use it as a baseline to compare against diffusion.

This is Experiment 017.

Usage:
    python src/train_ctgan.py --epochs 300
"""

import argparse
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ctgan import CTGAN

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


def load_data(data_path="data/ozel_rich/prepared.pt"):
    """Load prepared ozel rich data."""
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    return data


def load_preprocessors(preprocessor_path="data/ozel_rich/preprocessors.pkl"):
    """Load preprocessors for inverse transform."""
    with open(preprocessor_path, "rb") as f:
        return pickle.load(f)


def prepare_dataframe(X_num, cat_idx, y, cat_columns, cat_cardinalities):
    """Convert numpy arrays to DataFrame for CTGAN."""
    # Create DataFrame with numeric features
    df = pd.DataFrame(X_num, columns=["Cap", "Boy"])

    # Add categorical columns (as strings for CTGAN)
    for i, col_name in enumerate(cat_columns):
        df[col_name] = cat_idx[:, i].astype(str)

    # Add target
    df["MakineSure"] = y

    return df


def evaluate_regressor(name, model, X_train, y_train, X_test, y_test):
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

    print(f"\nData summary:")
    print(f"  Training samples: {len(X_train_num)}")
    print(f"  Test samples: {len(X_test_num)}")
    print(f"  Numeric features: Cap, Boy")
    print(f"  Categorical features: {cat_columns}")
    print(f"  Categorical cardinalities: {cat_cardinalities}")

    # Prepare DataFrame for CTGAN
    train_df = prepare_dataframe(
        X_train_num, cat_train, y_train, cat_columns, cat_cardinalities
    )

    print(f"\nTraining DataFrame shape: {train_df.shape}")
    print(f"Columns: {list(train_df.columns)}")

    # Define discrete columns for CTGAN
    discrete_columns = list(cat_columns)

    print(f"\nDiscrete columns: {discrete_columns}")

    # Initialize and train CTGAN
    print(f"\n{'='*70}")
    print(f"TRAINING CTGAN")
    print(f"{'='*70}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    ctgan = CTGAN(
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
    )

    ctgan.fit(train_df, discrete_columns=discrete_columns)

    # Save model
    checkpoint_dir = Path("checkpoints/ctgan")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ctgan.save(checkpoint_dir / "ozel_rich_model.pkl")
    print(f"\nModel saved to {checkpoint_dir / 'ozel_rich_model.pkl'}")

    # Generate synthetic samples
    n_synthetic = len(train_df)
    print(f"\n{'='*70}")
    print(f"GENERATING {n_synthetic} SYNTHETIC SAMPLES")
    print(f"{'='*70}")

    synthetic_df = ctgan.sample(n_synthetic)

    print(f"Synthetic DataFrame shape: {synthetic_df.shape}")
    print(f"\nSynthetic data sample:")
    print(synthetic_df.head())

    # Save synthetic data
    synthetic_df.to_csv(checkpoint_dir / "synthetic_ozel_rich.csv", index=False)
    print(f"\nSynthetic data saved to {checkpoint_dir / 'synthetic_ozel_rich.csv'}")

    # Evaluate: Train on synthetic, test on real
    print(f"\n{'='*70}")
    print(f"EVALUATION: Train on Synthetic, Test on Real")
    print(f"{'='*70}")

    # Prepare features for ML
    def prepare_ml_features(df, cat_columns, cat_cardinalities):
        """Convert DataFrame to ML features (numeric + one-hot)."""
        X_num = df[["Cap", "Boy"]].values.astype(np.float32)
        y = df["MakineSure"].values.astype(np.float32)

        # One-hot encode categoricals
        onehot_list = []
        for col_name in cat_columns:
            # Get unique values and create mapping
            unique_vals = sorted(df[col_name].unique())
            val_to_idx = {v: i for i, v in enumerate(unique_vals)}
            indices = df[col_name].map(val_to_idx).values

            # Find the cardinality from training data
            col_idx = list(cat_columns).index(col_name)
            card = cat_cardinalities[col_idx]

            # One-hot encode (handle potential out-of-range indices)
            indices = np.clip(indices, 0, card - 1)
            onehot_list.append(np.eye(card)[indices])

        cat_onehot = np.hstack(onehot_list).astype(np.float32)
        X = np.column_stack([X_num, cat_onehot])

        return X, y

    # Prepare training features from original data
    def onehot_encode(cat_idx, cardinalities):
        result = []
        for i, card in enumerate(cardinalities):
            result.append(np.eye(card)[cat_idx[:, i]])
        return np.hstack(result).astype(np.float32)

    cat_train_onehot = onehot_encode(cat_train, cat_cardinalities)
    cat_test_onehot = onehot_encode(cat_test, cat_cardinalities)

    X_train_ml = np.column_stack([X_train_num, cat_train_onehot])
    X_test_ml = np.column_stack([X_test_num, cat_test_onehot])

    # Prepare synthetic features
    X_synthetic, y_synthetic = prepare_ml_features(
        synthetic_df, cat_columns, cat_cardinalities
    )

    print(f"\nFeature shapes:")
    print(f"  Original train: {X_train_ml.shape}")
    print(f"  Synthetic train: {X_synthetic.shape}")
    print(f"  Test: {X_test_ml.shape}")

    # Datasets to evaluate
    datasets = {
        "Original": (X_train_ml, y_train),
        "CTGAN Synthetic": (X_synthetic, y_synthetic),
        "Original + CTGAN": (
            np.vstack([X_train_ml, X_synthetic]),
            np.concatenate([y_train, y_synthetic])
        ),
    }

    # Evaluate
    regressors = [
        ("Random Forest", lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ("Gradient Boosting", lambda: GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("Ridge", lambda: Ridge(alpha=1.0)),
    ]

    results = {}

    for reg_name, reg_factory in regressors:
        print(f"\n{reg_name}:")
        print("-" * 50)

        results[reg_name] = {}
        baseline_rmse = None

        for data_name, (X, y) in datasets.items():
            reg = reg_factory()
            metrics = evaluate_regressor(reg_name, reg, X, y, X_test_ml, y_test)
            results[reg_name][data_name] = metrics

            if data_name == "Original":
                baseline_rmse = metrics['rmse']
                print(f"  {data_name:20}: RMSE={metrics['rmse']:.1f}, R²={metrics['r2']:.4f}")
            else:
                delta_pct = (metrics['rmse'] - baseline_rmse) / baseline_rmse * 100
                print(f"  {data_name:20}: RMSE={metrics['rmse']:.1f} ({delta_pct:+.1f}%), R²={metrics['r2']:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    print("\n| Model | Original R² | CTGAN Synthetic R² | Delta |")
    print("|-------|-------------|-------------------|-------|")

    for reg_name in results:
        orig_r2 = results[reg_name]["Original"]["r2"]
        ctgan_r2 = results[reg_name]["CTGAN Synthetic"]["r2"]
        delta = ctgan_r2 - orig_r2
        print(f"| {reg_name:17} | {orig_r2:.4f} | {ctgan_r2:.4f} | {delta:+.4f} |")

    print(f"\n{'='*70}")
    print(f"CTGAN training complete!")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/ozel_rich/prepared.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=500)
    args = parser.parse_args()

    main(args)
