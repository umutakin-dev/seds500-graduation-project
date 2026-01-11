"""
Prepare ozel (special engineering) dataset for diffusion model training.

This dataset is pre-split and has:
- Cap (diameter): numeric
- Boy (length): numeric
- IslemTipi (process type): categorical (FT, HT, Belirsiz)
- MakineSure: target (numeric) - manufacturing duration for 100k units

Since we have mixed types, this prepares data for HybridDiffusion.

Usage:
    python src/prepare_ozel_data.py --mode explore
    python src/prepare_ozel_data.py --mode prepare --output data/ozel/prepared.pt
"""

import argparse
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer, LabelEncoder


TRAIN_PATH = Path("data/ozel/train.csv")
TEST_PATH = Path("data/ozel/test.csv")
TARGET_COL = "MakineSure"


def load_data():
    """Load the ozel train/test data from CSV."""
    print(f"Loading train data from {TRAIN_PATH}...")
    train = pd.read_csv(TRAIN_PATH)
    print(f"Train shape: {train.shape}")

    print(f"Loading test data from {TEST_PATH}...")
    test = pd.read_csv(TEST_PATH)
    print(f"Test shape: {test.shape}")

    return train, test


def explore_data(train, test):
    """Print detailed information about the dataset."""
    print("\n" + "=" * 80)
    print("OZEL DATASET EXPLORATION")
    print("=" * 80)

    print(f"\nTrain: {train.shape[0]} rows, Test: {test.shape[0]} rows")
    print(f"Columns: {list(train.columns)}")

    # Numeric features
    print("\n" + "-" * 80)
    print("NUMERIC FEATURES")
    print("-" * 80)

    print(f"\nCap (diameter):")
    print(f"  Train: min={train['Cap'].min():.2f}, max={train['Cap'].max():.2f}, mean={train['Cap'].mean():.2f}")
    print(f"  Test:  min={test['Cap'].min():.2f}, max={test['Cap'].max():.2f}, mean={test['Cap'].mean():.2f}")

    print(f"\nBoy (length):")
    print(f"  Train: min={train['Boy'].min():.2f}, max={train['Boy'].max():.2f}, mean={train['Boy'].mean():.2f}")
    print(f"  Test:  min={test['Boy'].min():.2f}, max={test['Boy'].max():.2f}, mean={test['Boy'].mean():.2f}")

    # Target
    print("\n" + "-" * 80)
    print(f"TARGET: {TARGET_COL}")
    print("-" * 80)
    print(f"  Train: min={train[TARGET_COL].min():.1f}, max={train[TARGET_COL].max():.1f}, mean={train[TARGET_COL].mean():.1f}, std={train[TARGET_COL].std():.1f}")
    print(f"  Test:  min={test[TARGET_COL].min():.1f}, max={test[TARGET_COL].max():.1f}, mean={test[TARGET_COL].mean():.1f}, std={test[TARGET_COL].std():.1f}")

    # Categorical feature
    print("\n" + "-" * 80)
    print("CATEGORICAL: IslemTipi")
    print("-" * 80)
    print(f"Train distribution:\n{train['IslemTipi'].value_counts()}")
    print(f"\nTest distribution:\n{test['IslemTipi'].value_counts()}")

    # Correlations
    print("\n" + "-" * 80)
    print("CORRELATIONS WITH TARGET")
    print("-" * 80)
    print(f"Cap: {train['Cap'].corr(train[TARGET_COL]):.4f}")
    print(f"Boy: {train['Boy'].corr(train[TARGET_COL]):.4f}")

    # Target by IslemTipi
    print(f"\nMean target by IslemTipi:")
    print(train.groupby('IslemTipi')[TARGET_COL].agg(['mean', 'std', 'count']))


def prepare_data(train, test):
    """
    Prepare data for HybridDiffusion training.

    Returns dict with train/test splits and preprocessing objects.
    Format: [Cap, Boy, target, IslemTipi_onehot(3)]
    """
    print("\n" + "=" * 80)
    print("PREPARING DATA FOR HYBRID DIFFUSION")
    print("=" * 80)

    print(f"Columns: Cap, Boy, IslemTipi, {TARGET_COL}")

    # Remove rows with invalid values (e.g., -1)
    train_clean = train[(train['Cap'] > 0) & (train['Boy'] > 0)].copy()
    test_clean = test[(test['Cap'] > 0) & (test['Boy'] > 0)].copy()
    print(f"\nAfter removing invalid rows: Train={len(train_clean)}, Test={len(test_clean)}")

    # Extract features
    X_train_num = train_clean[['Cap', 'Boy']].values.astype(np.float32)
    X_test_num = test_clean[['Cap', 'Boy']].values.astype(np.float32)

    y_train = train_clean[TARGET_COL].values.astype(np.float32)
    y_test = test_clean[TARGET_COL].values.astype(np.float32)

    # Encode categorical
    le = LabelEncoder()
    cat_train = le.fit_transform(train_clean['IslemTipi'])
    cat_test = le.transform(test_clean['IslemTipi'])

    cat_cardinality = len(le.classes_)
    print(f"IslemTipi categories: {list(le.classes_)} (cardinality={cat_cardinality})")

    # One-hot encode
    cat_train_onehot = np.eye(cat_cardinality)[cat_train].astype(np.float32)
    cat_test_onehot = np.eye(cat_cardinality)[cat_test].astype(np.float32)

    print(f"\nTrain size: {len(X_train_num)}")
    print(f"Test size: {len(X_test_num)}")

    # Scale numeric features
    scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)
    X_train_scaled = np.clip(X_train_scaled, -3, 3) / 3
    X_test_scaled = np.clip(X_test_scaled, -3, 3) / 3

    # Scale target
    target_scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    y_train_scaled = np.clip(y_train_scaled, -3, 3) / 3
    y_test_scaled = np.clip(y_test_scaled, -3, 3) / 3

    # Combine: [Cap, Boy, target] + [IslemTipi_onehot(3)]
    # Format for HybridDiffusion: numerical first, then one-hot categorical
    X_train_full = np.column_stack([
        X_train_scaled,      # Cap, Boy (scaled)
        y_train_scaled.reshape(-1, 1),  # target (scaled)
        cat_train_onehot     # IslemTipi one-hot (3)
    ]).astype(np.float32)

    X_test_full = np.column_stack([
        X_test_scaled,
        y_test_scaled.reshape(-1, 1),
        cat_test_onehot
    ]).astype(np.float32)

    print(f"\nFinal shapes:")
    print(f"  Train: {X_train_full.shape} (Cap, Boy, target, IslemTipi[3])")
    print(f"  Test: {X_test_full.shape}")

    # Dimensions for HybridDiffusion
    num_numerical = 3  # Cap, Boy, target
    cat_cardinalities = [cat_cardinality]  # IslemTipi has 3 categories

    return {
        "X_train": X_train_full,
        "X_test": X_test_full,
        "X_train_num_unscaled": X_train_num,  # Original numeric features
        "X_test_num_unscaled": X_test_num,
        "y_train": y_train,  # Original target
        "y_test": y_test,
        "cat_train": cat_train,  # Categorical indices
        "cat_test": cat_test,
        "num_numerical": num_numerical,
        "cat_cardinalities": cat_cardinalities,
        "feature_names": ["Çap", "Boy", "target", "IslemTipi"],
        "cat_classes": list(le.classes_),
        "scaler": scaler,
        "target_scaler": target_scaler,
        "label_encoder": le,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare ozel data for diffusion")
    parser.add_argument("--mode", choices=["explore", "prepare"], default="explore",
                        help="Mode: explore (analyze data) or prepare (create training data)")
    parser.add_argument("--output", type=str, default="data/ozel/prepared.pt",
                        help="Output path for prepared data")
    args = parser.parse_args()

    train, test = load_data()

    if args.mode == "explore":
        explore_data(train, test)

    elif args.mode == "prepare":
        data = prepare_data(train, test)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save tensors and config
        torch.save({
            "X_train": torch.tensor(data["X_train"]),
            "X_test": torch.tensor(data["X_test"]),
            "X_train_num_unscaled": data["X_train_num_unscaled"],
            "X_test_num_unscaled": data["X_test_num_unscaled"],
            "y_train": data["y_train"],
            "y_test": data["y_test"],
            "cat_train": data["cat_train"],
            "cat_test": data["cat_test"],
            "num_numerical": data["num_numerical"],
            "cat_cardinalities": data["cat_cardinalities"],
            "feature_names": data["feature_names"],
            "cat_classes": data["cat_classes"],
        }, output_path)
        print(f"\nData saved to {output_path}")

        # Save preprocessors
        preprocessor_path = output_path.parent / "preprocessors.pkl"
        with open(preprocessor_path, "wb") as f:
            pickle.dump({
                "scaler": data["scaler"],
                "target_scaler": data["target_scaler"],
                "label_encoder": data["label_encoder"],
            }, f)
        print(f"Preprocessors saved to {preprocessor_path}")


if __name__ == "__main__":
    main()
