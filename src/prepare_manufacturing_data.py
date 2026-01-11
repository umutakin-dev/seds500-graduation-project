"""
Prepare manufacturing duration data for diffusion model training.

This script:
1. Loads the manufacturing data (Teklif Maliyet SSEK data)
2. Extracts Çap (diameter) and Boy (length) from material description
3. Prepares data for GaussianDiffusion (numeric only)

NOTE: We intentionally DO NOT use Kısa tanım or İş Yeri as features because
they encode "which machine" and explain 80% of the variance - essentially
giving away the answer. With only Çap and Boy, the problem is genuinely
challenging (R² ~ 0.75 baseline) and suitable for testing augmentation.

Usage:
    python src/prepare_manufacturing_data.py --mode explore
    python src/prepare_manufacturing_data.py --mode prepare --output data/manufacturing/prepared.pt
"""

import argparse
import pickle
import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/manufacturing/Teklif Maliyet SSEK data.xlsx")
TARGET_COL = "Makine Süre (100.000 ADET) DK"


def load_data():
    """Load the manufacturing data."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    print(f"Shape: {df.shape}")
    return df


def extract_dimensions(text):
    """
    Extract Çap (diameter) and Boy (length) from material description.

    Handles formats like:
    - M8x25 (standard) -> cap=8, boy=25
    - M8x1,00x50 (with pitch) -> cap=8, boy=50
    - Ø8.91x67,50 (diameter notation) -> cap=8.91, boy=67.50
    """
    text = str(text)

    # Pattern 1: M8x1,00x50 (with pitch) -> cap=8, boy=50
    match = re.match(r'[MØ]?(\d+(?:[.,]\d+)?)[xX]\d+[.,]\d+[xX](\d+(?:[.,]\d+)?)', text)
    if match:
        cap = float(match.group(1).replace(',', '.'))
        boy = float(match.group(2).replace(',', '.'))
        return cap, boy

    # Pattern 2: M8x25 (standard) -> cap=8, boy=25
    match = re.match(r'[MØ]?(\d+(?:[.,]\d+)?)[xX](\d+(?:[.,]\d+)?)', text)
    if match:
        cap = float(match.group(1).replace(',', '.'))
        boy = float(match.group(2).replace(',', '.'))
        # Sanity check: boy should be < 500mm for bolts
        if boy > 500:
            return cap, None  # Parsing error, keep cap but discard boy
        return cap, boy

    return None, None


def explore_data(df):
    """Print detailed information about the dataset."""
    print("\n" + "=" * 80)
    print("MANUFACTURING DATA EXPLORATION")
    print("=" * 80)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Target statistics
    print("\n" + "-" * 80)
    print(f"TARGET: {TARGET_COL}")
    print("-" * 80)
    target = df[TARGET_COL]
    print(f"Count: {target.notna().sum()}")
    print(f"Missing: {target.isna().sum()}")
    print(f"Min: {target.min():.1f}")
    print(f"Max: {target.max():.1f}")
    print(f"Mean: {target.mean():.1f}")
    print(f"Median: {target.median():.1f}")
    print(f"Std: {target.std():.1f}")

    # Test dimension extraction
    print("\n" + "-" * 80)
    print("DIMENSION EXTRACTION")
    print("-" * 80)
    df['Çap'], df['Boy'] = zip(*df['Malzeme Tanımı'].apply(extract_dimensions))
    print(f"Çap extracted: {df['Çap'].notna().sum()} / {len(df)} ({df['Çap'].notna().sum()/len(df)*100:.1f}%)")
    print(f"Boy extracted: {df['Boy'].notna().sum()} / {len(df)} ({df['Boy'].notna().sum()/len(df)*100:.1f}%)")
    print(f"\nÇap range: {df['Çap'].min():.1f} - {df['Çap'].max():.1f} mm")
    print(f"Boy range: {df['Boy'].min():.1f} - {df['Boy'].max():.1f} mm")
    print(f"\nÇap correlation with target: {df['Çap'].corr(df[TARGET_COL]):.4f}")
    print(f"Boy correlation with target: {df['Boy'].corr(df[TARGET_COL]):.4f}")

    # Feature set explanation
    print("\n" + "-" * 80)
    print("FEATURE SELECTION RATIONALE")
    print("-" * 80)
    print("Using ONLY Çap and Boy as features because:")
    print("  - Kısa tanım and İş Yeri encode 'which machine' → explains 80% of variance")
    print("  - Using them would be data leakage (too easy)")
    print("  - With only Çap/Boy, baseline R² ~ 0.75 (room for augmentation to help)")


def prepare_data(df, test_size=0.2):
    """
    Prepare data for GaussianDiffusion training (numeric only).

    Uses only Çap and Boy as features - no categorical features.
    This creates a harder prediction problem suitable for testing augmentation.

    Returns dict with train/test splits and preprocessing objects.
    """
    print("\n" + "=" * 80)
    print("PREPARING DATA FOR DIFFUSION (NUMERIC ONLY)")
    print("=" * 80)

    # Extract dimensions
    print("Extracting dimensions from Malzeme Tanımı...")
    df['Çap'], df['Boy'] = zip(*df['Malzeme Tanımı'].apply(extract_dimensions))

    # Remove rows with missing dimensions or target
    initial_count = len(df)
    df = df.dropna(subset=['Çap', 'Boy', TARGET_COL])
    print(f"Rows after removing missing: {len(df)} / {initial_count} ({len(df)/initial_count*100:.1f}%)")

    # Numeric features: Çap, Boy
    X = df[['Çap', 'Boy']].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    print(f"Features: Çap, Boy (2 numeric)")
    print(f"Target: {TARGET_COL}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    # Scale features
    scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = np.clip(X_train_scaled, -3, 3) / 3
    X_test_scaled = np.clip(X_test_scaled, -3, 3) / 3

    # Scale target
    target_scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    y_train_scaled = np.clip(y_train_scaled, -3, 3) / 3
    y_test_scaled = np.clip(y_test_scaled, -3, 3) / 3

    # Combine features and target for diffusion training
    # Format: [Çap, Boy, target]
    X_train_full = np.column_stack([X_train_scaled, y_train_scaled]).astype(np.float32)
    X_test_full = np.column_stack([X_test_scaled, y_test_scaled]).astype(np.float32)

    print(f"\nFinal shapes:")
    print(f"  Train: {X_train_full.shape} (Çap, Boy, target)")
    print(f"  Test: {X_test_full.shape}")

    return {
        "X_train": X_train_full,
        "X_test": X_test_full,
        "X_train_unscaled": X_train,  # Original features
        "X_test_unscaled": X_test,
        "y_train": y_train,  # Original target
        "y_test": y_test,
        "num_features": 3,  # Çap, Boy, target
        "feature_names": ["Çap", "Boy", "target"],
        "scaler": scaler,
        "target_scaler": target_scaler,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare manufacturing data for diffusion")
    parser.add_argument("--mode", choices=["explore", "prepare"], default="explore",
                        help="Mode: explore (analyze data) or prepare (create training data)")
    parser.add_argument("--output", type=str, default="data/manufacturing/prepared.pt",
                        help="Output path for prepared data")
    args = parser.parse_args()

    df = load_data()

    if args.mode == "explore":
        explore_data(df)

    elif args.mode == "prepare":
        data = prepare_data(df)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save tensors
        torch.save({
            "X_train": torch.tensor(data["X_train"]),
            "X_test": torch.tensor(data["X_test"]),
            "X_train_unscaled": data["X_train_unscaled"],
            "X_test_unscaled": data["X_test_unscaled"],
            "y_train": data["y_train"],
            "y_test": data["y_test"],
            "num_features": data["num_features"],
            "feature_names": data["feature_names"],
        }, output_path)
        print(f"\nData saved to {output_path}")

        # Save preprocessors
        preprocessor_path = output_path.parent / "preprocessors.pkl"
        with open(preprocessor_path, "wb") as f:
            pickle.dump({
                "scaler": data["scaler"],
                "target_scaler": data["target_scaler"],
            }, f)
        print(f"Preprocessors saved to {preprocessor_path}")


if __name__ == "__main__":
    main()
