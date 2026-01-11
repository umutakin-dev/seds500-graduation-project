"""
Prepare production data for diffusion model training.

This script:
1. Loads the production data (quotation documents)
2. Identifies numeric and categorical columns
3. Handles missing values and Turkish number formats
4. Prepares data for Gaussian (numeric-only) or Hybrid (full) diffusion

Usage:
    python src/prepare_production_data.py --mode explore
    python src/prepare_production_data.py --mode numeric --output data/production/numeric.pt
    python src/prepare_production_data.py --mode full --output data/production/full.pt
"""

import argparse
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/production/6_Ay_Teklif_Dokumanlari_Sayfa2.xlsx")

# Target columns
TARGET_COLS = {
    "kar_marji": "KAR MARJI",
    "teklif_miktari": "İlk Girilen Teklif Miktarı",
}


def load_data():
    """Load the production data."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    print(f"Shape: {df.shape}")
    return df


def parse_turkish_number(value):
    """Parse Turkish number format (1.234,56 -> 1234.56)."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    try:
        # Remove thousand separators (.) and replace decimal comma with dot
        s = str(value).strip()
        s = s.replace(".", "").replace(",", ".")
        return float(s)
    except (ValueError, AttributeError):
        return np.nan


def explore_data(df):
    """Print detailed information about the dataset."""
    print("\n" + "=" * 80)
    print("DATA EXPLORATION")
    print("=" * 80)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Identify column types
    numeric_cols = []
    categorical_cols = []
    date_cols = []

    print("\n" + "-" * 80)
    print("COLUMN ANALYSIS")
    print("-" * 80)
    print(f"{'#':>3} {'Column':<40} {'Type':<12} {'Filled':>7} {'Unique':>7}")
    print("-" * 80)

    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        pct = non_null / len(df) * 100
        unique = df[col].nunique()

        # Classify column
        if "date" in col.lower() or "tarih" in col.lower():
            col_type = "date"
            date_cols.append(col)
        elif dtype in ["float64", "int64"]:
            col_type = "numeric"
            numeric_cols.append(col)
        elif dtype == "object":
            # Check if it's a number in Turkish format
            sample = df[col].dropna().head(10)
            is_numeric = all(
                isinstance(v, (int, float)) or
                (isinstance(v, str) and v.replace(".", "").replace(",", "").replace("-", "").isdigit())
                for v in sample if pd.notna(v)
            )
            if is_numeric and unique > 20:
                col_type = "numeric*"
                numeric_cols.append(col)
            else:
                col_type = "categorical"
                categorical_cols.append(col)
        else:
            col_type = str(dtype)
            categorical_cols.append(col)

        print(f"{i+1:>3} {col[:40]:<40} {col_type:<12} {pct:>6.1f}% {unique:>7}")

    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"Numeric columns:     {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Date columns:        {len(date_cols)}")

    # Check targets
    print("\n" + "-" * 80)
    print("TARGET COLUMNS")
    print("-" * 80)
    for name, col in TARGET_COLS.items():
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct = non_null / len(df) * 100
            print(f"{name}: '{col}' - {pct:.1f}% filled ({non_null} rows)")
        else:
            print(f"{name}: '{col}' - NOT FOUND")

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "date_cols": date_cols,
    }


def prepare_numeric_data(df, target_col="kar_marji", test_size=0.2):
    """
    Prepare numeric-only data for Gaussian diffusion.

    Returns dict with train/test splits and preprocessing objects.
    """
    print("\n" + "=" * 80)
    print("PREPARING NUMERIC DATA")
    print("=" * 80)

    target = TARGET_COLS[target_col]
    print(f"Target: {target}")

    # Identify numeric columns (excluding target and IDs)
    exclude_patterns = ["id", "no", "numara", "kod", "tarih", "date"]

    numeric_cols = []
    for col in df.columns:
        if col == target:
            continue
        if any(p in col.lower() for p in exclude_patterns):
            continue

        # Check if column is numeric or can be parsed as numeric
        if df[col].dtype in ["float64", "int64"]:
            numeric_cols.append(col)
        elif df[col].dtype == "object":
            # Try parsing as Turkish number
            parsed = df[col].apply(parse_turkish_number)
            if parsed.notna().sum() > len(df) * 0.5:  # At least 50% parseable
                numeric_cols.append(col)

    print(f"Found {len(numeric_cols)} numeric columns")

    # Parse and collect numeric data
    X_data = []
    valid_cols = []

    for col in numeric_cols:
        if df[col].dtype == "object":
            values = df[col].apply(parse_turkish_number).values
        else:
            values = df[col].values.astype(float)

        # Check if column has enough variation
        non_null = (~np.isnan(values)).sum()
        if non_null < len(df) * 0.3:  # Skip if <30% filled
            continue

        X_data.append(values)
        valid_cols.append(col)

    X = np.column_stack(X_data)

    # Parse target
    if df[target].dtype == "object":
        y = df[target].apply(parse_turkish_number).values
    else:
        y = df[target].values.astype(float)

    print(f"Data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target filled: {(~np.isnan(y)).sum()} / {len(y)}")

    # Remove rows with missing target
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"After removing missing targets: {X.shape[0]} rows")

    # Impute missing values in features (median)
    for i in range(X.shape[1]):
        col_median = np.nanmedian(X[:, i])
        X[np.isnan(X[:, i]), i] = col_median

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Scale features
    scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Clip to [-3, 3] and normalize to [-1, 1]
    X_train_scaled = np.clip(X_train_scaled, -3, 3) / 3
    X_test_scaled = np.clip(X_test_scaled, -3, 3) / 3

    # Scale target too
    target_scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    y_train_scaled = np.clip(y_train_scaled, -3, 3) / 3
    y_test_scaled = np.clip(y_test_scaled, -3, 3) / 3

    # Combine features and target for diffusion training
    X_train_full = np.column_stack([X_train_scaled, y_train_scaled])
    X_test_full = np.column_stack([X_test_scaled, y_test_scaled])

    print(f"\nFinal shapes:")
    print(f"  Train: {X_train_full.shape}")
    print(f"  Test: {X_test_full.shape}")

    return {
        "X_train": X_train_full.astype(np.float32),
        "X_test": X_test_full.astype(np.float32),
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols": valid_cols,
        "target_col": target,
        "scaler": scaler,
        "target_scaler": target_scaler,
        "num_features": len(valid_cols),
    }


def prepare_full_data(df, target_col="teklif_miktari", test_size=0.2, min_category_freq=50):
    """
    Prepare full data (numeric + categorical) for HybridDiffusion.

    Args:
        df: DataFrame with production data
        target_col: Target column key
        test_size: Fraction for test split
        min_category_freq: Minimum frequency for a category to be kept (others -> "Other")

    Returns dict with train/test splits and preprocessing objects.
    """
    print("\n" + "=" * 80)
    print("PREPARING FULL DATA (NUMERIC + CATEGORICAL)")
    print("=" * 80)

    target = TARGET_COLS[target_col]
    print(f"Target: {target}")

    # Identify columns by type
    exclude_patterns = ["id", "numara", "kod", "tarih", "date", "tanım", "malzeme"]

    numerical_cols = []
    categorical_cols = []

    for col in df.columns:
        if col == target:
            continue
        col_lower = col.lower()
        if any(p in col_lower for p in exclude_patterns):
            continue

        # Check fill rate
        fill_rate = df[col].notna().sum() / len(df)
        if fill_rate < 0.5:  # Skip columns with <50% fill rate
            continue

        dtype = df[col].dtype
        unique = df[col].nunique()

        if dtype in ["float64", "int64"]:
            if unique > 10:  # Treat as numeric if >10 unique values
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
        elif dtype == "object":
            # Try parsing as number
            sample = df[col].dropna().head(20)
            numeric_count = sum(
                isinstance(v, (int, float)) or
                (isinstance(v, str) and v.replace(".", "").replace(",", "").replace("-", "").replace(" ", "").isdigit())
                for v in sample
            )
            if numeric_count > len(sample) * 0.8 and unique > 20:
                numerical_cols.append(col)
            elif unique <= 50:  # Limit categorical cardinality
                categorical_cols.append(col)

    print(f"Selected {len(numerical_cols)} numeric columns")
    print(f"Selected {len(categorical_cols)} categorical columns")

    # Parse target
    if df[target].dtype == "object":
        y = df[target].apply(parse_turkish_number).values
    else:
        y = df[target].values.astype(float)

    # Remove rows with missing target
    valid_mask = ~np.isnan(y)
    df_valid = df[valid_mask].copy()
    y = y[valid_mask]

    print(f"After removing missing targets: {len(df_valid)} rows")

    # Process numerical columns
    X_num = []
    valid_num_cols = []
    for col in numerical_cols:
        if df_valid[col].dtype == "object":
            values = df_valid[col].apply(parse_turkish_number).values
        else:
            values = df_valid[col].values.astype(float)

        # Impute with median
        median = np.nanmedian(values)
        values = np.where(np.isnan(values), median, values)
        X_num.append(values)
        valid_num_cols.append(col)

    X_num = np.column_stack(X_num) if X_num else np.zeros((len(df_valid), 0))

    # Process categorical columns
    label_encoders = {}
    cat_cardinalities = []
    X_cat = []
    valid_cat_cols = []

    for col in categorical_cols:
        # Fill missing with "Missing"
        values = df_valid[col].fillna("Missing").astype(str)

        # Group rare categories into "Other"
        value_counts = values.value_counts()
        rare_cats = value_counts[value_counts < min_category_freq].index
        values = values.replace(rare_cats, "Other")

        # Encode
        le = LabelEncoder()
        encoded = le.fit_transform(values)
        label_encoders[col] = le
        cat_cardinalities.append(len(le.classes_))
        X_cat.append(encoded)
        valid_cat_cols.append(col)

    X_cat = np.column_stack(X_cat) if X_cat else np.zeros((len(df_valid), 0), dtype=np.int64)

    print(f"\nFinal columns:")
    print(f"  Numeric: {len(valid_num_cols)}")
    print(f"  Categorical: {len(valid_cat_cols)}")
    print(f"  Categorical cardinalities: {cat_cardinalities}")

    # Train/test split (keeping indices aligned)
    indices = np.arange(len(df_valid))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)

    X_num_train, X_num_test = X_num[train_idx], X_num[test_idx]
    X_cat_train, X_cat_test = X_cat[train_idx], X_cat[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale numerical features
    scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)
    X_num_train_scaled = np.clip(X_num_train_scaled, -3, 3) / 3
    X_num_test_scaled = np.clip(X_num_test_scaled, -3, 3) / 3

    # Scale target (treat as additional numeric feature)
    target_scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    y_train_scaled = np.clip(y_train_scaled, -3, 3) / 3
    y_test_scaled = np.clip(y_test_scaled, -3, 3) / 3

    # Include target as last numerical feature
    X_num_train_scaled = np.column_stack([X_num_train_scaled, y_train_scaled])
    X_num_test_scaled = np.column_stack([X_num_test_scaled, y_test_scaled])

    # Convert categorical to one-hot
    def to_onehot(X_cat, cardinalities):
        batch_size = X_cat.shape[0]
        onehot_list = []
        for i, card in enumerate(cardinalities):
            onehot = np.zeros((batch_size, card), dtype=np.float32)
            onehot[np.arange(batch_size), X_cat[:, i]] = 1.0
            onehot_list.append(onehot)
        return np.hstack(onehot_list) if onehot_list else np.zeros((batch_size, 0), dtype=np.float32)

    X_cat_train_onehot = to_onehot(X_cat_train, cat_cardinalities)
    X_cat_test_onehot = to_onehot(X_cat_test, cat_cardinalities)

    # Combine: [numerical (including target), categorical_onehot]
    X_train = np.hstack([X_num_train_scaled, X_cat_train_onehot]).astype(np.float32)
    X_test = np.hstack([X_num_test_scaled, X_cat_test_onehot]).astype(np.float32)

    num_numerical = X_num_train_scaled.shape[1]  # Includes target

    print(f"\nFinal shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Numerical dims: {num_numerical} (including target)")
    print(f"  Categorical dims: {sum(cat_cardinalities)}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "num_numerical": num_numerical,
        "cat_cardinalities": cat_cardinalities,
        "numerical_cols": valid_num_cols + [target],  # Include target
        "categorical_cols": valid_cat_cols,
        "target_col": target,
        "scaler": scaler,
        "target_scaler": target_scaler,
        "label_encoders": label_encoders,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["explore", "numeric", "full"], default="explore")
    parser.add_argument("--target", choices=["kar_marji", "teklif_miktari"], default="kar_marji")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    df = load_data()

    if args.mode == "explore":
        explore_data(df)

    elif args.mode == "numeric":
        data = prepare_numeric_data(df, target_col=args.target)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save tensors
            torch.save({
                "X_train": torch.tensor(data["X_train"]),
                "X_test": torch.tensor(data["X_test"]),
                "y_train": data["y_train"],
                "y_test": data["y_test"],
                "feature_cols": data["feature_cols"],
                "target_col": data["target_col"],
                "num_features": data["num_features"],
            }, output_path)
            print(f"\nData saved to {output_path}")

            # Save preprocessors
            preprocessor_path = output_path.parent / "preprocessors_numeric.pkl"
            with open(preprocessor_path, "wb") as f:
                pickle.dump({
                    "scaler": data["scaler"],
                    "target_scaler": data["target_scaler"],
                }, f)
            print(f"Preprocessors saved to {preprocessor_path}")

    elif args.mode == "full":
        data = prepare_full_data(df, target_col=args.target)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save tensors
            torch.save({
                "X_train": torch.tensor(data["X_train"]),
                "X_test": torch.tensor(data["X_test"]),
                "y_train": data["y_train"],
                "y_test": data["y_test"],
                "num_numerical": data["num_numerical"],
                "cat_cardinalities": data["cat_cardinalities"],
                "numerical_cols": data["numerical_cols"],
                "categorical_cols": data["categorical_cols"],
                "target_col": data["target_col"],
            }, output_path)
            print(f"\nData saved to {output_path}")

            # Save preprocessors
            preprocessor_path = output_path.parent / "preprocessors_full.pkl"
            with open(preprocessor_path, "wb") as f:
                pickle.dump({
                    "scaler": data["scaler"],
                    "target_scaler": data["target_scaler"],
                    "label_encoders": data["label_encoders"],
                }, f)
            print(f"Preprocessors saved to {preprocessor_path}")


if __name__ == "__main__":
    main()
