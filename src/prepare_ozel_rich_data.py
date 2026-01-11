"""
Prepare ozel subset from manufacturing data with richer features.

Experiment 013: Extract ozel rows from the 18k manufacturing dataset
and use more features than the standalone ozel dataset (exp 012).

Features:
- Cap (diameter): numeric, extracted from Malzeme Tanımı
- Boy (length): numeric, extracted from Malzeme Tanımı
- IslemTipi (FT/HT/Belirsiz): categorical, extracted from Malzeme Tanımı
- AnmaOlcusu (thread size M6/M8/M10...): categorical
- Sartname (OZEL/OZEL+A/OZEL+R...): categorical
- UY (production area 2101/2501): categorical

Target: Makine Süre (100.000 ADET) DK

NOTE: We do NOT use İş Yeri or cost columns as they cause data leakage.

Usage:
    python src/prepare_ozel_rich_data.py --mode explore
    python src/prepare_ozel_rich_data.py --mode prepare --output data/ozel_rich/prepared.pt
"""

import argparse
import pickle
import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/manufacturing/Teklif Maliyet SSEK data.xlsx")
TARGET_COL = "Makine Süre (100.000 ADET) DK"


def load_data():
    """Load manufacturing data and filter to ozel rows."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    print(f"Total rows: {len(df)}")

    # Filter to ozel rows
    ozel_mask = df['Malzeme Tanımı'].str.lower().str.contains('özel|ozel', na=False)
    df_ozel = df[ozel_mask].copy()
    print(f"Ozel rows: {len(df_ozel)}")

    return df_ozel


def extract_dimensions(text):
    """Extract Cap and Boy from Malzeme Tanımı (e.g., M8x25 -> cap=8, boy=25)."""
    text = str(text)
    match = re.match(r'M(\d+(?:[.,]\d+)?)[xX](\d+(?:[.,]\d+)?)', text)
    if match:
        cap = float(match.group(1).replace(',', '.'))
        boy = float(match.group(2).replace(',', '.'))
        return cap, boy
    return None, None


def extract_islem_tipi(text):
    """Extract IslemTipi (FT, HT, or Belirsiz) from Malzeme Tanımı."""
    text = str(text).upper()
    if ' FT ' in text or text.endswith(' FT'):
        return 'FT'
    elif ' HT ' in text or text.endswith(' HT'):
        return 'HT'
    else:
        return 'Belirsiz'


def simplify_anma_olcusu(value):
    """Simplify AnmaOlcusu to standard thread sizes."""
    value = str(value).strip()
    # Keep only M-series standard sizes
    if value.startswith('M') and value[1:].isdigit():
        return value
    elif value.startswith('M') and '.' not in value:
        # M10, M12, etc.
        return value
    else:
        return 'Other'


def simplify_sartname(value):
    """Simplify Sartname to main categories."""
    value = str(value).strip().upper()
    if value == 'OZEL':
        return 'OZEL'
    elif value.startswith('OZEL+A'):
        return 'OZEL+A'
    elif value.startswith('OZEL+R'):
        return 'OZEL+R'
    elif value.startswith('OZEL+T'):
        return 'OZEL+T'
    else:
        return 'Other'


def explore_data(df):
    """Print detailed information about the dataset."""
    print("\n" + "=" * 80)
    print("OZEL RICH DATASET EXPLORATION")
    print("=" * 80)

    # Extract features
    df['Cap'], df['Boy'] = zip(*df['Malzeme Tanımı'].apply(extract_dimensions))
    df['IslemTipi'] = df['Malzeme Tanımı'].apply(extract_islem_tipi)
    df['AnmaOlcusu'] = df['Anma ölçüsü'].apply(simplify_anma_olcusu)
    df['SartnameSimple'] = df['Sartname'].apply(simplify_sartname)

    # Filter valid rows
    df_valid = df.dropna(subset=['Cap', 'Boy', TARGET_COL])
    print(f"\nValid rows (with Cap/Boy): {len(df_valid)} / {len(df)}")

    print(f"\n--- NUMERIC FEATURES ---")
    print(f"Cap: min={df_valid['Cap'].min():.1f}, max={df_valid['Cap'].max():.1f}, mean={df_valid['Cap'].mean():.1f}")
    print(f"Boy: min={df_valid['Boy'].min():.1f}, max={df_valid['Boy'].max():.1f}, mean={df_valid['Boy'].mean():.1f}")

    print(f"\n--- TARGET ---")
    print(f"{TARGET_COL}:")
    print(f"  min={df_valid[TARGET_COL].min():.1f}, max={df_valid[TARGET_COL].max():.1f}")
    print(f"  mean={df_valid[TARGET_COL].mean():.1f}, std={df_valid[TARGET_COL].std():.1f}")

    print(f"\n--- CATEGORICAL FEATURES ---")
    print(f"\nIslemTipi:")
    print(df_valid['IslemTipi'].value_counts())

    print(f"\nAnmaOlcusu (simplified):")
    print(df_valid['AnmaOlcusu'].value_counts())

    print(f"\nSartname (simplified):")
    print(df_valid['SartnameSimple'].value_counts())

    print(f"\nUY (production area):")
    print(df_valid['ÜY'].value_counts())

    print(f"\n--- CORRELATIONS WITH TARGET ---")
    print(f"Cap: {df_valid['Cap'].corr(df_valid[TARGET_COL]):.4f}")
    print(f"Boy: {df_valid['Boy'].corr(df_valid[TARGET_COL]):.4f}")

    print(f"\n--- MEAN TARGET BY CATEGORICAL ---")
    print(f"\nBy IslemTipi:")
    print(df_valid.groupby('IslemTipi')[TARGET_COL].agg(['mean', 'count']))
    print(f"\nBy AnmaOlcusu (top 6):")
    print(df_valid.groupby('AnmaOlcusu')[TARGET_COL].agg(['mean', 'count']).sort_values('count', ascending=False).head(6))


def prepare_data(df, test_size=0.2):
    """
    Prepare data for HybridDiffusion training.

    Features:
    - Numeric: Cap, Boy, target (3)
    - Categorical: IslemTipi, AnmaOlcusu, SartnameSimple, UY (4)
    """
    print("\n" + "=" * 80)
    print("PREPARING DATA FOR HYBRID DIFFUSION (RICH FEATURES)")
    print("=" * 80)

    # Extract features
    df['Cap'], df['Boy'] = zip(*df['Malzeme Tanımı'].apply(extract_dimensions))
    df['IslemTipi'] = df['Malzeme Tanımı'].apply(extract_islem_tipi)
    df['AnmaOlcusu'] = df['Anma ölçüsü'].apply(simplify_anma_olcusu)
    df['SartnameSimple'] = df['Sartname'].apply(simplify_sartname)
    df['UY'] = df['ÜY'].astype(str)

    # Filter valid rows
    df_valid = df.dropna(subset=['Cap', 'Boy', TARGET_COL]).copy()
    print(f"Valid rows: {len(df_valid)}")

    # Extract arrays
    X_num = df_valid[['Cap', 'Boy']].values.astype(np.float32)
    y = df_valid[TARGET_COL].values.astype(np.float32)

    # Encode categoricals
    cat_columns = ['IslemTipi', 'AnmaOlcusu', 'SartnameSimple', 'UY']
    encoders = {}
    cat_indices = []
    cat_cardinalities = []

    for col in cat_columns:
        le = LabelEncoder()
        cat_idx = le.fit_transform(df_valid[col])
        encoders[col] = le
        cat_indices.append(cat_idx)
        cat_cardinalities.append(len(le.classes_))
        print(f"{col}: {len(le.classes_)} categories - {list(le.classes_)}")

    cat_indices = np.column_stack(cat_indices)

    # Train/test split
    (X_num_train, X_num_test, y_train, y_test,
     cat_train, cat_test) = train_test_split(
        X_num, y, cat_indices, test_size=test_size, random_state=42
    )

    print(f"\nTrain size: {len(X_num_train)}")
    print(f"Test size: {len(X_num_test)}")

    # Scale numeric features
    scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)
    X_num_train_scaled = np.clip(X_num_train_scaled, -3, 3) / 3
    X_num_test_scaled = np.clip(X_num_test_scaled, -3, 3) / 3

    # Scale target
    target_scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    y_train_scaled = np.clip(y_train_scaled, -3, 3) / 3
    y_test_scaled = np.clip(y_test_scaled, -3, 3) / 3

    # One-hot encode categoricals
    cat_train_onehot = []
    cat_test_onehot = []
    for i, card in enumerate(cat_cardinalities):
        cat_train_onehot.append(np.eye(card)[cat_train[:, i]])
        cat_test_onehot.append(np.eye(card)[cat_test[:, i]])

    cat_train_onehot = np.hstack(cat_train_onehot).astype(np.float32)
    cat_test_onehot = np.hstack(cat_test_onehot).astype(np.float32)

    # Combine: [Cap, Boy, target] + [categoricals one-hot]
    X_train_full = np.column_stack([
        X_num_train_scaled,
        y_train_scaled.reshape(-1, 1),
        cat_train_onehot
    ]).astype(np.float32)

    X_test_full = np.column_stack([
        X_num_test_scaled,
        y_test_scaled.reshape(-1, 1),
        cat_test_onehot
    ]).astype(np.float32)

    num_numerical = 3  # Cap, Boy, target
    total_cat_dims = sum(cat_cardinalities)

    print(f"\nFinal shapes:")
    print(f"  Train: {X_train_full.shape} (3 numeric + {total_cat_dims} one-hot)")
    print(f"  Test: {X_test_full.shape}")
    print(f"  Cat cardinalities: {cat_cardinalities}")

    return {
        "X_train": X_train_full,
        "X_test": X_test_full,
        "X_train_num_unscaled": X_num_train,
        "X_test_num_unscaled": X_num_test,
        "y_train": y_train,
        "y_test": y_test,
        "cat_train": cat_train,
        "cat_test": cat_test,
        "num_numerical": num_numerical,
        "cat_cardinalities": cat_cardinalities,
        "cat_columns": cat_columns,
        "feature_names": ["Cap", "Boy", "target"] + cat_columns,
        "encoders": encoders,
        "scaler": scaler,
        "target_scaler": target_scaler,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare ozel rich data for diffusion")
    parser.add_argument("--mode", choices=["explore", "prepare"], default="explore")
    parser.add_argument("--output", type=str, default="data/ozel_rich/prepared.pt")
    args = parser.parse_args()

    df = load_data()

    if args.mode == "explore":
        explore_data(df)

    elif args.mode == "prepare":
        data = prepare_data(df)

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
            "cat_columns": data["cat_columns"],
            "feature_names": data["feature_names"],
        }, output_path)
        print(f"\nData saved to {output_path}")

        # Save preprocessors
        preprocessor_path = output_path.parent / "preprocessors.pkl"
        with open(preprocessor_path, "wb") as f:
            pickle.dump({
                "scaler": data["scaler"],
                "target_scaler": data["target_scaler"],
                "encoders": data["encoders"],
            }, f)
        print(f"Preprocessors saved to {preprocessor_path}")


if __name__ == "__main__":
    main()
