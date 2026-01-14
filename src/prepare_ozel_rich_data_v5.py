"""
Prepare ozel data for HybridDiffusion - V5 (MinMax scaling).

Key insight from V3/V4 failures:
- QuantileTransformer outputs unbounded Gaussian
- Diffusion needs bounded input [-1, 1] for stable sampling
- These are incompatible → use MinMax instead

V5 approach:
- MinMax scale to [-1, 1] (no quantile transform)
- Simple linear inverse transform (no distribution mismatch)
- Should fix the boundary collapse / explosion issues
"""

import argparse
import pickle
import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/manufacturing/Teklif Maliyet SSEK data.xlsx")
TARGET_COL = "Makine Süre (100.000 ADET) DK"


def load_data():
    """Load manufacturing data and filter to ozel rows."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    print(f"Total rows: {len(df)}")

    ozel_mask = df['Malzeme Tanımı'].str.lower().str.contains('özel|ozel', na=False)
    df_ozel = df[ozel_mask].copy()
    print(f"Ozel rows: {len(df_ozel)}")

    return df_ozel


def extract_dimensions(text):
    """Extract Cap and Boy from Malzeme Tanımı."""
    text = str(text)
    match = re.match(r'M(\d+(?:[.,]\d+)?)[xX](\d+(?:[.,]\d+)?)', text)
    if match:
        cap = float(match.group(1).replace(',', '.'))
        boy = float(match.group(2).replace(',', '.'))
        return cap, boy
    return None, None


def extract_islem_tipi(text):
    """Extract IslemTipi (FT, HT, or Belirsiz)."""
    text = str(text).upper()
    if ' FT ' in text or text.endswith(' FT'):
        return 'FT'
    elif ' HT ' in text or text.endswith(' HT'):
        return 'HT'
    else:
        return 'Belirsiz'


def simplify_anma_olcusu(value):
    value = str(value).strip()
    if value.startswith('M') and value[1:].isdigit():
        return value
    elif value.startswith('M') and '.' not in value:
        return value
    else:
        return 'Other'


def simplify_sartname(value):
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


def prepare_data(df, test_size=0.2):
    """
    Prepare data with MinMax scaling to [-1, 1].

    This avoids the QuantileTransformer/diffusion incompatibility.
    """
    print("\n" + "=" * 80)
    print("V5: MINMAX SCALING TO [-1, 1]")
    print("=" * 80)

    # Extract features
    df['Cap'], df['Boy'] = zip(*df['Malzeme Tanımı'].apply(extract_dimensions))
    df['IslemTipi'] = df['Malzeme Tanımı'].apply(extract_islem_tipi)
    df['AnmaOlcusu'] = df['Anma ölçüsü'].apply(simplify_anma_olcusu)
    df['SartnameSimple'] = df['Sartname'].apply(simplify_sartname)
    df['UY'] = df['ÜY'].astype(str)

    df_valid = df.dropna(subset=['Cap', 'Boy', TARGET_COL]).copy()
    print(f"Valid rows: {len(df_valid)}")

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
        print(f"{col}: {len(le.classes_)} categories")

    cat_indices = np.column_stack(cat_indices)

    # Train/test split
    (X_num_train, X_num_test, y_train, y_test,
     cat_train, cat_test) = train_test_split(
        X_num, y, cat_indices, test_size=test_size, random_state=42
    )

    print(f"\nTrain: {len(X_num_train)}, Test: {len(X_num_test)}")

    # Scale numeric features with MinMax to [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)

    # Scale target with MinMax to [-1, 1]
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # Print stats
    all_num = np.column_stack([X_num_train_scaled, y_train_scaled.reshape(-1, 1)])
    print(f"\nScaled numeric stats (should be in [-1, 1]):")
    print(f"  Mean: {all_num.mean(axis=0)}")
    print(f"  Std:  {all_num.std(axis=0)}")
    print(f"  Min:  {all_num.min(axis=0)}")
    print(f"  Max:  {all_num.max(axis=0)}")

    # Combine: [Cap, Boy, target]
    X_train_full = np.column_stack([
        X_num_train_scaled,
        y_train_scaled.reshape(-1, 1),
    ]).astype(np.float32)

    X_test_full = np.column_stack([
        X_num_test_scaled,
        y_test_scaled.reshape(-1, 1),
    ]).astype(np.float32)

    num_numerical = 3

    return {
        "X_train_num": X_train_full,
        "X_test_num": X_test_full,
        "X_train_num_unscaled": X_num_train,
        "X_test_num_unscaled": X_num_test,
        "y_train": y_train,
        "y_test": y_test,
        "cat_train": cat_train,
        "cat_test": cat_test,
        "num_numerical": num_numerical,
        "cat_cardinalities": cat_cardinalities,
        "cat_columns": cat_columns,
        "encoders": encoders,
        "scaler": scaler,
        "target_scaler": target_scaler,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/ozel_rich_v5/prepared.pt")
    args = parser.parse_args()

    df = load_data()
    data = prepare_data(df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "X_train_num": torch.tensor(data["X_train_num"]),
        "X_test_num": torch.tensor(data["X_test_num"]),
        "X_train_num_unscaled": data["X_train_num_unscaled"],
        "X_test_num_unscaled": data["X_test_num_unscaled"],
        "y_train": data["y_train"],
        "y_test": data["y_test"],
        "cat_train": data["cat_train"],
        "cat_test": data["cat_test"],
        "num_numerical": data["num_numerical"],
        "cat_cardinalities": data["cat_cardinalities"],
        "cat_columns": data["cat_columns"],
    }, output_path)
    print(f"\nData saved to {output_path}")

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
