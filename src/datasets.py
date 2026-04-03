"""
Unified dataset loader for Phase 2 experiments.

Loads 10 public datasets with standardized format:
- Automatic download from sklearn/OpenML/UCI/Kaggle
- Consistent train/test split (80/20, stratified for classification)
- Column type identification (numerical vs categorical)
- Preprocessing: MinMaxScaler for numerical, LabelEncoder for categorical
- Saves/loads as .pt files for fast reuse

Usage:
    from datasets import load_dataset, list_datasets

    # List all available datasets
    list_datasets()

    # Load a dataset
    data = load_dataset("insurance")
    print(data["X_train"].shape, data["task_type"])
"""

import os
import json
import hashlib
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Dataset Registry
# =============================================================================

DATASET_REGISTRY = {
    "iris": {
        "id": "D1",
        "name": "Iris",
        "source": "sklearn",
        "task": "classification",
        "domain": "Botany",
        "description": "4 numeric features, 150 samples, 3 classes",
    },
    "california": {
        "id": "D2",
        "name": "California Housing",
        "source": "sklearn",
        "task": "regression",
        "domain": "Real Estate",
        "description": "8 numeric features, 20640 samples",
    },
    "insurance": {
        "id": "D3",
        "name": "Insurance Charges",
        "source": "kaggle_csv",
        "task": "regression",
        "domain": "Healthcare",
        "description": "4 num + 3 cat features, 1338 samples",
    },
    "maintenance": {
        "id": "D4",
        "name": "AI4I Predictive Maintenance",
        "source": "uci",
        "uci_id": 601,
        "task": "classification",
        "domain": "Manufacturing",
        "description": "6 num + 3 cat features, 10000 samples",
    },
    "steel": {
        "id": "D5",
        "name": "Steel Plates Faults",
        "source": "uci",
        "uci_id": 198,
        "task": "classification",
        "domain": "Manufacturing",
        "description": "27 num + 7 cat features, 1941 samples",
    },
    "bank": {
        "id": "D6",
        "name": "Bank Marketing",
        "source": "uci",
        "uci_id": 222,
        "task": "classification",
        "domain": "Finance",
        "description": "7 num + 9 cat features, 45211 samples",
    },
    "credit": {
        "id": "D7",
        "name": "Credit Default",
        "source": "uci",
        "uci_id": 350,
        "task": "classification",
        "domain": "Finance",
        "description": "14 num + 9 cat features, 30000 samples",
    },
    "supply_chain": {
        "id": "D8",
        "name": "Supply Chain Pricing",
        "source": "kaggle_csv",
        "task": "regression",
        "domain": "Manufacturing",
        "description": "6 num + 8 cat features, 10324 samples",
    },
    "news": {
        "id": "D9",
        "name": "Online News Popularity",
        "source": "uci",
        "uci_id": 332,
        "task": "regression",
        "domain": "Media",
        "description": "58 num + 2 cat features, 39644 samples",
    },
    "adult": {
        "id": "D10",
        "name": "Adult",
        "source": "openml",
        "task": "classification",
        "domain": "Census",
        "description": "6 num + 8 cat features, 48842 samples",
    },
}


# =============================================================================
# Data Directory
# =============================================================================

DATA_DIR = Path(__file__).parent.parent / "data"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# Individual Dataset Loaders (raw DataFrames)
# =============================================================================

def _load_iris_raw() -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    """Returns (df, target_col, num_cols, cat_cols)."""
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    df = data.frame
    target = "target"
    num_cols = [c for c in df.columns if c != target]
    cat_cols = []
    return df, target, num_cols, cat_cols


def _load_california_raw() -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    target = "MedHouseVal"
    num_cols = [c for c in df.columns if c != target]
    cat_cols = []
    return df, target, num_cols, cat_cols


def _load_insurance_raw() -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    """Insurance Charges dataset from Kaggle (CC0)."""
    dataset_dir = _ensure_dir(DATA_DIR / "insurance")
    csv_path = dataset_dir / "insurance.csv"

    if not csv_path.exists():
        # Download from GitHub mirror (original Kaggle, CC0 license)
        import urllib.request
        url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
        print(f"Downloading Insurance dataset to {csv_path}...")
        urllib.request.urlretrieve(url, csv_path)

    df = pd.read_csv(csv_path)
    target = "charges"
    cat_cols = ["sex", "smoker", "region"]
    num_cols = [c for c in df.columns if c not in cat_cols and c != target]
    return df, target, num_cols, cat_cols


def _load_maintenance_raw() -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    """AI4I 2020 Predictive Maintenance (UCI ID 601)."""
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=601)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    target = "Machine failure"
    # Drop UDI and Product ID (identifiers, not features)
    drop_cols = ["UDI", "Product ID"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Type column is categorical
    cat_cols = ["Type"]
    # Failure mode columns are binary categorical
    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    cat_cols += [c for c in failure_modes if c in df.columns]
    num_cols = [c for c in df.columns if c not in cat_cols and c != target]
    return df, target, num_cols, cat_cols


def _load_steel_raw() -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    """Steel Plates Faults (UCI ID 198)."""
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=198)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    # The target is a multi-class fault type (7 binary columns in original)
    # Convert 7 binary fault columns to single target
    fault_cols = [c for c in dataset.data.targets.columns]
    if len(fault_cols) > 1:
        # Multi-label to single label
        df["fault_type"] = dataset.data.targets.values.argmax(axis=1)
        df = df.drop(columns=fault_cols)
        target = "fault_type"
    else:
        target = fault_cols[0]

    # Identify binary/categorical columns (columns with few unique values)
    cat_cols = []
    num_cols = []
    for c in df.columns:
        if c == target:
            continue
        if df[c].nunique() <= 10:
            cat_cols.append(c)
        else:
            num_cols.append(c)

    return df, target, num_cols, cat_cols


def _load_bank_raw() -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    """Bank Marketing (UCI ID 222)."""
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=222)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    target = "y"
    # Encode target: yes/no -> 1/0
    if df[target].dtype == object:
        df[target] = (df[target] == "yes").astype(int)

    cat_cols = []
    num_cols = []
    for c in df.columns:
        if c == target:
            continue
        if df[c].dtype == object or df[c].nunique() <= 10:
            cat_cols.append(c)
        else:
            num_cols.append(c)

    return df, target, num_cols, cat_cols


def _load_credit_raw() -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    """Default of Credit Card Clients (UCI ID 350)."""
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=350)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    target = "default payment next month" if "default payment next month" in df.columns else df.columns[-1]

    # UCI version uses X1-X23 naming. Map known categoricals:
    # X2=SEX, X3=EDUCATION, X4=MARRIAGE, X6-X11=PAY_0..PAY_5
    # Named version: SEX, EDUCATION, MARRIAGE, PAY_0..PAY_6
    known_cat_named = ["SEX", "EDUCATION", "MARRIAGE"]
    known_cat_x = ["X2", "X3", "X4", "X6", "X7", "X8", "X9", "X10", "X11"]  # UCI generic names

    cat_cols = []
    for c in df.columns:
        if c == target or c.upper() == "ID":
            continue
        c_up = c.upper()
        if c in known_cat_named or c in known_cat_x:
            cat_cols.append(c)
        elif c_up.startswith("PAY") and "AMT" not in c_up:
            cat_cols.append(c)
    num_cols = [c for c in df.columns if c not in cat_cols and c != target and c.upper() != "ID"]

    # Drop ID if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    return df, target, num_cols, cat_cols


def _load_supply_chain_raw() -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    """Supply Chain Shipment Pricing (Kaggle, CC0)."""
    dataset_dir = _ensure_dir(DATA_DIR / "supply_chain")
    csv_path = dataset_dir / "supply_chain.csv"

    if not csv_path.exists():
        import urllib.request
        url = "https://raw.githubusercontent.com/jrcinco/supply-chain-shipment-price-data/master/SCMS_Delivery_History_Dataset.csv"
        print(f"Downloading Supply Chain dataset to {csv_path}...")
        urllib.request.urlretrieve(url, csv_path)

    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Target: freight cost or line item value
    target_candidates = ["Freight Cost (USD)", "Line Item Value", "Weight (Kilograms)"]
    target = None
    for tc in target_candidates:
        if tc in df.columns:
            target = tc
            break
    if target is None:
        # Fallback: use last numeric column
        num_candidates = df.select_dtypes(include=[np.number]).columns
        target = num_candidates[-1] if len(num_candidates) > 0 else df.columns[-1]

    cat_cols = []
    num_cols = []
    # Drop ID-like and date columns
    drop_patterns = ["ID", "id", "Date", "date", "PQ #", "PO #", "ASN/DN #"]
    keep_cols = [c for c in df.columns if c != target and not any(p in c for p in drop_patterns)]

    for c in keep_cols:
        if df[c].dtype == object or df[c].nunique() <= 20:
            cat_cols.append(c)
        else:
            num_cols.append(c)

    # Convert target to numeric — drop non-numeric rows (e.g., "Freight Included in Commodity Cost")
    if df[target].dtype == object:
        df[target] = pd.to_numeric(df[target].astype(str).str.replace(",", ""), errors="coerce")
    df = df.dropna(subset=[target])

    # Also convert numeric columns that may have mixed types
    for c in list(num_cols):
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")

    return df, target, num_cols, cat_cols


def _load_news_raw() -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    """Online News Popularity (UCI ID 332)."""
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=332)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    target = "shares" if "shares" in df.columns else df.columns[-1]

    # Remove non-predictive columns
    drop_cols = ["url", "timedelta"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Binary columns (is_weekend, data_channel_is_*) are categorical
    cat_cols = []
    num_cols = []
    for c in df.columns:
        if c == target:
            continue
        if df[c].nunique() == 2 and set(df[c].unique()).issubset({0, 1, 0.0, 1.0}):
            cat_cols.append(c)
        else:
            num_cols.append(c)

    return df, target, num_cols, cat_cols


def _load_adult_raw() -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    """Adult Census Income (OpenML)."""
    from sklearn.datasets import fetch_openml
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.frame

    target = "income" if "income" in df.columns else df.columns[-1]

    # Encode target
    if df[target].dtype == object or df[target].dtype.name == "category":
        df[target] = (df[target].astype(str).str.strip().str.contains(">50K")).astype(int)

    cat_cols = []
    num_cols = []
    for c in df.columns:
        if c == target:
            continue
        if df[c].dtype == object or df[c].dtype.name == "category":
            cat_cols.append(c)
        else:
            num_cols.append(c)

    return df, target, num_cols, cat_cols


# Loader dispatch
_LOADERS = {
    "iris": _load_iris_raw,
    "california": _load_california_raw,
    "insurance": _load_insurance_raw,
    "maintenance": _load_maintenance_raw,
    "steel": _load_steel_raw,
    "bank": _load_bank_raw,
    "credit": _load_credit_raw,
    "supply_chain": _load_supply_chain_raw,
    "news": _load_news_raw,
    "adult": _load_adult_raw,
}


# =============================================================================
# Preprocessing
# =============================================================================

def preprocess_dataset(
    df: pd.DataFrame,
    target: str,
    num_cols: List[str],
    cat_cols: List[str],
    task: str,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler_type: str = "minmax",
    outlier_clip: bool = True,
) -> Dict[str, Any]:
    """
    Preprocess a dataset into train/test tensors.

    Args:
        scaler_type: "minmax" (our approach) or "quantile" (vanilla TabDDPM)
        outlier_clip: If True, clip numerical outliers to 1st-99th percentile (our approach)

    Returns dict with:
        X_num_train, X_num_test: Scaled numerical features (torch tensors)
        X_cat_train, X_cat_test: Label-encoded categorical indices (torch tensors)
        y_train, y_test: Target values (torch tensors)
        num_cols, cat_cols: Column names
        cat_cardinalities: Number of unique values per categorical feature
        task_type: "regression" or "classification"
        scaler: Fitted scaler for numerical features
        label_encoders: Fitted label encoders for categorical features
        target_encoder: Fitted encoder for target (classification) or scaler (regression)
        n_train, n_test: Sample counts
        d_numerical, d_categorical: Feature counts
    """
    df = df.copy()

    # Convert target to numeric for regression tasks (handles mixed-type targets)
    if task == "regression":
        df[target] = pd.to_numeric(df[target].astype(str).str.replace(",", ""), errors="coerce")

    # Drop rows with missing target
    df = df.dropna(subset=[target])

    # Handle missing values in features
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if c in df.columns:
            # Convert categorical dtype to string first to avoid setitem errors
            if hasattr(df[c], "cat"):
                df[c] = df[c].astype(str)
            df[c] = df[c].fillna("_missing_").astype(str)

    # Filter to only existing columns
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]

    # Extract features and target
    X_num = df[num_cols].values.astype(np.float32) if num_cols else np.empty((len(df), 0), dtype=np.float32)
    X_cat_raw = df[cat_cols] if cat_cols else pd.DataFrame()
    y = df[target].values

    # Train/test split
    stratify = y if task == "classification" and len(np.unique(y)) <= 50 else None
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=stratify
    )

    X_num_train, X_num_test = X_num[train_idx], X_num[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Encode categoricals
    label_encoders = {}
    cat_encoded_train = []
    cat_encoded_test = []
    cat_cardinalities = []

    for c in cat_cols:
        le = LabelEncoder()
        col_train = X_cat_raw.iloc[train_idx][c].values
        col_test = X_cat_raw.iloc[test_idx][c].values

        # Fit on all data to handle unseen categories in test
        le.fit(np.concatenate([col_train, col_test]))
        cat_encoded_train.append(le.transform(col_train))
        cat_encoded_test.append(le.transform(col_test))
        cat_cardinalities.append(len(le.classes_))
        label_encoders[c] = le

    X_cat_train = np.column_stack(cat_encoded_train).astype(np.int64) if cat_cols else np.empty((len(train_idx), 0), dtype=np.int64)
    X_cat_test = np.column_stack(cat_encoded_test).astype(np.int64) if cat_cols else np.empty((len(test_idx), 0), dtype=np.int64)

    # Scale numerical features
    if num_cols:
        if outlier_clip:
            # Our approach: clip to 1st-99th percentile before scaling
            p1 = np.percentile(X_num_train, 1, axis=0)
            p99 = np.percentile(X_num_train, 99, axis=0)
            X_num_train = np.clip(X_num_train, p1, p99)
            X_num_test = np.clip(X_num_test, p1, p99)

        if scaler_type == "minmax":
            scaler = MinMaxScaler(feature_range=(-1, 1))
        elif scaler_type == "quantile":
            scaler = QuantileTransformer(output_distribution="normal", random_state=random_state)
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")

        X_num_train = scaler.fit_transform(X_num_train)
        X_num_test = scaler.transform(X_num_test)
    else:
        scaler = None

    # Encode target
    if task == "classification":
        target_encoder = LabelEncoder()
        y_train = target_encoder.fit_transform(y_train.astype(str))
        y_test = target_encoder.transform(y_test.astype(str))
    else:
        target_encoder = None
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

    return {
        "X_num_train": torch.tensor(X_num_train, dtype=torch.float32),
        "X_num_test": torch.tensor(X_num_test, dtype=torch.float32),
        "X_cat_train": torch.tensor(X_cat_train, dtype=torch.long),
        "X_cat_test": torch.tensor(X_cat_test, dtype=torch.long),
        "y_train": torch.tensor(y_train, dtype=torch.long if task == "classification" else torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.long if task == "classification" else torch.float32),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_cardinalities": cat_cardinalities,
        "task_type": task,
        "scaler": scaler,
        "scaler_type": scaler_type,
        "label_encoders": label_encoders,
        "target_encoder": target_encoder,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "d_numerical": len(num_cols),
        "d_categorical": len(cat_cols),
        "d_onehot": sum(cat_cardinalities),
    }


# =============================================================================
# Public API
# =============================================================================

def list_datasets() -> None:
    """Print all available datasets."""
    print(f"\n{'ID':<5} {'Key':<15} {'Name':<30} {'Task':<15} {'Domain':<15}")
    print("=" * 80)
    for key, info in DATASET_REGISTRY.items():
        print(f"{info['id']:<5} {key:<15} {info['name']:<30} {info['task']:<15} {info['domain']:<15}")
    print()


def get_dataset_info(name: str) -> Dict:
    """Get metadata for a dataset."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]


def load_dataset(
    name: str,
    scaler_type: str = "minmax",
    outlier_clip: bool = True,
    cache: bool = True,
    force_reload: bool = False,
) -> Dict[str, Any]:
    """
    Load and preprocess a dataset.

    Args:
        name: Dataset key (e.g., "iris", "insurance", "adult")
        scaler_type: "minmax" (our approach) or "quantile" (vanilla TabDDPM)
        outlier_clip: Whether to clip outliers (our approach)
        cache: Whether to cache preprocessed data as .pt files
        force_reload: Force re-download and re-preprocessing

    Returns:
        Dict with train/test tensors, metadata, and fitted preprocessors
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")

    info = DATASET_REGISTRY[name]
    cache_dir = _ensure_dir(DATA_DIR / name)
    cache_key = f"{scaler_type}_{'clip' if outlier_clip else 'noclip'}"
    cache_path = cache_dir / f"prepared_{cache_key}.pt"

    if cache and cache_path.exists() and not force_reload:
        print(f"Loading cached {info['name']} from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    print(f"Loading {info['name']} ({info['id']})...")
    loader = _LOADERS[name]
    df, target, num_cols, cat_cols = loader()

    print(f"  Raw: {len(df)} samples, {len(num_cols)} numerical, {len(cat_cols)} categorical")

    data = preprocess_dataset(
        df=df,
        target=target,
        num_cols=num_cols,
        cat_cols=cat_cols,
        task=info["task"],
        scaler_type=scaler_type,
        outlier_clip=outlier_clip,
    )

    # Add metadata
    data["dataset_name"] = name
    data["dataset_info"] = info

    print(f"  Preprocessed: {data['n_train']} train, {data['n_test']} test")
    print(f"  Dimensions: {data['d_numerical']} num + {data['d_categorical']} cat = {data['d_onehot'] + data['d_numerical']} total")

    if cache:
        torch.save(data, cache_path)
        print(f"  Cached to {cache_path}")

    return data


def load_all_datasets(
    scaler_type: str = "minmax",
    outlier_clip: bool = True,
    **kwargs,
) -> Dict[str, Dict]:
    """Load all 10 datasets. Returns dict keyed by dataset name."""
    results = {}
    for name in DATASET_REGISTRY:
        try:
            results[name] = load_dataset(name, scaler_type=scaler_type, outlier_clip=outlier_clip, **kwargs)
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dataset loader for Phase 2 experiments")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to load (or 'all')")
    parser.add_argument("--scaler", type=str, default="minmax", choices=["minmax", "quantile"])
    parser.add_argument("--no-clip", action="store_true", help="Disable outlier clipping")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    if args.list:
        list_datasets()
    elif args.dataset == "all":
        data = load_all_datasets(
            scaler_type=args.scaler,
            outlier_clip=not args.no_clip,
            force_reload=args.force,
        )
        print(f"\nLoaded {len(data)} datasets successfully.")
        for name, d in data.items():
            print(f"  {d['dataset_info']['id']} {name}: {d['n_train']}+{d['n_test']} samples, "
                  f"{d['d_numerical']}num+{d['d_categorical']}cat={d['d_onehot']+d['d_numerical']}dims")
    elif args.dataset:
        data = load_dataset(
            args.dataset,
            scaler_type=args.scaler,
            outlier_clip=not args.no_clip,
            force_reload=args.force,
        )
        print(f"\nDataset: {data['dataset_info']['name']}")
        print(f"  Task: {data['task_type']}")
        print(f"  Train: {data['n_train']}, Test: {data['n_test']}")
        print(f"  Numerical: {data['d_numerical']} cols, Categorical: {data['d_categorical']} cols")
        print(f"  Total dims (with one-hot): {data['d_onehot'] + data['d_numerical']}")
        print(f"  Cat cardinalities: {data['cat_cardinalities']}")
    else:
        list_datasets()
