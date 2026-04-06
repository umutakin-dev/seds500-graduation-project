"""
Unified dataset loader for Phase 2 experiments.

Each dataset has an explicit configuration defining:
- Which columns are numerical
- Which columns are categorical
- What the target is
- Max cardinality caps where needed

No heuristic-based column type detection. Every column assignment is intentional.

Usage:
    from datasets import load_dataset, list_datasets

    list_datasets()
    data = load_dataset("insurance")
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path(__file__).parent.parent / "data"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# Dataset Registry — explicit column configs
# =============================================================================

DATASET_REGISTRY = {
    "iris": {
        "id": "D1",
        "name": "Iris",
        "task": "classification",
        "domain": "Botany",
        "description": "4 numeric, 150 samples, 3 classes",
        "target": "target",
        "num_cols": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
        "cat_cols": [],
    },
    "california": {
        "id": "D2",
        "name": "California Housing",
        "task": "regression",
        "domain": "Real Estate",
        "description": "8 numeric, 20640 samples",
        "target": "MedHouseVal",
        "num_cols": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"],
        "cat_cols": [],
    },
    "insurance": {
        "id": "D3",
        "name": "Insurance Charges",
        "task": "regression",
        "domain": "Healthcare",
        "description": "3 num + 3 cat, 1338 samples",
        "target": "charges",
        "num_cols": ["age", "bmi", "children"],
        "cat_cols": ["sex", "smoker", "region"],
    },
    "maintenance": {
        "id": "D4",
        "name": "AI4I Predictive Maintenance",
        "task": "classification",
        "domain": "Manufacturing",
        "description": "5 num + 6 cat, 10000 samples",
        "target": "Machine failure",
        "num_cols": ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"],
        "cat_cols": ["Type", "TWF", "HDF", "PWF", "OSF", "RNF"],
        "drop_cols": ["UDI", "Product ID"],
    },
    "steel": {
        "id": "D5",
        "name": "Steel Plates Faults",
        "task": "classification",
        "domain": "Manufacturing",
        "description": "24 num + 3 cat, 1941 samples",
        "target": "fault_type",  # derived from 7 binary fault columns
        "num_cols": [
            "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas",
            "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity", "Minimum_of_Luminosity",
            "Maximum_of_Luminosity", "Length_of_Conveyer", "Steel_Plate_Thickness",
            "Edges_Index", "Empty_Index", "Square_Index", "Outside_X_Index",
            "Edges_X_Index", "Edges_Y_Index", "LogOfAreas", "Log_X_Index",
            "Log_Y_Index", "Orientation_Index", "Luminosity_Index", "SigmoidOfAreas",
        ],
        "cat_cols": ["TypeOfSteel_A300", "TypeOfSteel_A400", "Outside_Global_Index"],
    },
    "bank": {
        "id": "D6",
        "name": "Bank Marketing",
        "task": "classification",
        "domain": "Finance",
        "description": "7 num + 9 cat, 45211 samples",
        "target": "y",
        "num_cols": ["age", "balance", "duration", "campaign", "pdays", "previous", "day_of_week"],
        "cat_cols": ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"],
    },
    "credit": {
        "id": "D7",
        "name": "Credit Default",
        "task": "classification",
        "domain": "Finance",
        "description": "14 num + 9 cat, 30000 samples",
        "target": "Y",
        "num_cols": ["X1", "X5", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23"],
        "cat_cols": ["X2", "X3", "X4", "X6", "X7", "X8", "X9", "X10", "X11"],
        "drop_cols": ["ID"],
    },
    "supply_chain": {
        "id": "D8",
        "name": "Supply Chain Pricing",
        "task": "regression",
        "domain": "Manufacturing",
        "description": "5 num + 11 cat, ~10000 samples",
        "target": "Freight Cost (USD)",
        "num_cols": [
            "Line Item Quantity", "Line Item Value", "Pack Price", "Unit Price",
            "Line Item Insurance (USD)",
        ],
        "cat_cols": [
            "Country", "Managed By", "Fulfill Via", "Vendor INCO Term",
            "Shipment Mode", "Product Group", "Sub Classification",
            "Dosage Form", "First Line Designation", "Manufacturing Site",
            "Brand",
        ],
        "cat_max_cardinality": {
            "Manufacturing Site": 20,  # 88 unique factory names → top 20 + Other
            "Brand": 20,  # 48 brands but 71% is "Generic" → top 20 + Other
        },
        "drop_cols": ["ID", "Project Code", "PQ #", "PO / SO #", "ASN/DN #",
                       "Scheduled Delivery Date", "Delivered to Client Date",
                       "Delivery Recorded Date", "Item Description",
                       "Molecule/Test Type", "Dosage", "Vendor",
                       "Unit of Measure (Per Pack)", "Weight (Kilograms)"],
    },
    "news": {
        "id": "D9",
        "name": "Online News Popularity",
        "task": "regression",
        "domain": "Media",
        "description": "44 num + 14 cat (binary), 39644 samples",
        "target": " shares",  # note: leading space from UCI
        "num_cols": "auto_non_binary",  # all non-binary numeric columns
        "cat_cols": "auto_binary",  # all binary (0/1) columns
    },
    "adult": {
        "id": "D10",
        "name": "Adult",
        "task": "classification",
        "domain": "Census",
        "description": "6 num + 8 cat, 48842 samples",
        "target": "income",
        "num_cols": ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"],
        "cat_cols": ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"],
    },
    "ames": {
        "id": "D11",
        "name": "Ames Housing",
        "task": "regression",
        "domain": "Real Estate",
        "description": "~20 num + ~46 cat, 2930 samples, ~250 one-hot dims",
        "target": "SalePrice",
        "num_cols": [
            # Continuous measurements
            "Lot Frontage", "Lot Area", "Mas Vnr Area", "BsmtFin SF 1",
            "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF", "1st Flr SF",
            "2nd Flr SF", "Low Qual Fin SF", "Gr Liv Area", "Garage Area",
            "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch",
            "Screen Porch", "Pool Area", "Misc Val",
            # Year/date features — ordinal, not categorical
            "Year Built", "Year Remod/Add", "Garage Yr Blt", "Yr Sold",
            # Counts — ordinal numeric
            "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath", "Half Bath",
            "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd", "Fireplaces",
            "Garage Cars",
        ],
        "cat_cols": [
            # Building classification
            "MS SubClass", "MS Zoning", "Street", "Alley", "Lot Shape",
            "Land Contour", "Utilities", "Lot Config", "Land Slope",
            "Neighborhood", "Condition 1", "Condition 2", "Bldg Type",
            "House Style",
            # Quality/condition ratings (ordinal but categorical semantics)
            "Overall Qual", "Overall Cond",
            "Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond",
            "Heating QC", "Kitchen Qual", "Garage Qual", "Garage Cond",
            "Pool QC", "Fireplace Qu",
            # Material/type features
            "Roof Style", "Roof Matl", "Exterior 1st", "Exterior 2nd",
            "Mas Vnr Type", "Foundation", "Bsmt Exposure",
            "BsmtFin Type 1", "BsmtFin Type 2",
            "Heating", "Central Air", "Electrical",
            "Garage Type", "Garage Finish", "Paved Drive",
            "Fence", "Misc Feature", "Functional",
            # Sale info
            "Mo Sold", "Sale Type", "Sale Condition",
        ],
        "drop_cols": ["Order", "PID"],
    },
}


# =============================================================================
# Raw Data Fetchers — download/load only, no column decisions
# =============================================================================

def _fetch_iris() -> pd.DataFrame:
    from sklearn.datasets import load_iris
    return load_iris(as_frame=True).frame

def _fetch_california() -> pd.DataFrame:
    from sklearn.datasets import fetch_california_housing
    return fetch_california_housing(as_frame=True).frame

def _fetch_insurance() -> pd.DataFrame:
    csv_path = _ensure_dir(DATA_DIR / "insurance") / "insurance.csv"
    if not csv_path.exists():
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv",
            csv_path,
        )
    return pd.read_csv(csv_path)

def _fetch_maintenance() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=601)
    return pd.concat([ds.data.features, ds.data.targets], axis=1)

def _fetch_steel() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=198)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    # Convert 7 binary fault columns → single fault_type
    fault_cols = list(ds.data.targets.columns)
    if len(fault_cols) > 1:
        df["fault_type"] = ds.data.targets.values.argmax(axis=1)
        df = df.drop(columns=fault_cols)
    return df

def _fetch_bank() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=222)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    if df["y"].dtype == object:
        df["y"] = (df["y"] == "yes").astype(int)
    return df

def _fetch_credit() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=350)
    return pd.concat([ds.data.features, ds.data.targets], axis=1)

def _fetch_supply_chain() -> pd.DataFrame:
    csv_path = _ensure_dir(DATA_DIR / "supply_chain") / "supply_chain.csv"
    if not csv_path.exists():
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/jrcinco/supply-chain-shipment-price-data/master/SCMS_Delivery_History_Dataset.csv",
            csv_path,
        )
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df

def _fetch_news() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=332)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    drop_cols = ["url", "timedelta", " url", " timedelta"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return df

def _fetch_adult() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.frame
    # Rename target if needed
    target_col = [c for c in df.columns if c.lower() in ("income", "class", "target")]
    if target_col and target_col[0] != "income":
        df = df.rename(columns={target_col[0]: "income"})
    # Encode target
    if "income" in df.columns:
        df["income"] = (df["income"].astype(str).str.strip().str.contains(">50K")).astype(int)
    return df

def _fetch_ames() -> pd.DataFrame:
    csv_path = _ensure_dir(DATA_DIR / "ames") / "ames.csv"
    if not csv_path.exists():
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/STATCowboy/pbidataflowstalk/master/AmesHousing.csv",
            csv_path,
        )
    return pd.read_csv(csv_path)


_FETCHERS = {
    "iris": _fetch_iris,
    "california": _fetch_california,
    "insurance": _fetch_insurance,
    "maintenance": _fetch_maintenance,
    "steel": _fetch_steel,
    "bank": _fetch_bank,
    "credit": _fetch_credit,
    "supply_chain": _fetch_supply_chain,
    "news": _fetch_news,
    "adult": _fetch_adult,
    "ames": _fetch_ames,
}


# =============================================================================
# Column Resolution — apply explicit config to raw DataFrame
# =============================================================================

def _resolve_columns(df: pd.DataFrame, config: dict) -> Tuple[str, List[str], List[str]]:
    """
    Resolve target, numerical, and categorical columns from explicit config.
    Handles 'auto_binary' and 'auto_non_binary' for News dataset.
    """
    target = config["target"]

    # Find target column (handle case variations)
    if target not in df.columns:
        matches = [c for c in df.columns if c.strip() == target.strip()]
        if matches:
            target = matches[0]
        else:
            raise ValueError(f"Target '{target}' not found. Available: {list(df.columns)}")

    # Drop specified columns
    drop_cols = config.get("drop_cols", [])
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Resolve numerical columns
    num_cols_cfg = config["num_cols"]
    if num_cols_cfg == "auto_non_binary":
        # News dataset: all non-binary numeric columns
        num_cols = []
        for c in df.columns:
            if c == target:
                continue
            if df[c].dtype in (np.float64, np.int64, float, int):
                unique = set(df[c].dropna().unique())
                if not unique.issubset({0, 1, 0.0, 1.0}):
                    num_cols.append(c)
    else:
        num_cols = [c for c in num_cols_cfg if c in df.columns]

    # Resolve categorical columns
    cat_cols_cfg = config["cat_cols"]
    if cat_cols_cfg == "auto_binary":
        # News dataset: all binary (0/1) columns
        cat_cols = []
        for c in df.columns:
            if c == target or c in num_cols:
                continue
            if df[c].dtype in (np.float64, np.int64, float, int):
                unique = set(df[c].dropna().unique())
                if unique.issubset({0, 1, 0.0, 1.0}):
                    cat_cols.append(c)
    else:
        cat_cols = [c for c in cat_cols_cfg if c in df.columns]

    return target, num_cols, cat_cols, df


# =============================================================================
# Preprocessing
# =============================================================================

def preprocess_dataset(
    df: pd.DataFrame,
    target: str,
    num_cols: List[str],
    cat_cols: List[str],
    task: str,
    cat_max_cardinality: Optional[Dict[str, int]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler_type: str = "minmax",
    outlier_clip: bool = True,
) -> Dict[str, Any]:
    """Preprocess dataset into train/test tensors with explicit column config."""
    df = df.copy()

    # Convert target to numeric for regression
    if task == "regression":
        df[target] = pd.to_numeric(df[target].astype(str).str.replace(",", ""), errors="coerce")

    df = df.dropna(subset=[target])

    # Handle missing numerical values
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())

    # Handle missing categorical values + convert to string
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("_missing_")

    # Cap high-cardinality categoricals per config
    if cat_max_cardinality:
        for c, max_card in cat_max_cardinality.items():
            if c in cat_cols and c in df.columns:
                if df[c].nunique() > max_card:
                    top_values = df[c].value_counts().head(max_card).index.tolist()
                    df[c] = df[c].where(df[c].isin(top_values), other="_other_")

    # Filter to existing columns only
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]

    # Extract arrays
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
    print(f"\n{'ID':<5} {'Key':<15} {'Name':<30} {'Task':<15} {'Domain':<15}")
    print("=" * 80)
    for key, info in DATASET_REGISTRY.items():
        print(f"{info['id']:<5} {key:<15} {info['name']:<30} {info['task']:<15} {info['domain']:<15}")
    print()


def get_dataset_info(name: str) -> Dict:
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
    """Load and preprocess a dataset using explicit column configuration."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")

    config = DATASET_REGISTRY[name]
    cache_dir = _ensure_dir(DATA_DIR / name)
    cache_key = f"{scaler_type}_{'clip' if outlier_clip else 'noclip'}"
    cache_path = cache_dir / f"prepared_{cache_key}.pt"

    if cache and cache_path.exists() and not force_reload:
        print(f"Loading cached {config['name']} from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    print(f"Loading {config['name']} ({config['id']})...")

    # Fetch raw data
    df = _FETCHERS[name]()

    # Resolve columns from config
    target, num_cols, cat_cols, df = _resolve_columns(df, config)

    print(f"  Raw: {len(df)} samples, {len(num_cols)} numerical, {len(cat_cols)} categorical")

    # Preprocess
    data = preprocess_dataset(
        df=df,
        target=target,
        num_cols=num_cols,
        cat_cols=cat_cols,
        task=config["task"],
        cat_max_cardinality=config.get("cat_max_cardinality"),
        scaler_type=scaler_type,
        outlier_clip=outlier_clip,
    )

    data["dataset_name"] = name
    data["dataset_info"] = config

    print(f"  Preprocessed: {data['n_train']} train, {data['n_test']} test")
    print(f"  Dimensions: {data['d_numerical']} num + {data['d_categorical']} cat = {data['d_onehot'] + data['d_numerical']} total")

    if cache:
        torch.save(data, cache_path)
        print(f"  Cached to {cache_path}")

    return data


def load_all_datasets(scaler_type: str = "minmax", outlier_clip: bool = True, **kwargs) -> Dict[str, Dict]:
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
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_datasets()
    elif args.dataset == "all":
        data = load_all_datasets(scaler_type=args.scaler, outlier_clip=not args.no_clip, force_reload=args.force)
        print(f"\nLoaded {len(data)} datasets:")
        for name, d in data.items():
            total = d['d_numerical'] + d['d_onehot']
            print(f"  {d['dataset_info']['id']} {name}: {d['n_train']}+{d['n_test']} samples, "
                  f"{d['d_numerical']}num+{d['d_categorical']}cat={total}dims")
    elif args.dataset:
        data = load_dataset(args.dataset, scaler_type=args.scaler, outlier_clip=not args.no_clip, force_reload=args.force)
        total = data['d_numerical'] + data['d_onehot']
        print(f"\n{data['dataset_info']['name']}: {data['n_train']}+{data['n_test']} samples, "
              f"{data['d_numerical']}num+{data['d_categorical']}cat={total}dims, "
              f"cardinalities={data['cat_cardinalities']}")
    else:
        list_datasets()
