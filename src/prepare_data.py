"""
Prepare and save datasets for experiments.

This script saves the original datasets to the data/ folder,
ensuring consistent train/test splits across all experiments.

Usage:
    python src/prepare_data.py --dataset iris
    python src/prepare_data.py --dataset california
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split


def prepare_iris(output_dir: Path):
    """Prepare and save Iris dataset."""
    data = load_iris()
    X, y = data.data, data.target
    feature_names = [name.replace(' (cm)', '') for name in data.feature_names]

    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save training data
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['label'] = y_train
    df_train.to_csv(output_dir / "train.csv", index=False)

    # Save test data
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['label'] = y_test
    df_test.to_csv(output_dir / "test.csv", index=False)

    # Save metadata
    metadata = {
        'name': 'Iris',
        'source': 'sklearn.datasets.load_iris',
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'feature_names': feature_names,
        'target_names': list(data.target_names),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'random_state': 42,
    }

    with open(output_dir / "metadata.txt", 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    print(f"Iris dataset saved to {output_dir}/")
    print(f"  - train.csv: {len(X_train)} samples")
    print(f"  - test.csv: {len(X_test)} samples")
    print(f"  - metadata.txt")


def prepare_california(output_dir: Path):
    """Prepare and save California Housing dataset."""
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save training data
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['target'] = y_train
    df_train.to_csv(output_dir / "train.csv", index=False)

    # Save test data
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['target'] = y_test
    df_test.to_csv(output_dir / "test.csv", index=False)

    # Save metadata
    metadata = {
        'name': 'California Housing',
        'source': 'sklearn.datasets.fetch_california_housing',
        'n_samples': len(X),
        'n_features': X.shape[1],
        'task': 'regression',
        'feature_names': feature_names,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'random_state': 42,
    }

    with open(output_dir / "metadata.txt", 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    print(f"California Housing dataset saved to {output_dir}/")
    print(f"  - train.csv: {len(X_train)} samples")
    print(f"  - test.csv: {len(X_test)} samples")
    print(f"  - metadata.txt")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=["iris", "california", "all"])
    args = parser.parse_args()

    data_dir = Path("data")

    if args.dataset in ["iris", "all"]:
        iris_dir = data_dir / "iris"
        iris_dir.mkdir(parents=True, exist_ok=True)
        prepare_iris(iris_dir)

    if args.dataset in ["california", "all"]:
        california_dir = data_dir / "california"
        california_dir.mkdir(parents=True, exist_ok=True)
        prepare_california(california_dir)


if __name__ == "__main__":
    main()
