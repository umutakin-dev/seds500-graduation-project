"""
Evaluate baseline ML models on ozel rich dataset (from manufacturing data).

Uses Cap, Boy (numeric) and IslemTipi, AnmaOlcusu, SartnameSimple, UY (categorical).
This is Experiment 013 baseline.

Usage:
    python src/evaluate_ozel_rich_baseline.py
"""

import pickle
import torch
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data(data_path="data/ozel_rich/prepared.pt"):
    """Load prepared ozel rich data."""
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    return data


def evaluate_regressor(name, model, X_train, y_train, X_test, y_test):
    """Train and evaluate a regressor."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {"rmse": rmse, "mae": mae, "r2": r2, "y_pred": y_pred}


def main():
    # Load data
    data = load_data()

    # Get unscaled numeric features and categorical indices
    X_train_num = data["X_train_num_unscaled"]  # [Cap, Boy]
    X_test_num = data["X_test_num_unscaled"]
    cat_train = data["cat_train"]  # [IslemTipi, AnmaOlcusu, SartnameSimple, UY]
    cat_test = data["cat_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    cat_cardinalities = data["cat_cardinalities"]
    cat_columns = data["cat_columns"]

    # One-hot encode categoricals
    cat_train_onehot = []
    cat_test_onehot = []
    for i, card in enumerate(cat_cardinalities):
        cat_train_onehot.append(np.eye(card)[cat_train[:, i]])
        cat_test_onehot.append(np.eye(card)[cat_test[:, i]])

    cat_train_onehot = np.hstack(cat_train_onehot)
    cat_test_onehot = np.hstack(cat_test_onehot)

    # Combine features
    X_train = np.column_stack([X_train_num, cat_train_onehot])
    X_test = np.column_stack([X_test_num, cat_test_onehot])

    total_cat_dims = sum(cat_cardinalities)

    print(f"\nDataset info:")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: Cap, Boy (2 numeric) + {cat_columns} ({total_cat_dims} one-hot)")
    print(f"  Total features: {X_train.shape[1]}")
    print(f"  Target: MakineSure (minutes for 100k units)")

    print(f"\nFeature stats:")
    print(f"  Cap: {X_train_num[:, 0].min():.1f} - {X_train_num[:, 0].max():.1f} mm")
    print(f"  Boy: {X_train_num[:, 1].min():.1f} - {X_train_num[:, 1].max():.1f} mm")

    print(f"\nTarget stats:")
    print(f"  Train: min={y_train.min():.1f}, max={y_train.max():.1f}, mean={y_train.mean():.1f}, std={y_train.std():.1f}")
    print(f"  Test: min={y_test.min():.1f}, max={y_test.max():.1f}, mean={y_test.mean():.1f}")

    # Evaluate baseline regressors
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION (Experiment 013 - Ozel Rich Dataset)")
    print("=" * 70)
    print("\nPredicting: MakineSure - Machine duration in minutes for 100k units")
    print(f"Features: Cap, Boy + {cat_columns}")

    regressors = [
        ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("Ridge", Ridge(alpha=1.0)),
    ]

    results = {}

    for reg_name, reg in regressors:
        metrics = evaluate_regressor(reg_name, reg, X_train, y_train, X_test, y_test)
        results[reg_name] = metrics
        print(f"\n{reg_name}:")
        print(f"  RMSE: {metrics['rmse']:.1f} minutes")
        print(f"  MAE:  {metrics['mae']:.1f} minutes")
        print(f"  R2:   {metrics['r2']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)

    print("\n| Model | RMSE (min) | R2 |")
    print("|-------|------------|-----|")
    for reg_name in results:
        print(f"| {reg_name} | {results[reg_name]['rmse']:.1f} | {results[reg_name]['r2']:.4f} |")

    # Assessment
    best_r2 = max(r["r2"] for r in results.values())
    print(f"\nBest R2: {best_r2:.4f}")

    if best_r2 > 0.9:
        print("Assessment: HIGH baseline - augmentation may not improve much")
    elif best_r2 > 0.7:
        print("Assessment: MODERATE baseline - good candidate for augmentation!")
    elif best_r2 > 0.5:
        print("Assessment: MODERATE-LOW baseline - augmentation should help!")
    else:
        print("Assessment: LOW baseline - augmentation should definitely help!")


if __name__ == "__main__":
    main()
