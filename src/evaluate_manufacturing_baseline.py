"""
Evaluate baseline ML models on manufacturing duration prediction.

Uses only Çap and Boy as features (no categorical).
This is Experiment 010: Establish baseline performance before augmentation.

Expected baseline: R² ~ 0.75 (harder problem, room for augmentation to help)

Usage:
    python src/evaluate_manufacturing_baseline.py
"""

import pickle
import torch
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data(data_path="data/manufacturing/prepared.pt"):
    """Load prepared manufacturing data."""
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

    # Get unscaled features and targets for ML evaluation
    X_train = data["X_train_unscaled"]  # [Çap, Boy]
    X_test = data["X_test_unscaled"]
    y_train = data["y_train"]  # Original target in minutes
    y_test = data["y_test"]

    print(f"\nDataset info:")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: Çap, Boy (2 numeric only)")
    print(f"  Target: Machine duration (minutes for 100k units)")

    print(f"\nFeature stats:")
    print(f"  Çap: {X_train[:, 0].min():.1f} - {X_train[:, 0].max():.1f} mm")
    print(f"  Boy: {X_train[:, 1].min():.1f} - {X_train[:, 1].max():.1f} mm")

    print(f"\nTarget stats:")
    print(f"  Train: min={y_train.min():.1f}, max={y_train.max():.1f}, mean={y_train.mean():.1f}")
    print(f"  Test: min={y_test.min():.1f}, max={y_test.max():.1f}, mean={y_test.mean():.1f}")

    # Evaluate baseline regressors
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION (Experiment 010)")
    print("=" * 70)
    print("\nPredicting: Makine Süre (100.000 ADET) DK - Machine duration in minutes")
    print("Features: Çap (diameter), Boy (length) - NO categorical features")
    print("\nNote: Kısa tanım and İş Yeri were excluded as they explain 80% of variance")

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
        print(f"  R²:   {metrics['r2']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)

    print("\n| Model | RMSE (min) | R² |")
    print("|-------|------------|-----|")
    for reg_name in results:
        print(f"| {reg_name} | {results[reg_name]['rmse']:.1f} | {results[reg_name]['r2']:.4f} |")

    # Assessment
    best_r2 = max(r["r2"] for r in results.values())
    print(f"\nBest R²: {best_r2:.4f}")

    if best_r2 > 0.9:
        print("Assessment: HIGH baseline - augmentation may not improve much")
    elif best_r2 > 0.7:
        print("Assessment: MODERATE baseline - good candidate for augmentation!")
    else:
        print("Assessment: LOW baseline - augmentation should definitely help!")


if __name__ == "__main__":
    main()
