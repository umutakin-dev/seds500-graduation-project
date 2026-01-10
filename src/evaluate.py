"""
ML Efficiency Evaluation for tabular diffusion models.

Evaluates whether synthetic data can improve ML model performance:
- Real -> Real: Train on real, test on real (baseline)
- Diffusion -> Real: Train on diffusion synthetic, test on real
- SMOTE -> Real: Train on SMOTE synthetic, test on real (classification only)
- Augmented-Diffusion -> Real: Train on real + diffusion, test on real
- Augmented-SMOTE -> Real: Train on real + SMOTE, test on real (classification only)
- Augmented-Noise -> Real: Train on real + Gaussian noise, test on real (regression)

Usage:
    python src/evaluate.py --dataset iris --checkpoint checkpoints/iris/final_model.pt
    python src/evaluate.py --dataset california --checkpoint checkpoints/california/final_model.pt
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from imblearn.over_sampling import SMOTE

from diffusion import GaussianDiffusion
from models import MLPDenoiser


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_iris_data():
    """Load Iris dataset with features and labels."""
    data = load_iris()
    X, y = data.data, data.target
    feature_names = [name.replace(' (cm)', '') for name in data.feature_names]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = np.clip(X_train_scaled, -3, 3) / 3
    X_test_scaled = np.clip(X_test_scaled, -3, 3) / 3

    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled,
        'scaler': scaler, 'feature_names': feature_names,
        'task': 'classification', 'target_names': list(data.target_names)
    }


def load_california_data():
    """Load California Housing dataset with features and target."""
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # For diffusion, we scale features + target together (as in training)
    X_with_target_train = np.hstack([X_train, y_train.reshape(-1, 1)])
    X_with_target_test = np.hstack([X_test, y_test.reshape(-1, 1)])

    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_with_target_train_scaled = scaler.fit_transform(X_with_target_train)
    X_with_target_test_scaled = scaler.transform(X_with_target_test)
    X_with_target_train_scaled = np.clip(X_with_target_train_scaled, -3, 3) / 3
    X_with_target_test_scaled = np.clip(X_with_target_test_scaled, -3, 3) / 3

    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_train_scaled': X_with_target_train_scaled,
        'X_test_scaled': X_with_target_test_scaled,
        'scaler': scaler, 'feature_names': feature_names,
        'task': 'regression', 'n_features': X.shape[1]
    }


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_data(model, diffusion, n_samples, n_features, device):
    """Generate synthetic samples using trained diffusion model."""
    model.eval()
    with torch.no_grad():
        shape = (n_samples, n_features)
        X_synthetic = diffusion.sample(model, shape, device=device)
        X_synthetic = X_synthetic.cpu().numpy()
    return X_synthetic


def generate_synthetic_labels(X_synthetic, X_train_scaled, y_train):
    """Assign labels to synthetic samples using nearest neighbor."""
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train_scaled, y_train)
    return knn.predict(X_synthetic)


def generate_gaussian_noise_augmentation(X_train, y_train, n_samples, noise_scale=0.1):
    """Generate augmented data using Gaussian noise (for regression baseline)."""
    indices = np.random.choice(len(X_train), size=n_samples, replace=True)
    X_noise = X_train[indices] + np.random.normal(0, noise_scale, (n_samples, X_train.shape[1]))
    y_noise = y_train[indices]
    return X_noise, y_noise


# =============================================================================
# Classification Evaluation
# =============================================================================

def evaluate_classifier(clf, X_train, y_train, X_test, y_test):
    """Train classifier and return accuracy."""
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred), y_pred


def run_classification_evaluation(data, model, diffusion, device, n_synthetic, output_dir=None):
    """Run full classification evaluation (Iris)."""
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    X_train_scaled = data['X_train_scaled']
    scaler, feature_names = data['scaler'], data['feature_names']

    print(f"    Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"    Classes: {np.unique(y_train)}")

    # Generate diffusion synthetic data
    print(f"\n[3] Generating {n_synthetic} diffusion samples...")
    d_in = X_train_scaled.shape[1]
    X_synthetic_scaled = generate_synthetic_data(model, diffusion, n_synthetic, d_in, device)

    # Inverse transform
    X_synthetic_unscaled = X_synthetic_scaled * 3
    X_synthetic = scaler.inverse_transform(np.clip(X_synthetic_unscaled, -3, 3))

    # Assign labels
    print("[4] Assigning labels to diffusion samples...")
    y_synthetic = generate_synthetic_labels(X_synthetic_scaled, X_train_scaled, y_train)
    print(f"    Diffusion label distribution: {np.bincount(y_synthetic)}")
    print(f"    Real label distribution:      {np.bincount(y_train)}")

    # Generate SMOTE data
    print("\n[5] Generating SMOTE samples for comparison...")
    k_neighbors = min(5, min(np.bincount(y_train)) - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_smote_full, y_smote_full = smote.fit_resample(X_train, y_train)
    n_original = len(X_train)
    X_smote = X_smote_full[n_original:]
    y_smote = y_smote_full[n_original:]

    if len(X_smote) < n_synthetic:
        smote_ratio = {cls: n_synthetic // 3 + n_original // 3 for cls in np.unique(y_train)}
        smote = SMOTE(sampling_strategy=smote_ratio, random_state=42, k_neighbors=k_neighbors)
        X_smote_full, y_smote_full = smote.fit_resample(X_train, y_train)
        X_smote = X_smote_full[n_original:]
        y_smote = y_smote_full[n_original:]
    print(f"    SMOTE samples generated: {len(X_smote)}")

    # Evaluate
    print("\n[6] Evaluating ML models...")
    print("=" * 60)

    classifiers = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]

    results = {}
    X_aug_diff = np.vstack([X_train, X_synthetic])
    y_aug_diff = np.concatenate([y_train, y_synthetic])
    X_aug_smote = np.vstack([X_train, X_smote])
    y_aug_smote = np.concatenate([y_train, y_smote])

    for clf_name, clf in classifiers:
        print(f"\n{clf_name}:")
        print("-" * 50)

        acc_rr, _ = evaluate_classifier(type(clf)(**clf.get_params()), X_train, y_train, X_test, y_test)
        acc_diff, _ = evaluate_classifier(type(clf)(**clf.get_params()), X_synthetic, y_synthetic, X_test, y_test)
        acc_smote, _ = evaluate_classifier(type(clf)(**clf.get_params()), X_smote, y_smote, X_test, y_test)
        acc_aug_diff, _ = evaluate_classifier(type(clf)(**clf.get_params()), X_aug_diff, y_aug_diff, X_test, y_test)
        acc_aug_smote, _ = evaluate_classifier(type(clf)(**clf.get_params()), X_aug_smote, y_aug_smote, X_test, y_test)

        print(f"  Real -> Real (baseline):      {acc_rr:.4f}")
        print(f"  Diffusion -> Real:            {acc_diff:.4f} ({(acc_diff - acc_rr)*100:+.1f}%)")
        print(f"  SMOTE -> Real:                {acc_smote:.4f} ({(acc_smote - acc_rr)*100:+.1f}%)")
        print(f"  Augmented-Diffusion -> Real:  {acc_aug_diff:.4f} ({(acc_aug_diff - acc_rr)*100:+.1f}%)")
        print(f"  Augmented-SMOTE -> Real:      {acc_aug_smote:.4f} ({(acc_aug_smote - acc_rr)*100:+.1f}%)")

        results[clf_name] = {
            'real_to_real': acc_rr,
            'diffusion_to_real': acc_diff,
            'smote_to_real': acc_smote,
            'aug_diffusion_to_real': acc_aug_diff,
            'aug_smote_to_real': acc_aug_smote,
        }

    # Confusion matrices
    print("\n" + "=" * 60)
    print("Confusion Matrices (Random Forest)")
    print("=" * 60)

    clf_diff_final = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_diff_final.fit(X_aug_diff, y_aug_diff)
    y_pred_diff = clf_diff_final.predict(X_test)

    clf_smote_final = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_smote_final.fit(X_aug_smote, y_aug_smote)
    y_pred_smote = clf_smote_final.predict(X_test)

    print("\nAugmented-Diffusion:")
    print(confusion_matrix(y_test, y_pred_diff))
    print("\nAugmented-SMOTE:")
    print(confusion_matrix(y_test, y_pred_smote))

    print("\nClassification Report (Augmented-Diffusion):")
    print(classification_report(y_test, y_pred_diff, target_names=data['target_names']))

    # Summary
    print_classification_summary(results)

    # Save data
    if output_dir:
        save_classification_data(X_synthetic, y_synthetic, feature_names, output_dir)

    return results


def print_classification_summary(results):
    """Print summary of classification results."""
    print("\n" + "=" * 60)
    print("SUMMARY: Diffusion vs SMOTE")
    print("=" * 60)
    for clf_name, res in results.items():
        print(f"\n{clf_name}:")
        print(f"  Baseline (Real->Real):        {res['real_to_real']:.4f}")
        print(f"  Diffusion only:               {res['diffusion_to_real']:.4f} ({(res['diffusion_to_real'] - res['real_to_real'])*100:+.1f}%)")
        print(f"  SMOTE only:                   {res['smote_to_real']:.4f} ({(res['smote_to_real'] - res['real_to_real'])*100:+.1f}%)")
        print(f"  Augmented-Diffusion:          {res['aug_diffusion_to_real']:.4f} ({(res['aug_diffusion_to_real'] - res['real_to_real'])*100:+.1f}%)")
        print(f"  Augmented-SMOTE:              {res['aug_smote_to_real']:.4f} ({(res['aug_smote_to_real'] - res['real_to_real'])*100:+.1f}%)")

        diff_vs_smote = res['aug_diffusion_to_real'] - res['aug_smote_to_real']
        if diff_vs_smote > 0:
            print(f"  --> Diffusion beats SMOTE by {diff_vs_smote*100:.1f}%")
        elif diff_vs_smote < 0:
            print(f"  --> SMOTE beats Diffusion by {-diff_vs_smote*100:.1f}%")
        else:
            print(f"  --> Diffusion and SMOTE perform equally")


def save_classification_data(X_synthetic, y_synthetic, feature_names, output_dir):
    """Save synthetic classification data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X_synthetic, columns=feature_names)
    df['label'] = y_synthetic
    df.to_csv(output_dir / "synthetic.csv", index=False)
    print(f"\n[*] Synthetic data saved to {output_dir}/synthetic.csv ({len(df)} samples)")


# =============================================================================
# Regression Evaluation
# =============================================================================

def evaluate_regressor(reg, X_train, y_train, X_test, y_test):
    """Train regressor and return metrics."""
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    return {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'y_pred': y_pred
    }


def run_regression_evaluation(data, model, diffusion, device, n_synthetic, output_dir=None):
    """Run full regression evaluation (California Housing)."""
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    scaler = data['scaler']
    feature_names = data['feature_names']
    n_features = data['n_features']

    print(f"    Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"    Features: {n_features}, Target: median_house_value")

    # Generate diffusion synthetic data (features + target)
    print(f"\n[3] Generating {n_synthetic} diffusion samples...")
    d_in = n_features + 1  # features + target
    X_synthetic_scaled = generate_synthetic_data(model, diffusion, n_synthetic, d_in, device)

    # Inverse transform
    X_synthetic_unscaled = X_synthetic_scaled * 3
    X_synthetic_full = scaler.inverse_transform(np.clip(X_synthetic_unscaled, -3, 3))

    # Split features and target
    X_synthetic = X_synthetic_full[:, :n_features]
    y_synthetic = X_synthetic_full[:, -1]

    print(f"    Generated target range: [{y_synthetic.min():.2f}, {y_synthetic.max():.2f}]")
    print(f"    Real target range:      [{y_train.min():.2f}, {y_train.max():.2f}]")

    # Generate Gaussian noise augmentation (SMOTE doesn't apply to regression)
    print("\n[4] Generating Gaussian noise samples for comparison...")
    X_noise, y_noise = generate_gaussian_noise_augmentation(X_train, y_train, n_synthetic, noise_scale=0.1)
    print(f"    Noise samples generated: {len(X_noise)}")

    # Evaluate
    print("\n[5] Evaluating ML models...")
    print("=" * 60)

    regressors = [
        ("Ridge Regression", Ridge(alpha=1.0, random_state=42)),
        ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ]

    results = {}
    X_aug_diff = np.vstack([X_train, X_synthetic])
    y_aug_diff = np.concatenate([y_train, y_synthetic])
    X_aug_noise = np.vstack([X_train, X_noise])
    y_aug_noise = np.concatenate([y_train, y_noise])

    for reg_name, reg in regressors:
        print(f"\n{reg_name}:")
        print("-" * 50)

        # Real -> Real
        res_rr = evaluate_regressor(type(reg)(**reg.get_params()), X_train, y_train, X_test, y_test)

        # Diffusion -> Real
        res_diff = evaluate_regressor(type(reg)(**reg.get_params()), X_synthetic, y_synthetic, X_test, y_test)

        # Noise -> Real
        res_noise = evaluate_regressor(type(reg)(**reg.get_params()), X_noise, y_noise, X_test, y_test)

        # Augmented-Diffusion -> Real
        res_aug_diff = evaluate_regressor(type(reg)(**reg.get_params()), X_aug_diff, y_aug_diff, X_test, y_test)

        # Augmented-Noise -> Real
        res_aug_noise = evaluate_regressor(type(reg)(**reg.get_params()), X_aug_noise, y_aug_noise, X_test, y_test)

        print(f"  Real -> Real (baseline):      R2={res_rr['r2']:.4f}, RMSE={res_rr['rmse']:.4f}")
        print(f"  Diffusion -> Real:            R2={res_diff['r2']:.4f}, RMSE={res_diff['rmse']:.4f} ({(res_diff['r2'] - res_rr['r2'])*100:+.2f}%)")
        print(f"  Noise -> Real:                R2={res_noise['r2']:.4f}, RMSE={res_noise['rmse']:.4f} ({(res_noise['r2'] - res_rr['r2'])*100:+.2f}%)")
        print(f"  Augmented-Diffusion -> Real:  R2={res_aug_diff['r2']:.4f}, RMSE={res_aug_diff['rmse']:.4f} ({(res_aug_diff['r2'] - res_rr['r2'])*100:+.2f}%)")
        print(f"  Augmented-Noise -> Real:      R2={res_aug_noise['r2']:.4f}, RMSE={res_aug_noise['rmse']:.4f} ({(res_aug_noise['r2'] - res_rr['r2'])*100:+.2f}%)")

        results[reg_name] = {
            'real_to_real': res_rr,
            'diffusion_to_real': res_diff,
            'noise_to_real': res_noise,
            'aug_diffusion_to_real': res_aug_diff,
            'aug_noise_to_real': res_aug_noise,
        }

    # Summary
    print_regression_summary(results)

    # Save data
    if output_dir:
        save_regression_data(X_synthetic, y_synthetic, feature_names, output_dir)

    return results


def print_regression_summary(results):
    """Print summary of regression results."""
    print("\n" + "=" * 60)
    print("SUMMARY: Diffusion vs Gaussian Noise Augmentation")
    print("=" * 60)
    for reg_name, res in results.items():
        baseline_r2 = res['real_to_real']['r2']
        print(f"\n{reg_name}:")
        print(f"  Baseline (Real->Real):        R2={baseline_r2:.4f}")
        print(f"  Diffusion only:               R2={res['diffusion_to_real']['r2']:.4f} ({(res['diffusion_to_real']['r2'] - baseline_r2)*100:+.2f}%)")
        print(f"  Noise only:                   R2={res['noise_to_real']['r2']:.4f} ({(res['noise_to_real']['r2'] - baseline_r2)*100:+.2f}%)")
        print(f"  Augmented-Diffusion:          R2={res['aug_diffusion_to_real']['r2']:.4f} ({(res['aug_diffusion_to_real']['r2'] - baseline_r2)*100:+.2f}%)")
        print(f"  Augmented-Noise:              R2={res['aug_noise_to_real']['r2']:.4f} ({(res['aug_noise_to_real']['r2'] - baseline_r2)*100:+.2f}%)")

        diff_vs_noise = res['aug_diffusion_to_real']['r2'] - res['aug_noise_to_real']['r2']
        if diff_vs_noise > 0.001:
            print(f"  --> Diffusion beats Noise by {diff_vs_noise*100:.2f}% R2")
        elif diff_vs_noise < -0.001:
            print(f"  --> Noise beats Diffusion by {-diff_vs_noise*100:.2f}% R2")
        else:
            print(f"  --> Diffusion and Noise perform similarly")


def save_regression_data(X_synthetic, y_synthetic, feature_names, output_dir):
    """Save synthetic regression data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X_synthetic, columns=feature_names)
    df['target'] = y_synthetic
    df.to_csv(output_dir / "synthetic.csv", index=False)
    print(f"\n[*] Synthetic data saved to {output_dir}/synthetic.csv ({len(df)} samples)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ML Efficiency Evaluation")
    parser.add_argument("--dataset", type=str, required=True, choices=["iris", "california"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_synthetic", type=int, default=None, help="Number of synthetic samples")
    parser.add_argument("--output", type=str, default=None, help="Output directory for saving data")
    args = parser.parse_args()

    print("=" * 60)
    print(f"ML Efficiency Evaluation - {args.dataset.upper()}")
    print("=" * 60)

    # Load data
    print(f"\n[1] Loading {args.dataset} dataset...")
    if args.dataset == "iris":
        data = load_iris_data()
    else:
        data = load_california_data()

    # Load model
    print(f"\n[2] Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    d_in = checkpoint['model_config']['d_in']
    hidden_dims = checkpoint['model_config']['hidden_dims']

    model = MLPDenoiser(d_in=d_in, hidden_dims=hidden_dims).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    diffusion = GaussianDiffusion(
        num_timesteps=checkpoint['diffusion_config']['num_timesteps'],
        beta_schedule=checkpoint['diffusion_config']['beta_schedule']
    ).to(args.device)

    print("    Model loaded successfully")

    # Determine sample size
    n_synthetic = args.n_synthetic or len(data['X_train'])

    # Run evaluation
    if data['task'] == 'classification':
        results = run_classification_evaluation(data, model, diffusion, args.device, n_synthetic, args.output)
    else:
        results = run_regression_evaluation(data, model, diffusion, args.device, n_synthetic, args.output)

    return results


if __name__ == "__main__":
    main()
