"""
ML Efficiency Evaluation for tabular diffusion models.

Evaluates whether synthetic data can improve ML model performance:
- Real → Real: Train on real, test on real (baseline)
- Synthetic → Real: Train on synthetic, test on real (ML efficiency)
- Augmented → Real: Train on real+synthetic, test on real (augmentation)

Usage:
    python src/evaluate.py --checkpoint checkpoints/iris/final_model.pt
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from diffusion import GaussianDiffusion
from models import MLPDenoiser


def load_iris_with_labels():
    """Load Iris dataset with features and labels."""
    data = load_iris()
    X, y = data.data, data.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features for diffusion
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = np.clip(X_train_scaled, -3, 3) / 3
    X_test_scaled = np.clip(X_test_scaled, -3, 3) / 3

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def generate_synthetic_data(model, diffusion, n_samples, n_features, device):
    """Generate synthetic samples using trained diffusion model."""
    model.eval()
    with torch.no_grad():
        shape = (n_samples, n_features)
        X_synthetic = diffusion.sample(model, shape, device=device)
        X_synthetic = X_synthetic.cpu().numpy()
    return X_synthetic


def generate_synthetic_labels(X_synthetic, X_train_scaled, y_train):
    """
    Assign labels to synthetic samples using nearest neighbor.
    This is a simple approach - each synthetic sample gets the label
    of its nearest real sample.
    """
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train_scaled, y_train)
    y_synthetic = knn.predict(X_synthetic)
    return y_synthetic


def evaluate_classifier(clf, X_train, y_train, X_test, y_test, name=""):
    """Train classifier and return accuracy."""
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc, y_pred


def main():
    parser = argparse.ArgumentParser(description="ML Efficiency Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/iris/final_model.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_synthetic", type=int, default=None, help="Number of synthetic samples (default: same as training)")
    args = parser.parse_args()

    print("=" * 60)
    print("ML Efficiency Evaluation")
    print("=" * 60)

    # Load data
    print("\n[1] Loading Iris dataset...")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = load_iris_with_labels()
    print(f"    Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"    Classes: {np.unique(y_train)}")

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

    print(f"    Model loaded successfully")

    # Generate synthetic data
    n_synthetic = args.n_synthetic or len(X_train)
    print(f"\n[3] Generating {n_synthetic} synthetic samples...")
    X_synthetic_scaled = generate_synthetic_data(model, diffusion, n_synthetic, d_in, args.device)

    # Inverse transform to original scale
    X_synthetic_unscaled = X_synthetic_scaled * 3  # Undo the /3
    X_synthetic = scaler.inverse_transform(np.clip(X_synthetic_unscaled, -3, 3))

    # Assign labels to synthetic data
    print("[4] Assigning labels to synthetic samples...")
    y_synthetic = generate_synthetic_labels(X_synthetic_scaled, X_train_scaled, y_train)
    print(f"    Synthetic label distribution: {np.bincount(y_synthetic)}")
    print(f"    Real label distribution:      {np.bincount(y_train)}")

    # Evaluate classifiers
    print("\n[5] Evaluating ML models...")
    print("=" * 60)

    classifiers = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]

    results = {}

    for clf_name, clf in classifiers:
        print(f"\n{clf_name}:")
        print("-" * 40)

        # Real → Real (baseline)
        clf_rr = type(clf)(**clf.get_params())
        acc_rr, _ = evaluate_classifier(clf_rr, X_train, y_train, X_test, y_test)

        # Synthetic → Real
        clf_sr = type(clf)(**clf.get_params())
        acc_sr, _ = evaluate_classifier(clf_sr, X_synthetic, y_synthetic, X_test, y_test)

        # Augmented → Real (real + synthetic)
        X_augmented = np.vstack([X_train, X_synthetic])
        y_augmented = np.concatenate([y_train, y_synthetic])
        clf_ar = type(clf)(**clf.get_params())
        acc_ar, y_pred_ar = evaluate_classifier(clf_ar, X_augmented, y_augmented, X_test, y_test)

        print(f"  Real -> Real (baseline):    {acc_rr:.4f}")
        print(f"  Synthetic -> Real:          {acc_sr:.4f} ({'+' if acc_sr >= acc_rr else ''}{(acc_sr - acc_rr)*100:.1f}%)")
        print(f"  Augmented -> Real:          {acc_ar:.4f} ({'+' if acc_ar >= acc_rr else ''}{(acc_ar - acc_rr)*100:.1f}%)")

        results[clf_name] = {
            'real_to_real': acc_rr,
            'synthetic_to_real': acc_sr,
            'augmented_to_real': acc_ar,
        }

    # Print confusion matrix for best augmented model
    print("\n" + "=" * 60)
    print("Confusion Matrix (Random Forest, Augmented -> Real)")
    print("=" * 60)

    clf_final = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_final.fit(X_augmented, y_augmented)
    y_pred_final = clf_final.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_final)
    print(f"\n{cm}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_final, target_names=['setosa', 'versicolor', 'virginica']))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for clf_name, res in results.items():
        delta_sr = (res['synthetic_to_real'] - res['real_to_real']) * 100
        delta_ar = (res['augmented_to_real'] - res['real_to_real']) * 100
        print(f"\n{clf_name}:")
        print(f"  ML Efficiency (Syn->Real vs Real->Real): {delta_sr:+.1f}%")
        print(f"  Augmentation benefit (Aug->Real vs Real->Real): {delta_ar:+.1f}%")

    return results


if __name__ == "__main__":
    main()
