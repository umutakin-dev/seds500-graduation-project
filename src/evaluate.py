"""
ML Efficiency Evaluation for tabular diffusion models.

Evaluates whether synthetic data can improve ML model performance:
- Real -> Real: Train on real, test on real (baseline)
- Diffusion -> Real: Train on diffusion synthetic, test on real
- SMOTE -> Real: Train on SMOTE synthetic, test on real
- Augmented-Diffusion -> Real: Train on real + diffusion, test on real
- Augmented-SMOTE -> Real: Train on real + SMOTE, test on real

Usage:
    python src/evaluate.py --checkpoint checkpoints/iris/final_model.pt
    python src/evaluate.py --checkpoint checkpoints/iris/final_model.pt --output experiments/experiment-001-iris-baseline/data
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

from diffusion import GaussianDiffusion
from models import MLPDenoiser


def load_iris_with_labels():
    """Load Iris dataset with features and labels."""
    data = load_iris()
    X, y = data.data, data.target
    feature_names = [name.replace(' (cm)', '') for name in data.feature_names]

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

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_names


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


def save_synthetic_data(X_synthetic, y_synthetic, feature_names, output_dir):
    """Save synthetic data to experiment folder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic data only
    df_synthetic = pd.DataFrame(X_synthetic, columns=feature_names)
    df_synthetic['label'] = y_synthetic
    df_synthetic.to_csv(output_dir / "synthetic.csv", index=False)

    print(f"\n[*] Synthetic data saved to {output_dir}/synthetic.csv ({len(df_synthetic)} samples)")


def main():
    parser = argparse.ArgumentParser(description="ML Efficiency Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/iris/final_model.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_synthetic", type=int, default=None, help="Number of synthetic samples (default: same as training)")
    parser.add_argument("--output", type=str, default=None, help="Output directory for saving data (optional)")
    args = parser.parse_args()

    print("=" * 60)
    print("ML Efficiency Evaluation")
    print("=" * 60)

    # Load data
    print("\n[1] Loading Iris dataset...")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_names = load_iris_with_labels()
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
    print("[4] Assigning labels to diffusion samples...")
    y_synthetic = generate_synthetic_labels(X_synthetic_scaled, X_train_scaled, y_train)
    print(f"    Diffusion label distribution: {np.bincount(y_synthetic)}")
    print(f"    Real label distribution:      {np.bincount(y_train)}")

    # Generate SMOTE data for comparison
    print("\n[5] Generating SMOTE samples for comparison...")
    smote = SMOTE(random_state=42, k_neighbors=min(5, min(np.bincount(y_train)) - 1))
    # SMOTE doubles the dataset, so we take only the new samples
    X_smote_full, y_smote_full = smote.fit_resample(X_train, y_train)
    # Extract only the synthetic samples (SMOTE appends them at the end)
    n_original = len(X_train)
    X_smote = X_smote_full[n_original:]
    y_smote = y_smote_full[n_original:]
    # If SMOTE generated fewer samples than diffusion, resample to match
    if len(X_smote) < n_synthetic:
        # Repeat SMOTE with higher sampling
        smote_ratio = {cls: n_synthetic // 3 + n_original // 3 for cls in np.unique(y_train)}
        smote = SMOTE(sampling_strategy=smote_ratio, random_state=42, k_neighbors=min(5, min(np.bincount(y_train)) - 1))
        X_smote_full, y_smote_full = smote.fit_resample(X_train, y_train)
        X_smote = X_smote_full[n_original:]
        y_smote = y_smote_full[n_original:]
    print(f"    SMOTE samples generated: {len(X_smote)}")
    print(f"    SMOTE label distribution: {np.bincount(y_smote)}")

    # Evaluate classifiers
    print("\n[6] Evaluating ML models...")
    print("=" * 60)

    classifiers = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]

    results = {}

    for clf_name, clf in classifiers:
        print(f"\n{clf_name}:")
        print("-" * 50)

        # Real -> Real (baseline)
        clf_rr = type(clf)(**clf.get_params())
        acc_rr, _ = evaluate_classifier(clf_rr, X_train, y_train, X_test, y_test)

        # Diffusion -> Real
        clf_diff = type(clf)(**clf.get_params())
        acc_diff, _ = evaluate_classifier(clf_diff, X_synthetic, y_synthetic, X_test, y_test)

        # SMOTE -> Real
        clf_smote = type(clf)(**clf.get_params())
        acc_smote, _ = evaluate_classifier(clf_smote, X_smote, y_smote, X_test, y_test)

        # Augmented-Diffusion -> Real
        X_aug_diff = np.vstack([X_train, X_synthetic])
        y_aug_diff = np.concatenate([y_train, y_synthetic])
        clf_aug_diff = type(clf)(**clf.get_params())
        acc_aug_diff, _ = evaluate_classifier(clf_aug_diff, X_aug_diff, y_aug_diff, X_test, y_test)

        # Augmented-SMOTE -> Real
        X_aug_smote = np.vstack([X_train, X_smote])
        y_aug_smote = np.concatenate([y_train, y_smote])
        clf_aug_smote = type(clf)(**clf.get_params())
        acc_aug_smote, _ = evaluate_classifier(clf_aug_smote, X_aug_smote, y_aug_smote, X_test, y_test)

        print(f"  Real -> Real (baseline):      {acc_rr:.4f}")
        print(f"  Diffusion -> Real:            {acc_diff:.4f} ({'+' if acc_diff >= acc_rr else ''}{(acc_diff - acc_rr)*100:.1f}%)")
        print(f"  SMOTE -> Real:                {acc_smote:.4f} ({'+' if acc_smote >= acc_rr else ''}{(acc_smote - acc_rr)*100:.1f}%)")
        print(f"  Augmented-Diffusion -> Real:  {acc_aug_diff:.4f} ({'+' if acc_aug_diff >= acc_rr else ''}{(acc_aug_diff - acc_rr)*100:.1f}%)")
        print(f"  Augmented-SMOTE -> Real:      {acc_aug_smote:.4f} ({'+' if acc_aug_smote >= acc_rr else ''}{(acc_aug_smote - acc_rr)*100:.1f}%)")

        results[clf_name] = {
            'real_to_real': acc_rr,
            'diffusion_to_real': acc_diff,
            'smote_to_real': acc_smote,
            'aug_diffusion_to_real': acc_aug_diff,
            'aug_smote_to_real': acc_aug_smote,
        }

    # Print confusion matrices
    print("\n" + "=" * 60)
    print("Confusion Matrices (Random Forest)")
    print("=" * 60)

    # Augmented-Diffusion
    clf_diff_final = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_diff_final.fit(X_aug_diff, y_aug_diff)
    y_pred_diff = clf_diff_final.predict(X_test)
    cm_diff = confusion_matrix(y_test, y_pred_diff)

    # Augmented-SMOTE
    clf_smote_final = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_smote_final.fit(X_aug_smote, y_aug_smote)
    y_pred_smote = clf_smote_final.predict(X_test)
    cm_smote = confusion_matrix(y_test, y_pred_smote)

    print("\nAugmented-Diffusion:")
    print(cm_diff)
    print("\nAugmented-SMOTE:")
    print(cm_smote)

    print("\nClassification Report (Augmented-Diffusion):")
    print(classification_report(y_test, y_pred_diff, target_names=['setosa', 'versicolor', 'virginica']))

    # Summary
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

        # Direct comparison
        diff_vs_smote = res['aug_diffusion_to_real'] - res['aug_smote_to_real']
        if diff_vs_smote > 0:
            print(f"  --> Diffusion beats SMOTE by {diff_vs_smote*100:.1f}%")
        elif diff_vs_smote < 0:
            print(f"  --> SMOTE beats Diffusion by {-diff_vs_smote*100:.1f}%")
        else:
            print(f"  --> Diffusion and SMOTE perform equally")

    # Save synthetic data if output directory specified
    if args.output:
        save_synthetic_data(X_synthetic, y_synthetic, feature_names, args.output)

    return results


if __name__ == "__main__":
    main()
