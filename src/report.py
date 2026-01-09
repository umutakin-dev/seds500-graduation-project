"""
Generate visual report for experiments.

Creates visualizations for:
- Confusion matrix
- Real vs Synthetic distributions
- Correlation matrix comparison
- Training loss curve (if available)

Usage:
    python src/report.py --experiment experiment-001-iris-baseline
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from diffusion import GaussianDiffusion
from models import MLPDenoiser


def load_iris_data():
    """Load and preprocess Iris dataset."""
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = np.clip(X_train_scaled, -3, 3) / 3
    X_test_scaled = np.clip(X_test_scaled, -3, 3) / 3

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'feature_names': feature_names,
        'target_names': target_names,
    }


def generate_synthetic(model, diffusion, n_samples, n_features, device):
    """Generate synthetic samples."""
    model.eval()
    with torch.no_grad():
        X_syn = diffusion.sample(model, (n_samples, n_features), device=device)
        return X_syn.cpu().numpy()


def assign_labels(X_synthetic, X_train_scaled, y_train):
    """Assign labels using 1-NN."""
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train_scaled, y_train)
    return knn.predict(X_synthetic)


def plot_confusion_matrix(y_true, y_pred, target_names, title, save_path):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_distributions(X_real, X_synthetic, feature_names, save_path):
    """Plot real vs synthetic distributions for each feature."""
    n_features = X_real.shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(min(n_features, 4)):
        ax = axes[i]
        ax.hist(X_real[:, i], bins=20, alpha=0.6, label='Real', density=True, color='steelblue')
        ax.hist(X_synthetic[:, i], bins=20, alpha=0.6, label='Synthetic', density=True, color='coral')
        ax.set_title(feature_names[i], fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Real vs Synthetic Data Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_correlation_comparison(X_real, X_synthetic, save_path):
    """Plot correlation matrix comparison."""
    real_corr = np.corrcoef(X_real.T)
    syn_corr = np.corrcoef(X_synthetic.T)
    diff = np.abs(real_corr - syn_corr)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im1 = axes[0].imshow(real_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title('Real Data Correlations', fontsize=12)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    im2 = axes[1].imshow(syn_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title('Synthetic Data Correlations', fontsize=12)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    im3 = axes[2].imshow(diff, cmap='Reds', vmin=0, vmax=0.5)
    axes[2].set_title(f'Absolute Difference\n(mean: {diff.mean():.4f})', fontsize=12)
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    plt.suptitle('Correlation Matrix Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_ml_efficiency(results, save_path):
    """Plot ML efficiency comparison bar chart."""
    scenarios = ['Real -> Real\n(baseline)', 'Synthetic -> Real', 'Augmented -> Real']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(scenarios))
    width = 0.35

    lr_scores = [results['lr']['real_to_real'], results['lr']['syn_to_real'], results['lr']['aug_to_real']]
    rf_scores = [results['rf']['real_to_real'], results['rf']['syn_to_real'], results['rf']['aug_to_real']]

    bars1 = ax.bar(x - width/2, lr_scores, width, label='Logistic Regression', color='steelblue')
    bars2 = ax.bar(x + width/2, rf_scores, width, label='Random Forest', color='coral')

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('ML Efficiency Evaluation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.set_ylim(0.7, 1.05)
    ax.axhline(y=results['lr']['real_to_real'], color='steelblue', linestyle='--', alpha=0.3)
    ax.axhline(y=results['rf']['real_to_real'], color='coral', linestyle='--', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate experiment report")
    parser.add_argument("--experiment", type=str, default="experiment-001-iris-baseline")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/iris/final_model.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Generating Report: {args.experiment}")
    print("=" * 60)

    # Create output directory
    output_dir = Path("experiments") / args.experiment / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1] Loading data...")
    data = load_iris_data()

    # Load model
    print(f"[2] Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    model = MLPDenoiser(
        d_in=checkpoint['model_config']['d_in'],
        hidden_dims=checkpoint['model_config']['hidden_dims']
    ).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    diffusion = GaussianDiffusion(
        num_timesteps=checkpoint['diffusion_config']['num_timesteps'],
        beta_schedule=checkpoint['diffusion_config']['beta_schedule']
    ).to(args.device)

    # Generate synthetic data
    print("[3] Generating synthetic data...")
    X_synthetic_scaled = generate_synthetic(
        model, diffusion,
        len(data['X_train']),
        data['X_train'].shape[1],
        args.device
    )

    # Inverse transform
    X_synthetic_unscaled = X_synthetic_scaled * 3
    X_synthetic = data['scaler'].inverse_transform(np.clip(X_synthetic_unscaled, -3, 3))
    y_synthetic = assign_labels(X_synthetic_scaled, data['X_train_scaled'], data['y_train'])

    # Generate visualizations
    print("\n[4] Generating visualizations...")

    # 1. Distribution comparison
    plot_distributions(
        data['X_train_scaled'], X_synthetic_scaled,
        data['feature_names'],
        output_dir / "distributions.png"
    )

    # 2. Correlation comparison
    plot_correlation_comparison(
        data['X_train_scaled'], X_synthetic_scaled,
        output_dir / "correlations.png"
    )

    # 3. Train models and get confusion matrix
    print("\n[5] Training models for evaluation...")

    from sklearn.linear_model import LogisticRegression

    # Augmented data
    X_aug = np.vstack([data['X_train'], X_synthetic])
    y_aug = np.concatenate([data['y_train'], y_synthetic])

    # Train and evaluate
    results = {
        'lr': {},
        'rf': {}
    }

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(data['X_train'], data['y_train'])
    results['lr']['real_to_real'] = lr.score(data['X_test'], data['y_test'])

    lr_syn = LogisticRegression(max_iter=1000, random_state=42)
    lr_syn.fit(X_synthetic, y_synthetic)
    results['lr']['syn_to_real'] = lr_syn.score(data['X_test'], data['y_test'])

    lr_aug = LogisticRegression(max_iter=1000, random_state=42)
    lr_aug.fit(X_aug, y_aug)
    results['lr']['aug_to_real'] = lr_aug.score(data['X_test'], data['y_test'])

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(data['X_train'], data['y_train'])
    results['rf']['real_to_real'] = rf.score(data['X_test'], data['y_test'])

    rf_syn = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_syn.fit(X_synthetic, y_synthetic)
    results['rf']['syn_to_real'] = rf_syn.score(data['X_test'], data['y_test'])

    rf_aug = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_aug.fit(X_aug, y_aug)
    results['rf']['aug_to_real'] = rf_aug.score(data['X_test'], data['y_test'])
    y_pred_aug = rf_aug.predict(data['X_test'])

    # 4. ML Efficiency bar chart
    plot_ml_efficiency(results, output_dir / "ml_efficiency.png")

    # 5. Confusion matrices
    print("\n[6] Generating confusion matrices...")

    # Baseline confusion matrix
    y_pred_baseline = rf.predict(data['X_test'])
    plot_confusion_matrix(
        data['y_test'], y_pred_baseline,
        data['target_names'],
        'Confusion Matrix: Real -> Real (Baseline)',
        output_dir / "confusion_matrix_baseline.png"
    )

    # Augmented confusion matrix
    plot_confusion_matrix(
        data['y_test'], y_pred_aug,
        data['target_names'],
        'Confusion Matrix: Augmented -> Real',
        output_dir / "confusion_matrix_augmented.png"
    )

    print("\n" + "=" * 60)
    print("Report generated successfully!")
    print(f"Figures saved to: {output_dir}")
    print("=" * 60)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
