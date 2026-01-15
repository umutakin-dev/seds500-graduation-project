"""
Membership Inference Attack Test for Experiment 018 (TabDDPM-style).

Tests whether the TabDDPM-style synthetic data leaks information about training data.

Key metric: Attack AUC
- AUC ~ 0.5 = no information leak (random guess) = GOOD
- AUC > 0.6 = possible privacy concern
- AUC > 0.7 = significant privacy risk
"""

import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler

from diffusion_tabddpm import HybridDiffusionTabDDPM
from models import MLPDenoiser


def load_v5_data():
    """Load v5 data and preprocessors."""
    data_path = Path("data/ozel_rich_v5/prepared.pt")
    data = torch.load(data_path, weights_only=False)
    with open(data_path.parent / "preprocessors.pkl", "rb") as f:
        preprocessors = pickle.load(f)
    return data, preprocessors


def load_experiment_018_model(device):
    """Load trained experiment 018 model."""
    path = Path("checkpoints/experiment_018/best_model.pt")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model_config = checkpoint["model_config"]
    diffusion_config = checkpoint["diffusion_config"]

    model = MLPDenoiser(
        d_in=model_config["d_in"],
        hidden_dims=model_config["hidden_dims"],
        dropout=model_config.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    diffusion = HybridDiffusionTabDDPM(
        num_numerical=diffusion_config["num_numerical"],
        cat_cardinalities=diffusion_config["cat_cardinalities"],
        num_timesteps=diffusion_config["num_timesteps"],
        beta_schedule=diffusion_config["beta_schedule"],
    ).to(device)

    return model, diffusion


def onehot_encode(cat_idx, cardinalities):
    """Convert categorical indices to one-hot."""
    result = []
    for i, card in enumerate(cardinalities):
        result.append(np.eye(card)[cat_idx[:, i]])
    return np.hstack(result).astype(np.float32)


def prepare_features_for_attack(X_num, cat_idx, y, cat_cardinalities):
    """Prepare combined numeric + one-hot categorical features."""
    cat_onehot = onehot_encode(cat_idx, cat_cardinalities)
    return np.column_stack([X_num, cat_onehot, y])


def generate_samples(model, diffusion, n_samples, device, preprocessors):
    """Generate synthetic samples using experiment 018 model."""
    model.eval()
    cat_cardinalities = diffusion.cat_cardinalities
    scaler = preprocessors["scaler"]
    target_scaler = preprocessors["target_scaler"]

    with torch.no_grad():
        x_num, cat_indices = diffusion.sample(model, batch_size=n_samples, device=device)

    x_num = x_num.cpu().numpy()
    cat_indices = cat_indices.cpu().numpy()

    # Inverse transform
    gen_features_scaled = x_num[:, :2]
    gen_target_scaled = x_num[:, 2:3]
    gen_features = scaler.inverse_transform(gen_features_scaled)
    gen_target = target_scaler.inverse_transform(gen_target_scaled).flatten()

    # Convert categorical indices to one-hot
    cat_onehot = onehot_encode(cat_indices, cat_cardinalities)

    # Combine for attack features
    synthetic_features = np.column_stack([gen_features, cat_onehot, gen_target])

    return synthetic_features


def membership_inference_attack(
    train_features,
    holdout_features,
    synthetic_features,
    n_neighbors=1,
):
    """
    Perform membership inference attack using nearest neighbor distances.

    If synthetic data memorizes training records, members will have smaller distances.
    """
    # Standardize features for distance computation
    scaler = StandardScaler()
    synthetic_scaled = scaler.fit_transform(synthetic_features)
    train_scaled = scaler.transform(train_features)
    holdout_scaled = scaler.transform(holdout_features)

    # Fit nearest neighbors on synthetic data
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn.fit(synthetic_scaled)

    # Compute distances for members (train) and non-members (holdout)
    train_distances, _ = nn.kneighbors(train_scaled)
    holdout_distances, _ = nn.kneighbors(holdout_scaled)

    # Average distance if using multiple neighbors
    if n_neighbors > 1:
        train_distances = train_distances.mean(axis=1, keepdims=True)
        holdout_distances = holdout_distances.mean(axis=1, keepdims=True)

    # Create attack dataset
    X_attack = np.concatenate([train_distances, holdout_distances])
    y_attack = np.concatenate([
        np.ones(len(train_distances)),
        np.zeros(len(holdout_distances))
    ])

    # Train attack classifier
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_attack, y_attack)

    # Get predictions
    y_pred_proba = clf.predict_proba(X_attack)[:, 1]
    y_pred = clf.predict(X_attack)

    # Compute metrics
    auc = roc_auc_score(y_attack, y_pred_proba)
    accuracy = accuracy_score(y_attack, y_pred)

    # Compute TPR at low FPR
    fpr, tpr, thresholds = roc_curve(y_attack, y_pred_proba)
    tpr_at_1_fpr = tpr[np.argmax(fpr >= 0.01)] if np.any(fpr >= 0.01) else tpr[0]
    tpr_at_5_fpr = tpr[np.argmax(fpr >= 0.05)] if np.any(fpr >= 0.05) else tpr[0]

    # Statistics about distances
    train_dist_mean = train_distances.mean()
    holdout_dist_mean = holdout_distances.mean()

    return {
        'attack_auc': auc,
        'attack_accuracy': accuracy,
        'tpr_at_1_fpr': tpr_at_1_fpr,
        'tpr_at_5_fpr': tpr_at_5_fpr,
        'train_dist_mean': train_dist_mean,
        'holdout_dist_mean': holdout_dist_mean,
        'privacy_safe': auc < 0.6,
        'n_train': len(train_distances),
        'n_holdout': len(holdout_distances),
        'n_synthetic': len(synthetic_features),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_neighbors", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=5)
    args = parser.parse_args()

    print("=" * 70)
    print("PRIVACY TEST: EXPERIMENT 018 (TabDDPM-style)")
    print("=" * 70)
    print(f"Using device: {args.device}")

    # Load data and model
    print("\nLoading data and model...")
    data, preprocessors = load_v5_data()
    model, diffusion = load_experiment_018_model(args.device)

    # Get data
    X_train_num = data["X_train_num_unscaled"]
    X_test_num = data["X_test_num_unscaled"]
    cat_train = data["cat_train"]
    cat_test = data["cat_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    cat_cardinalities = data["cat_cardinalities"]

    print(f"\nData summary:")
    print(f"  Training set (members): {len(X_train_num)} samples")
    print(f"  Test set (non-members): {len(X_test_num)} samples")

    # Prepare features for attack
    train_features = prepare_features_for_attack(
        X_train_num, cat_train, y_train, cat_cardinalities
    )
    holdout_features = prepare_features_for_attack(
        X_test_num, cat_test, y_test, cat_cardinalities
    )

    print(f"  Feature dimensions: {train_features.shape[1]}")

    # Run attack multiple times to check consistency
    all_results = []

    for run in range(args.n_runs):
        torch.manual_seed(run * 42)
        np.random.seed(run * 42)

        # Generate synthetic samples
        n_synthetic = len(X_train_num)
        synthetic_features = generate_samples(
            model, diffusion, n_synthetic, args.device, preprocessors
        )

        # Run attack
        results = membership_inference_attack(
            train_features,
            holdout_features,
            synthetic_features,
            n_neighbors=args.n_neighbors
        )
        all_results.append(results)

        print(f"\n  Run {run + 1}: AUC = {results['attack_auc']:.4f} "
              f"[{'SAFE' if results['privacy_safe'] else 'CONCERN'}]")

    # Aggregate results
    mean_auc = np.mean([r['attack_auc'] for r in all_results])
    std_auc = np.std([r['attack_auc'] for r in all_results])
    mean_accuracy = np.mean([r['attack_accuracy'] for r in all_results])

    print("\n" + "=" * 70)
    print("PRIVACY TEST RESULTS: EXPERIMENT 018")
    print("=" * 70)

    print(f"\n  Attack AUC:      {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Attack Accuracy: {mean_accuracy:.4f}")
    print(f"  Number of runs:  {args.n_runs}")

    print("\n  Interpretation:")
    print("    - AUC ~ 0.5: No information leak (random guessing) = SAFE")
    print("    - AUC > 0.6: Possible privacy concern")
    print("    - AUC > 0.7: Significant privacy risk")

    if mean_auc < 0.55:
        status = "EXCELLENT - No privacy leak detected"
    elif mean_auc < 0.6:
        status = "SAFE - Minimal privacy risk"
    elif mean_auc < 0.7:
        status = "CAUTION - Some privacy concern"
    else:
        status = "WARNING - Privacy risk detected"

    print(f"\n  Overall Status: {status}")

    # Comparison with previous results
    print("\n" + "=" * 70)
    print("COMPARISON WITH PREVIOUS EXPERIMENTS")
    print("=" * 70)
    print("\n  | Method                | Attack AUC | Status    |")
    print("  |-----------------------|------------|-----------|")
    print(f"  | Exp 016 (V6 Diffusion)| 0.5116     | SAFE      |")
    print(f"  | Exp 016 (SMOGN)       | 0.5253     | SAFE      |")
    print(f"  | Exp 018 (TabDDPM)     | {mean_auc:.4f}     | {'SAFE' if mean_auc < 0.6 else 'CONCERN':9} |")

    if mean_auc < 0.5116:
        print(f"\n  -> Exp 018 is MORE private than V6 Diffusion!")
    elif mean_auc < 0.55:
        print(f"\n  -> Exp 018 has similar privacy to V6 Diffusion")
    else:
        print(f"\n  -> Exp 018 may have slightly less privacy than V6")

    return {
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'all_results': all_results,
        'privacy_safe': mean_auc < 0.6
    }


if __name__ == "__main__":
    main()
