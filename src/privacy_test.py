"""
Membership Inference Attack Test for Privacy Validation.

Tests whether synthetic data generation methods leak information about training data.
If an attacker can determine whether a specific record was in the training set,
the synthetic data is not privacy-preserving.

Key metric: Attack AUC
- AUC ~ 0.5 = no information leak (random guess) = GOOD
- AUC > 0.6 = possible privacy concern
- AUC > 0.7 = significant privacy risk

This is Experiment 016.

Usage:
    python src/privacy_test.py --device cuda
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

try:
    import smogn
    SMOGN_AVAILABLE = True
except ImportError:
    SMOGN_AVAILABLE = False

from diffusion import HybridDiffusion
from models import MLPDenoiser


def load_data(data_path="data/ozel_rich/prepared.pt"):
    """Load prepared ozel rich data."""
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    return data


def load_preprocessors(preprocessor_path="data/ozel_rich/preprocessors.pkl"):
    """Load preprocessors for inverse transform."""
    with open(preprocessor_path, "rb") as f:
        return pickle.load(f)


def load_model(checkpoint_path, device):
    """Load trained Hybrid diffusion model."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint["model_config"]
    diffusion_config = checkpoint["diffusion_config"]

    model = MLPDenoiser(
        d_in=model_config["d_in"],
        hidden_dims=model_config["hidden_dims"],
        dropout=model_config.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    diffusion = HybridDiffusion(
        num_numerical=diffusion_config["num_numerical"],
        cat_cardinalities=diffusion_config["cat_cardinalities"],
        num_timesteps=diffusion_config["num_timesteps"],
        beta_schedule=diffusion_config["beta_schedule"],
    ).to(device)

    return model, diffusion, checkpoint


def generate_samples(model, diffusion, n_samples, device):
    """Generate synthetic samples using Hybrid diffusion."""
    print(f"Generating {n_samples} synthetic samples...")
    with torch.no_grad():
        samples = diffusion.sample(
            model,
            batch_size=n_samples,
            device=device,
            clip_denoised=True,
        )
    return samples.cpu().numpy()


def prepare_features_for_attack(X_num, cat_idx, y, cat_cardinalities):
    """Prepare combined numeric + one-hot categorical features."""
    # One-hot encode categoricals
    onehot_list = []
    for i, card in enumerate(cat_cardinalities):
        onehot_list.append(np.eye(card)[cat_idx[:, i]])
    cat_onehot = np.hstack(onehot_list).astype(np.float32)

    # Combine all features: numeric + categorical + target
    return np.column_stack([X_num, cat_onehot, y])


def membership_inference_attack(
    train_features,
    holdout_features,
    synthetic_features,
    n_neighbors=1,
    use_target=True
):
    """
    Perform membership inference attack using nearest neighbor distances.

    The attack works by:
    1. Computing the distance from each real record to its nearest synthetic sample
    2. Training a classifier to distinguish "members" (train) from "non-members" (holdout)

    If the synthetic data memorizes training records, members will have smaller distances.

    Args:
        train_features: Features of records used to train the generative model
        holdout_features: Features of records NOT used in training (held out)
        synthetic_features: Features of generated synthetic samples
        n_neighbors: Number of nearest neighbors to consider
        use_target: Whether to include target variable in distance computation

    Returns:
        dict with attack metrics
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
    # Label: 1 = member (train), 0 = non-member (holdout)
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

    # Compute TPR at low FPR (more relevant for privacy)
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


def generate_smogn_samples(X_num, y, n_samples=None):
    """Generate SMOGN augmented samples."""
    if not SMOGN_AVAILABLE:
        return None, None

    import pandas as pd

    df = pd.DataFrame(X_num, columns=["Cap", "Boy"])
    df["target"] = y

    try:
        df_smogn = smogn.smoter(
            data=df,
            y="target",
            samp_method="extreme",
        )
        X_smogn = df_smogn[["Cap", "Boy"]].values.astype(np.float32)
        y_smogn = df_smogn["target"].values.astype(np.float32)
        return X_smogn, y_smogn
    except Exception as e:
        print(f"SMOGN failed: {e}")
        return None, None


def main(args):
    # Load data
    data = load_data(args.data_path)
    preprocessors = load_preprocessors()

    # Get data
    X_train_num = data["X_train_num_unscaled"]  # [Cap, Boy]
    X_test_num = data["X_test_num_unscaled"]
    cat_train = data["cat_train"]
    cat_test = data["cat_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    cat_cardinalities = data["cat_cardinalities"]
    num_numerical = data["num_numerical"]

    print(f"\nData summary:")
    print(f"  Training set (members): {len(X_train_num)} samples")
    print(f"  Test set (non-members): {len(X_test_num)} samples")
    print(f"  Numeric features: Cap, Boy")
    print(f"  Categorical features: {data['cat_columns']}")
    print(f"  Total feature dimensions: {2 + sum(cat_cardinalities)} (after one-hot)")

    # Prepare features for attack
    train_features = prepare_features_for_attack(
        X_train_num, cat_train, y_train, cat_cardinalities
    )
    holdout_features = prepare_features_for_attack(
        X_test_num, cat_test, y_test, cat_cardinalities
    )

    print(f"\nFeature shape for attack: {train_features.shape[1]} dimensions")

    # Load diffusion model
    checkpoint_path = Path("checkpoints/ozel_rich/final_model.pt")
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please run train_ozel_rich.py first.")
        return

    model, diffusion, checkpoint = load_model(checkpoint_path, args.device)

    # Generate synthetic samples
    n_synthetic = max(len(X_train_num), args.n_synthetic)
    synthetic_scaled = generate_samples(model, diffusion, n_synthetic, args.device)

    # Process synthetic samples
    scaler = preprocessors["scaler"]
    target_scaler = preprocessors["target_scaler"]

    synthetic_num_scaled = synthetic_scaled[:, :num_numerical]
    synthetic_cat_probs = synthetic_scaled[:, num_numerical:]

    # Unscale numeric
    synthetic_num_unclipped = synthetic_num_scaled * 3
    synthetic_features_unscaled = scaler.inverse_transform(synthetic_num_unclipped[:, :2])
    synthetic_target_unscaled = target_scaler.inverse_transform(
        synthetic_num_unclipped[:, 2:3]
    ).flatten()

    # Convert categorical probabilities to one-hot
    offset = 0
    synthetic_cat_onehot_list = []
    for card in cat_cardinalities:
        probs = synthetic_cat_probs[:, offset:offset + card]
        indices = probs.argmax(axis=1)
        synthetic_cat_onehot_list.append(np.eye(card)[indices])
        offset += card
    synthetic_cat_onehot = np.hstack(synthetic_cat_onehot_list).astype(np.float32)

    # Combine synthetic features
    synthetic_features = np.column_stack([
        synthetic_features_unscaled,
        synthetic_cat_onehot,
        synthetic_target_unscaled
    ])

    print(f"\nGenerated {len(synthetic_features)} synthetic samples")

    # Run membership inference attack on DIFFUSION
    print("\n" + "=" * 70)
    print("MEMBERSHIP INFERENCE ATTACK: DIFFUSION")
    print("=" * 70)

    diffusion_results = membership_inference_attack(
        train_features,
        holdout_features,
        synthetic_features,
        n_neighbors=args.n_neighbors
    )

    print(f"\nResults:")
    print(f"  Attack AUC:      {diffusion_results['attack_auc']:.4f}")
    print(f"  Attack Accuracy: {diffusion_results['attack_accuracy']:.4f}")
    print(f"  TPR @ 1% FPR:    {diffusion_results['tpr_at_1_fpr']:.4f}")
    print(f"  TPR @ 5% FPR:    {diffusion_results['tpr_at_5_fpr']:.4f}")
    print(f"\n  Mean distance (members):     {diffusion_results['train_dist_mean']:.4f}")
    print(f"  Mean distance (non-members): {diffusion_results['holdout_dist_mean']:.4f}")
    print(f"\n  Privacy Assessment: {'SAFE' if diffusion_results['privacy_safe'] else 'CONCERN'}")

    # Run membership inference attack on SMOGN
    smogn_results = None
    if SMOGN_AVAILABLE:
        print("\n" + "=" * 70)
        print("MEMBERSHIP INFERENCE ATTACK: SMOGN")
        print("=" * 70)

        X_smogn, y_smogn = generate_smogn_samples(X_train_num, y_train)

        if X_smogn is not None:
            # SMOGN only generates numeric, so we sample categoricals
            smogn_cat_indices = np.random.choice(len(cat_train), len(X_smogn), replace=True)
            smogn_cat_onehot_list = []
            for i, card in enumerate(cat_cardinalities):
                smogn_cat_onehot_list.append(np.eye(card)[cat_train[smogn_cat_indices, i]])
            smogn_cat_onehot = np.hstack(smogn_cat_onehot_list).astype(np.float32)

            smogn_features = np.column_stack([X_smogn, smogn_cat_onehot, y_smogn])

            print(f"\nGenerated {len(smogn_features)} SMOGN samples")

            smogn_results = membership_inference_attack(
                train_features,
                holdout_features,
                smogn_features,
                n_neighbors=args.n_neighbors
            )

            print(f"\nResults:")
            print(f"  Attack AUC:      {smogn_results['attack_auc']:.4f}")
            print(f"  Attack Accuracy: {smogn_results['attack_accuracy']:.4f}")
            print(f"  TPR @ 1% FPR:    {smogn_results['tpr_at_1_fpr']:.4f}")
            print(f"  TPR @ 5% FPR:    {smogn_results['tpr_at_5_fpr']:.4f}")
            print(f"\n  Mean distance (members):     {smogn_results['train_dist_mean']:.4f}")
            print(f"  Mean distance (non-members): {smogn_results['holdout_dist_mean']:.4f}")
            print(f"\n  Privacy Assessment: {'SAFE' if smogn_results['privacy_safe'] else 'CONCERN'}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("PRIVACY COMPARISON SUMMARY")
    print("=" * 70)

    print("\n| Method    | Attack AUC | TPR@1%FPR | TPR@5%FPR | Status  |")
    print("|-----------|------------|-----------|-----------|---------|")
    print(f"| Diffusion | {diffusion_results['attack_auc']:.4f}     | "
          f"{diffusion_results['tpr_at_1_fpr']:.4f}    | "
          f"{diffusion_results['tpr_at_5_fpr']:.4f}    | "
          f"{'SAFE' if diffusion_results['privacy_safe'] else 'CONCERN':7} |")

    if smogn_results:
        print(f"| SMOGN     | {smogn_results['attack_auc']:.4f}     | "
              f"{smogn_results['tpr_at_1_fpr']:.4f}    | "
              f"{smogn_results['tpr_at_5_fpr']:.4f}    | "
              f"{'SAFE' if smogn_results['privacy_safe'] else 'CONCERN':7} |")

    print("\nInterpretation:")
    print("  - AUC ~ 0.5: No information leak (random guessing)")
    print("  - AUC > 0.6: Possible privacy concern")
    print("  - AUC > 0.7: Significant privacy risk")
    print("\n  Lower is better for privacy.")

    if smogn_results:
        if diffusion_results['attack_auc'] < smogn_results['attack_auc']:
            print(f"\n  -> Diffusion is MORE private than SMOGN")
            print(f"     (AUC difference: {smogn_results['attack_auc'] - diffusion_results['attack_auc']:.4f})")
        else:
            print(f"\n  -> SMOGN is MORE private than Diffusion")
            print(f"     (AUC difference: {diffusion_results['attack_auc'] - smogn_results['attack_auc']:.4f})")

    # Return results for external use
    return {
        'diffusion': diffusion_results,
        'smogn': smogn_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/ozel_rich/prepared.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_synthetic", type=int, default=2000,
                        help="Minimum number of synthetic samples to generate")
    parser.add_argument("--n_neighbors", type=int, default=1,
                        help="Number of nearest neighbors for distance computation")
    args = parser.parse_args()

    main(args)
