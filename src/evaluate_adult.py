"""
Evaluate hybrid diffusion on Adult dataset.

Compares: Real vs Diffusion vs SMOTE for classification.

Usage:
    python src/evaluate_adult.py --checkpoint checkpoints/adult/final_model.pt
"""

import argparse
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

from diffusion import HybridDiffusion
from models import HybridMLPDenoiser


def load_adult_data():
    """Load Adult dataset (same as training)."""
    print("Loading Adult dataset...")
    adult = fetch_openml("adult", version=2, as_frame=True)

    df = adult.data.copy()
    y = (adult.target == ">50K").astype(int).values

    num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

    for col in cat_cols:
        df[col] = df[col].cat.add_categories("Missing").fillna("Missing")

    label_encoders = {}
    cat_cardinalities = []
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        cat_cardinalities.append(len(le.classes_))

    X_num = df[num_cols].values.astype(np.float32)
    X_cat = df[cat_cols].values.astype(np.int64)

    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = train_test_split(
        X_num, X_cat, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)
    X_num_train_scaled = np.clip(X_num_train_scaled, -3, 3) / 3
    X_num_test_scaled = np.clip(X_num_test_scaled, -3, 3) / 3

    def to_onehot(X_cat, cardinalities):
        batch_size = X_cat.shape[0]
        onehot_list = []
        for i, card in enumerate(cardinalities):
            onehot = np.zeros((batch_size, card), dtype=np.float32)
            onehot[np.arange(batch_size), X_cat[:, i]] = 1.0
            onehot_list.append(onehot)
        return np.hstack(onehot_list)

    X_cat_train_onehot = to_onehot(X_cat_train, cat_cardinalities)
    X_cat_test_onehot = to_onehot(X_cat_test, cat_cardinalities)

    X_train_hybrid = np.hstack([X_num_train_scaled, X_cat_train_onehot])
    X_test_hybrid = np.hstack([X_num_test_scaled, X_cat_test_onehot])

    # For ML models, use original features (not one-hot for diffusion)
    X_train_ml = np.hstack([X_num_train, X_cat_train])
    X_test_ml = np.hstack([X_num_test, X_cat_test])

    print(f"Train: {len(X_train_ml)}, Test: {len(X_test_ml)}")
    print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

    return {
        "X_train_hybrid": X_train_hybrid,
        "X_test_hybrid": X_test_hybrid,
        "X_train_ml": X_train_ml,
        "X_test_ml": X_test_ml,
        "X_num_train": X_num_train,
        "X_num_test": X_num_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "cat_cardinalities": cat_cardinalities,
        "num_numerical": len(num_cols),
    }


def generate_diffusion_samples(model, diffusion, n_samples, device):
    """Generate synthetic samples using trained hybrid diffusion."""
    model.eval()
    with torch.no_grad():
        samples = diffusion.sample(model, n_samples, device=device)
    return samples.cpu().numpy()


def decode_hybrid_samples(samples, num_numerical, cat_cardinalities, scaler):
    """Convert hybrid diffusion output back to ML-ready format."""
    # Split numerical and categorical
    X_num_scaled = samples[:, :num_numerical]

    # Inverse scale numerical
    X_num_unscaled = X_num_scaled * 3
    X_num = scaler.inverse_transform(np.clip(X_num_unscaled, -3, 3))

    # Decode categorical (argmax of one-hot)
    offset = num_numerical
    X_cat_list = []
    for card in cat_cardinalities:
        onehot = samples[:, offset:offset + card]
        cat_idx = np.argmax(onehot, axis=1)
        X_cat_list.append(cat_idx.reshape(-1, 1))
        offset += card

    X_cat = np.hstack(X_cat_list)

    # Combine for ML
    X_ml = np.hstack([X_num, X_cat])
    return X_ml


def assign_labels_knn(X_synthetic, X_train, y_train, n_neighbors=5):
    """Assign labels to synthetic samples using KNN."""
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn.predict(X_synthetic)


def evaluate_classifier(clf_class, clf_params, X_train, y_train, X_test, y_test):
    """Train and evaluate a classifier."""
    clf = clf_class(**clf_params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "y_pred": y_pred,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/adult/final_model.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_synthetic", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("Adult Dataset Evaluation - Hybrid Diffusion vs SMOTE")
    print("=" * 60)

    # Load data
    data = load_adult_data()
    X_train_ml = data["X_train_ml"]
    X_test_ml = data["X_test_ml"]
    X_train_hybrid = data["X_train_hybrid"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    n_synthetic = args.n_synthetic or len(y_train)

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    model = HybridMLPDenoiser(
        num_numerical=checkpoint["model_config"]["num_numerical"],
        cat_cardinalities=checkpoint["model_config"]["cat_cardinalities"],
        hidden_dims=checkpoint["model_config"]["hidden_dims"],
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    diffusion = HybridDiffusion(
        num_numerical=checkpoint["model_config"]["num_numerical"],
        cat_cardinalities=checkpoint["model_config"]["cat_cardinalities"],
        num_timesteps=checkpoint["diffusion_config"]["num_timesteps"],
        beta_schedule=checkpoint["diffusion_config"]["beta_schedule"],
    ).to(args.device)

    print(f"Model loaded (final loss: {checkpoint['final_loss']:.4f})")

    # Generate diffusion samples
    print(f"\nGenerating {n_synthetic} diffusion samples...")
    synthetic_hybrid = generate_diffusion_samples(model, diffusion, n_synthetic, args.device)

    # Decode to ML format
    X_diff = decode_hybrid_samples(
        synthetic_hybrid,
        data["num_numerical"],
        data["cat_cardinalities"],
        data["scaler"],
    )
    y_diff = assign_labels_knn(X_diff, X_train_ml, y_train)
    print(f"Diffusion label distribution: {np.bincount(y_diff)}")

    # Generate SMOTE samples (SMOTE only adds minority class samples to balance)
    print("\nGenerating SMOTE samples...")
    smote = SMOTE(random_state=42)
    X_aug_smote, y_aug_smote = smote.fit_resample(X_train_ml, y_train)
    print(f"SMOTE balanced dataset: {len(X_aug_smote)}, Label distribution: {np.bincount(y_aug_smote)}")

    # Prepare augmented diffusion dataset
    X_aug_diff = np.vstack([X_train_ml, X_diff])
    y_aug_diff = np.concatenate([y_train, y_diff])

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    classifiers = [
        ("Logistic Regression", LogisticRegression, {"max_iter": 1000, "random_state": 42}),
        ("Random Forest", RandomForestClassifier, {"n_estimators": 100, "random_state": 42, "n_jobs": -1}),
    ]

    results = {}
    for clf_name, clf_class, clf_params in classifiers:
        print(f"\n{clf_name}:")
        print("-" * 50)

        # Real -> Real
        res_rr = evaluate_classifier(clf_class, clf_params, X_train_ml, y_train, X_test_ml, y_test)

        # Diffusion only -> Real
        res_diff = evaluate_classifier(clf_class, clf_params, X_diff, y_diff, X_test_ml, y_test)

        # Augmented-Diffusion -> Real
        res_aug_diff = evaluate_classifier(clf_class, clf_params, X_aug_diff, y_aug_diff, X_test_ml, y_test)

        # SMOTE (balanced) -> Real
        res_aug_smote = evaluate_classifier(clf_class, clf_params, X_aug_smote, y_aug_smote, X_test_ml, y_test)

        print(f"  Real -> Real (baseline):      Acc={res_rr['accuracy']:.4f}, F1={res_rr['f1']:.4f}")
        print(f"  Diffusion only -> Real:       Acc={res_diff['accuracy']:.4f}, F1={res_diff['f1']:.4f} ({(res_diff['accuracy'] - res_rr['accuracy'])*100:+.2f}%)")
        print(f"  Augmented-Diffusion -> Real:  Acc={res_aug_diff['accuracy']:.4f}, F1={res_aug_diff['f1']:.4f} ({(res_aug_diff['accuracy'] - res_rr['accuracy'])*100:+.2f}%)")
        print(f"  SMOTE (balanced) -> Real:     Acc={res_aug_smote['accuracy']:.4f}, F1={res_aug_smote['f1']:.4f} ({(res_aug_smote['accuracy'] - res_rr['accuracy'])*100:+.2f}%)")

        results[clf_name] = {
            "real": res_rr,
            "diffusion": res_diff,
            "aug_diffusion": res_aug_diff,
            "aug_smote": res_aug_smote,
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for clf_name, res in results.items():
        baseline = res["real"]["accuracy"]
        diff_acc = res["aug_diffusion"]["accuracy"]
        smote_acc = res["aug_smote"]["accuracy"]

        print(f"\n{clf_name}:")
        print(f"  Baseline:           {baseline:.4f}")
        print(f"  Aug-Diffusion:      {diff_acc:.4f} ({(diff_acc - baseline)*100:+.2f}%)")
        print(f"  Aug-SMOTE:          {smote_acc:.4f} ({(smote_acc - baseline)*100:+.2f}%)")

        if diff_acc > smote_acc:
            print(f"  --> Diffusion beats SMOTE by {(diff_acc - smote_acc)*100:.2f}%")
        elif smote_acc > diff_acc:
            print(f"  --> SMOTE beats Diffusion by {(smote_acc - diff_acc)*100:.2f}%")
        else:
            print(f"  --> Tie")

    # Confusion matrix for best method
    print("\n" + "=" * 60)
    print("Confusion Matrix (Random Forest, Augmented-Diffusion)")
    print("=" * 60)
    print(confusion_matrix(y_test, results["Random Forest"]["aug_diffusion"]["y_pred"]))

    return results


if __name__ == "__main__":
    main()
