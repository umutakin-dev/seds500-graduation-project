"""
Evaluate class-conditional hybrid diffusion model on Adult dataset.

This script generates samples specifically for the minority class and compares:
- Class-conditional diffusion (minority class only)
- SMOTE (minority class only)
- Original training data

Usage:
    python src/evaluate_adult_conditional.py --device cuda --guidance_scale 2.0
"""

import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

from diffusion import HybridDiffusion
from models import HybridMLPDenoiser


def load_adult_data():
    """Load Adult dataset with same preprocessing as training."""
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

    return {
        "X_num_train": X_num_train,
        "X_num_test": X_num_test,
        "X_num_train_scaled": X_num_train_scaled,
        "X_num_test_scaled": X_num_test_scaled,
        "X_cat_train": X_cat_train,
        "X_cat_test": X_cat_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_cardinalities": cat_cardinalities,
    }


def decode_samples(samples, num_numerical, cat_cardinalities, scaler):
    """Decode generated samples back to original format."""
    samples = samples.cpu().numpy()

    # Decode numerical (inverse transform)
    X_num = samples[:, :num_numerical] * 3  # Undo clipping normalization
    X_num = scaler.inverse_transform(X_num)

    # Decode categorical (argmax of one-hot)
    offset = num_numerical
    X_cat = []
    for card in cat_cardinalities:
        onehot = samples[:, offset:offset + card]
        cat_idx = np.argmax(onehot, axis=1)
        X_cat.append(cat_idx)
        offset += card
    X_cat = np.column_stack(X_cat)

    return X_num, X_cat


def prepare_features(X_num, X_cat):
    """Combine numerical and categorical for ML model."""
    return np.hstack([X_num, X_cat.astype(np.float32)])


def evaluate_classifier(clf_name, clf, X_train, y_train, X_test, y_test):
    """Train and evaluate a classifier."""
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return acc, f1


def main(args):
    # Load data
    data = load_adult_data()

    # Load model checkpoint
    checkpoint_path = Path("checkpoints/adult_conditional/final_model.pt")
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please run train_adult_conditional.py first.")
        return

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)

    model_config = checkpoint["model_config"]
    diffusion_config = checkpoint["diffusion_config"]

    # Create model
    model = HybridMLPDenoiser(
        num_numerical=model_config["num_numerical"],
        cat_cardinalities=model_config["cat_cardinalities"],
        hidden_dims=model_config["hidden_dims"],
        num_classes=model_config["num_classes"],
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create diffusion
    diffusion = HybridDiffusion(
        num_numerical=model_config["num_numerical"],
        cat_cardinalities=model_config["cat_cardinalities"],
        num_timesteps=diffusion_config["num_timesteps"],
        beta_schedule=diffusion_config["beta_schedule"],
    ).to(args.device)

    # Calculate how many minority samples to generate (to balance the dataset)
    n_class_0 = (data["y_train"] == 0).sum()
    n_class_1 = (data["y_train"] == 1).sum()
    n_to_generate = n_class_0 - n_class_1  # Generate this many class 1 samples
    print(f"\nClass distribution: class 0 = {n_class_0}, class 1 = {n_class_1}")
    print(f"Will generate {n_to_generate} minority class samples")

    # Generate minority class samples with class-conditional diffusion
    print(f"\nGenerating {n_to_generate} samples for minority class (guidance_scale={args.guidance_scale})...")
    y_target = torch.ones(n_to_generate, dtype=torch.long, device=args.device)  # Class 1

    with torch.no_grad():
        if args.guidance_scale > 1.0:
            # Use classifier-free guidance
            samples = diffusion.sample_with_guidance(
                model,
                batch_size=n_to_generate,
                y=y_target,
                guidance_scale=args.guidance_scale,
                device=args.device,
            )
        else:
            # Regular conditional sampling
            samples = diffusion.sample(
                model,
                batch_size=n_to_generate,
                y=y_target,
                device=args.device,
            )

    # Decode generated samples
    X_num_gen, X_cat_gen = decode_samples(
        samples,
        model_config["num_numerical"],
        model_config["cat_cardinalities"],
        data["scaler"],
    )

    # Prepare datasets for evaluation
    X_train_orig = prepare_features(data["X_num_train"], data["X_cat_train"])
    X_test = prepare_features(data["X_num_test"], data["X_cat_test"])
    y_train_orig = data["y_train"]
    y_test = data["y_test"]

    # Augmented with diffusion (minority class only)
    X_gen = prepare_features(X_num_gen, X_cat_gen)
    y_gen = np.ones(n_to_generate, dtype=int)  # All generated samples are class 1
    X_train_diff = np.vstack([X_train_orig, X_gen])
    y_train_diff = np.concatenate([y_train_orig, y_gen])

    # Augmented with SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_orig, y_train_orig)

    print(f"\nDataset sizes:")
    print(f"  Original: {len(X_train_orig)} (class 0: {(y_train_orig==0).sum()}, class 1: {(y_train_orig==1).sum()})")
    print(f"  + Diffusion: {len(X_train_diff)} (class 0: {(y_train_diff==0).sum()}, class 1: {(y_train_diff==1).sum()})")
    print(f"  + SMOTE: {len(X_train_smote)} (class 0: {(y_train_smote==0).sum()}, class 1: {(y_train_smote==1).sum()})")

    # Evaluate classifiers
    print("\n" + "=" * 60)
    print("ML EFFICIENCY EVALUATION")
    print("=" * 60)

    results = {}

    for clf_name, clf_class, clf_kwargs in [
        ("Random Forest", RandomForestClassifier, {"n_estimators": 100, "random_state": 42, "n_jobs": -1}),
        ("Logistic Regression", LogisticRegression, {"max_iter": 1000, "random_state": 42}),
    ]:
        print(f"\n{clf_name}:")
        print("-" * 40)

        # Original
        clf = clf_class(**clf_kwargs)
        acc_orig, f1_orig = evaluate_classifier(clf_name, clf, X_train_orig, y_train_orig, X_test, y_test)
        print(f"  Original:     Accuracy={acc_orig:.4f}, F1={f1_orig:.4f}")

        # Diffusion augmented
        clf = clf_class(**clf_kwargs)
        acc_diff, f1_diff = evaluate_classifier(clf_name, clf, X_train_diff, y_train_diff, X_test, y_test)
        diff_acc = acc_diff - acc_orig
        diff_f1 = f1_diff - f1_orig
        print(f"  + Diffusion:  Accuracy={acc_diff:.4f} ({diff_acc:+.4f}), F1={f1_diff:.4f} ({diff_f1:+.4f})")

        # SMOTE augmented
        clf = clf_class(**clf_kwargs)
        acc_smote, f1_smote = evaluate_classifier(clf_name, clf, X_train_smote, y_train_smote, X_test, y_test)
        diff_acc_smote = acc_smote - acc_orig
        diff_f1_smote = f1_smote - f1_orig
        print(f"  + SMOTE:      Accuracy={acc_smote:.4f} ({diff_acc_smote:+.4f}), F1={f1_smote:.4f} ({diff_f1_smote:+.4f})")

        results[clf_name] = {
            "original": {"accuracy": acc_orig, "f1": f1_orig},
            "diffusion": {"accuracy": acc_diff, "f1": f1_diff},
            "smote": {"accuracy": acc_smote, "f1": f1_smote},
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nClass-conditional diffusion generates samples with known labels.")
    print(f"Guidance scale: {args.guidance_scale}")

    for clf_name in results:
        r = results[clf_name]
        diff_beats_smote_acc = r["diffusion"]["accuracy"] > r["smote"]["accuracy"]
        diff_beats_smote_f1 = r["diffusion"]["f1"] > r["smote"]["f1"]
        status = "BETTER" if diff_beats_smote_acc else "WORSE"
        print(f"\n{clf_name}: Diffusion vs SMOTE = {status}")
        print(f"  Accuracy: {r['diffusion']['accuracy']:.4f} vs {r['smote']['accuracy']:.4f}")
        print(f"  F1 Score: {r['diffusion']['f1']:.4f} vs {r['smote']['f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale (1.0 = no guidance, >1.0 = stronger class conditioning)")
    args = parser.parse_args()

    main(args)
