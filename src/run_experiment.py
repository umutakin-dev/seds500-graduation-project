"""
Phase 2 Experiment Runner — orchestrates full pipeline for one (dataset, method) pair.

Pipeline: load data → train/generate → evaluate (utility + fidelity + privacy)

Usage:
    # Run our TabDDPM on insurance dataset
    python src/run_experiment.py --dataset insurance --method our_tabddpm --device cuda

    # Run CTGAN on adult dataset
    python src/run_experiment.py --dataset adult --method ctgan

    # Run all methods on all datasets
    python src/run_experiment.py --dataset all --method all --device cuda

Methods: our_tabddpm, vanilla_tabddpm, ctgan, smogn
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from datasets import load_dataset, DATASET_REGISTRY
from evaluation_framework import evaluate_synthetic_data, format_results_table
from distribution_metrics import compute_fidelity_metrics, format_fidelity_table

RESULTS_DIR = Path(__file__).parent.parent / "experiments" / "phase2"


# =============================================================================
# TabDDPM Training & Generation
# =============================================================================

def _auto_hidden_dims(total_dims: int) -> list:
    """Auto-scale model capacity based on input dimensionality."""
    if total_dims <= 20:
        return [256, 256, 256]
    elif total_dims <= 50:
        return [512, 512, 512]
    else:
        return [512, 512, 512, 512]


def _auto_epochs(n_samples: int) -> int:
    """Auto-scale training epochs based on dataset size."""
    if n_samples < 500:
        return 1500
    elif n_samples < 2000:
        return 1000
    elif n_samples < 10000:
        return 750
    else:
        return 500


def train_tabddpm(data: dict, device: str, epochs: int = None, is_vanilla: bool = False) -> dict:
    """Train TabDDPM model and return checkpoint dict."""
    from diffusion_tabddpm import HybridDiffusionTabDDPM, GaussianDiffusion
    from models import MLPDenoiser

    X_num_train = data["X_num_train"]
    X_cat_train = data["X_cat_train"]
    cat_cardinalities = data["cat_cardinalities"]
    num_numerical = data["d_numerical"]
    has_cat = len(cat_cardinalities) > 0

    # Include target as part of numerical features for generation
    y_train = data["y_train"]
    if data["task_type"] == "regression":
        # Append target as extra numerical column
        y_scaled = y_train.float().unsqueeze(1)
        # Scale target to [-1, 1] range
        y_min, y_max = y_scaled.min(), y_scaled.max()
        if y_max > y_min:
            y_scaled = 2 * (y_scaled - y_min) / (y_max - y_min) - 1
        X_num_with_target = torch.cat([X_num_train, y_scaled], dim=1)
        num_numerical_total = num_numerical + 1
        target_stats = {"y_min": float(y_min), "y_max": float(y_max)}
    else:
        X_num_with_target = X_num_train
        num_numerical_total = num_numerical
        target_stats = None

    # For numeric-only datasets, add a dummy binary categorical to avoid empty-cat edge cases
    used_dummy_cat = False
    if not has_cat:
        cat_cardinalities = [2]  # dummy binary feature
        X_cat_train = torch.zeros(len(X_num_train), 1, dtype=torch.long)
        has_cat = True
        used_dummy_cat = True

    total_dims = num_numerical_total + sum(cat_cardinalities)
    hidden_dims = _auto_hidden_dims(total_dims)
    if epochs is None:
        epochs = _auto_epochs(len(X_num_train))

    d_in = num_numerical_total + sum(cat_cardinalities)
    model = MLPDenoiser(d_in=d_in, hidden_dims=hidden_dims, dropout=0.1).to(device)

    diffusion = HybridDiffusionTabDDPM(
        num_numerical=num_numerical_total,
        cat_cardinalities=cat_cardinalities,
        num_timesteps=1000,
        beta_schedule="cosine",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    cat_train = X_cat_train.to(device)
    batch_size = min(128, len(X_num_train))

    best_loss = float("inf")
    best_state = None

    method_name = "Vanilla TabDDPM" if is_vanilla else "Our TabDDPM"
    print(f"  Training {method_name}: {epochs} epochs, hidden={hidden_dims}, d_in={d_in}")

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(X_num_train))
        total_loss = 0
        n_batches = 0

        for start in range(0, len(X_num_train), batch_size):
            end = min(start + batch_size, len(X_num_train))
            batch_idx = indices[start:end]

            x_num = X_num_with_target[batch_idx].to(device)
            c_idx = cat_train[batch_idx]

            optimizer.zero_grad()

            losses = diffusion.training_loss(model, x_num, c_idx)

            loss = losses["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs} loss={avg_loss:.4f}")

    # Restore best model
    model.load_state_dict(best_state)
    model.eval()

    return {
        "model": model,
        "diffusion": diffusion,
        "num_numerical_total": num_numerical_total,
        "cat_cardinalities": cat_cardinalities,
        "target_stats": target_stats,
        "device": device,
        "epochs": epochs,
        "best_loss": best_loss,
        "used_dummy_cat": used_dummy_cat,
    }


def generate_tabddpm(checkpoint: dict, n_samples: int, task_type: str) -> tuple:
    """Generate synthetic data from trained TabDDPM model."""
    model = checkpoint["model"]
    diffusion = checkpoint["diffusion"]
    device = checkpoint["device"]
    num_numerical_total = checkpoint["num_numerical_total"]
    target_stats = checkpoint["target_stats"]
    used_dummy_cat = checkpoint.get("used_dummy_cat", False)

    model.eval()
    with torch.no_grad():
        x_num, cat_indices = diffusion.sample(model, batch_size=n_samples, device=device)

    x_num = x_num.cpu().numpy()
    cat_indices = cat_indices.cpu().numpy()

    # Strip dummy categorical if we added one for numeric-only datasets
    if used_dummy_cat:
        cat_indices = np.empty((n_samples, 0), dtype=np.int64)

    if task_type == "regression" and target_stats is not None:
        # Extract target (last numerical column) and inverse scale
        y_syn = x_num[:, -1]
        x_num_features = x_num[:, :-1]
        y_min, y_max = target_stats["y_min"], target_stats["y_max"]
        y_syn = (y_syn + 1) / 2 * (y_max - y_min) + y_min
    else:
        x_num_features = x_num
        y_syn = None

    return x_num_features, cat_indices, y_syn


# =============================================================================
# CTGAN
# =============================================================================

def run_ctgan(data: dict) -> tuple:
    """Train CTGAN and generate synthetic data."""
    from ctgan import CTGAN
    import pandas as pd

    X_num = data["X_num_train"].numpy()
    X_cat = data["X_cat_train"].numpy()
    y = data["y_train"].numpy()
    cat_cols = [f"cat_{i}" for i in range(X_cat.shape[1])]
    num_cols = [f"num_{i}" for i in range(X_num.shape[1])]

    # Build DataFrame
    df_dict = {}
    for i, c in enumerate(num_cols):
        df_dict[c] = X_num[:, i]
    for i, c in enumerate(cat_cols):
        df_dict[c] = X_cat[:, i].astype(str)
    df_dict["target"] = y
    df = pd.DataFrame(df_dict)

    discrete_cols = cat_cols
    if data["task_type"] == "classification":
        df["target"] = df["target"].astype(str)
        discrete_cols = cat_cols + ["target"]

    n_samples = len(df)
    epochs = 300 if n_samples > 5000 else 500

    print(f"  Training CTGAN: {epochs} epochs, {n_samples} samples")
    ctgan = CTGAN(epochs=epochs, verbose=False)
    ctgan.fit(df, discrete_columns=discrete_cols)

    syn_df = ctgan.sample(n_samples)

    X_num_syn = syn_df[num_cols].values.astype(np.float32)
    X_cat_syn = np.zeros((n_samples, len(cat_cols)), dtype=np.int64)
    for i, c in enumerate(cat_cols):
        # Map back to integer indices
        unique_vals = sorted(df[c].unique())
        val_to_idx = {v: idx for idx, v in enumerate(unique_vals)}
        X_cat_syn[:, i] = syn_df[c].map(val_to_idx).fillna(0).astype(np.int64).values

    if data["task_type"] == "regression":
        y_syn = syn_df["target"].values.astype(np.float32)
    else:
        unique_labels = sorted(df["target"].unique())
        label_to_idx = {v: idx for idx, v in enumerate(unique_labels)}
        y_syn = syn_df["target"].map(label_to_idx).fillna(0).astype(np.int64).values

    return X_num_syn, X_cat_syn, y_syn


# =============================================================================
# SMOGN
# =============================================================================

def run_smogn(data: dict) -> tuple:
    """Run SMOGN augmentation. Only works for regression."""
    import pandas as pd

    if data["task_type"] != "regression":
        # For classification, use simple oversampling as SMOGN baseline
        return _simple_oversample(data)

    try:
        import smogn
    except ImportError:
        print("  SMOGN not available, using simple noise augmentation")
        return _noise_augmentation(data)

    X_num = data["X_num_train"].numpy()
    X_cat = data["X_cat_train"].numpy()
    y = data["y_train"].numpy()

    # Build DataFrame
    cols = [f"num_{i}" for i in range(X_num.shape[1])]
    cat_names = [f"cat_{i}" for i in range(X_cat.shape[1])]
    cols += cat_names

    all_data = np.hstack([X_num, X_cat])
    df = pd.DataFrame(all_data, columns=cols)
    df["target"] = y

    print(f"  Running SMOGN on {len(df)} samples...")
    try:
        result = smogn.smoter(data=df, y="target", k=5, samp_method="extreme")

        X_syn_all = result.drop("target", axis=1).values.astype(np.float32)
        y_syn = result["target"].values.astype(np.float32)

        n_num = X_num.shape[1]
        X_num_syn = X_syn_all[:, :n_num]
        X_cat_syn = np.round(X_syn_all[:, n_num:]).astype(np.int64)

        # Clamp categorical indices
        for i, card in enumerate(data["cat_cardinalities"]):
            X_cat_syn[:, i] = np.clip(X_cat_syn[:, i], 0, card - 1)

        return X_num_syn, X_cat_syn, y_syn
    except Exception as e:
        print(f"  SMOGN failed: {e}. Using noise augmentation.")
        return _noise_augmentation(data)


def _simple_oversample(data: dict) -> tuple:
    """Simple random oversampling for classification tasks."""
    X_num = data["X_num_train"].numpy()
    X_cat = data["X_cat_train"].numpy()
    y = data["y_train"].numpy()
    n = len(y)

    # Random sample with replacement
    indices = np.random.choice(n, size=n, replace=True)
    # Add small noise to numerical features
    noise = np.random.normal(0, 0.01, size=(n, X_num.shape[1]))
    return X_num[indices] + noise.astype(np.float32), X_cat[indices], y[indices]


def _noise_augmentation(data: dict) -> tuple:
    """Fallback: add Gaussian noise to real data."""
    X_num = data["X_num_train"].numpy()
    X_cat = data["X_cat_train"].numpy()
    y = data["y_train"].numpy()
    noise = np.random.normal(0, 0.05, size=X_num.shape).astype(np.float32)
    return X_num + noise, X_cat.copy(), y.copy()


# =============================================================================
# Combine features for ML evaluation
# =============================================================================

def _prepare_features(X_num: np.ndarray, X_cat: np.ndarray, cat_cardinalities: list) -> np.ndarray:
    """Combine numerical and one-hot encoded categorical features for ML models."""
    if X_cat.shape[1] == 0:
        return X_num

    # One-hot encode categoricals
    onehot_parts = []
    for i, card in enumerate(cat_cardinalities):
        col = X_cat[:, i].astype(int)
        col = np.clip(col, 0, card - 1)
        oh = np.zeros((len(col), card), dtype=np.float32)
        oh[np.arange(len(col)), col] = 1.0
        onehot_parts.append(oh)

    X_cat_oh = np.hstack(onehot_parts)
    return np.hstack([X_num, X_cat_oh])


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_single_experiment(
    dataset_name: str,
    method: str,
    device: str = "cpu",
    epochs: int = None,
    save_results: bool = True,
) -> dict:
    """
    Run a single experiment: one method on one dataset.

    Returns dict with utility, fidelity, and timing results.
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {method} on {dataset_name}")
    print(f"{'='*60}")

    # Determine preprocessing based on method
    if method == "vanilla_tabddpm":
        scaler_type = "quantile"
        outlier_clip = False
    else:
        scaler_type = "minmax"
        outlier_clip = True

    # Load dataset
    data = load_dataset(dataset_name, scaler_type=scaler_type, outlier_clip=outlier_clip)
    cat_cardinalities = data["cat_cardinalities"]
    task_type = data["task_type"]
    n_train = data["n_train"]

    start_time = time.time()

    # --- Generate synthetic data ---
    if method in ("our_tabddpm", "vanilla_tabddpm"):
        is_vanilla = method == "vanilla_tabddpm"
        checkpoint = train_tabddpm(data, device, epochs=epochs, is_vanilla=is_vanilla)
        X_num_syn, X_cat_syn, y_syn = generate_tabddpm(checkpoint, n_train, task_type)
        training_time = time.time() - start_time
        extra_info = {"epochs": checkpoint["epochs"], "best_loss": checkpoint["best_loss"], "used_dummy_cat": checkpoint.get("used_dummy_cat", False)}
    elif method == "ctgan":
        X_num_syn, X_cat_syn, y_syn = run_ctgan(data)
        training_time = time.time() - start_time
        extra_info = {}
    elif method == "smogn":
        X_num_syn, X_cat_syn, y_syn = run_smogn(data)
        training_time = time.time() - start_time
        extra_info = {}
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"\n  Generation complete. Time: {training_time:.1f}s")
    print(f"  Synthetic: {len(X_num_syn)} samples")

    # Handle classification target for diffusion methods
    if task_type == "classification" and method in ("our_tabddpm", "vanilla_tabddpm"):
        # Assign labels via nearest neighbor from real data
        from sklearn.neighbors import KNeighborsClassifier
        X_real_combined = _prepare_features(
            data["X_num_train"].numpy(), data["X_cat_train"].numpy(), cat_cardinalities
        )
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_real_combined, data["y_train"].numpy())
        X_syn_combined = _prepare_features(X_num_syn, X_cat_syn, cat_cardinalities)
        y_syn = knn.predict(X_syn_combined)

    # --- Prepare features for ML evaluation ---
    X_real_train = _prepare_features(
        data["X_num_train"].numpy(), data["X_cat_train"].numpy(), cat_cardinalities
    )
    X_real_test = _prepare_features(
        data["X_num_test"].numpy(), data["X_cat_test"].numpy(), cat_cardinalities
    )
    X_syn = _prepare_features(X_num_syn, X_cat_syn, cat_cardinalities)

    y_real_train = data["y_train"].numpy()
    y_real_test = data["y_test"].numpy()

    # --- Utility evaluation ---
    print("  Evaluating utility...")
    utility = evaluate_synthetic_data(
        X_real_train, y_real_train,
        X_real_test, y_real_test,
        X_syn, y_syn,
        task_type=task_type,
    )

    # --- Fidelity metrics ---
    print("  Computing fidelity metrics...")
    fidelity = compute_fidelity_metrics(
        real_num=data["X_num_train"].numpy(),
        synthetic_num=X_num_syn,
        real_cat=data["X_cat_train"].numpy() if len(cat_cardinalities) > 0 else None,
        synthetic_cat=X_cat_syn if len(cat_cardinalities) > 0 else None,
        cat_cardinalities=cat_cardinalities if len(cat_cardinalities) > 0 else None,
        num_col_names=data["num_cols"],
        cat_col_names=data["cat_cols"],
    )

    # --- Compile results ---
    results = {
        "dataset": dataset_name,
        "dataset_info": {
            "name": data["dataset_info"]["name"],
            "id": data["dataset_info"]["id"],
            "task": task_type,
            "n_train": n_train,
            "n_test": data["n_test"],
            "d_numerical": data["d_numerical"],
            "d_categorical": data["d_categorical"],
            "d_onehot": data["d_onehot"],
            "total_dims": data["d_numerical"] + data["d_onehot"],
        },
        "method": method,
        "preprocessing": {"scaler": scaler_type, "outlier_clip": outlier_clip},
        "utility": utility,
        "fidelity": fidelity,
        "timing": {"training_seconds": training_time},
        "extra": extra_info,
    }

    if save_results:
        _save_results(results)

    return results


def _save_results(results: dict):
    """Save results to experiments/phase2/{dataset}_{method}/."""
    out_dir = RESULTS_DIR / f"{results['dataset']}_{results['method']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = out_dir / "RESULTS.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save markdown report
    md_path = out_dir / "RESULTS.md"
    with open(md_path, "w") as f:
        info = results["dataset_info"]
        f.write(f"# {results['method']} on {info['name']}\n\n")
        f.write(f"**Dataset:** {info['name']} ({info['id']})\n")
        f.write(f"**Task:** {info['task']}\n")
        f.write(f"**Dimensions:** {info['d_numerical']} num + {info['d_categorical']} cat = {info['total_dims']} total\n")
        f.write(f"**Samples:** {info['n_train']} train / {info['n_test']} test\n")
        f.write(f"**Preprocessing:** {results['preprocessing']['scaler']}, clip={results['preprocessing']['outlier_clip']}\n")
        f.write(f"**Training time:** {results['timing']['training_seconds']:.1f}s\n\n")
        f.write("## Utility\n")
        f.write(format_results_table(results["utility"]))
        f.write("\n\n## Fidelity\n")
        f.write(format_fidelity_table(results["fidelity"]))
        f.write("\n")

    print(f"  Results saved to {out_dir}/")


# =============================================================================
# CLI
# =============================================================================

ALL_METHODS = ["our_tabddpm", "vanilla_tabddpm", "ctgan", "smogn"]

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Experiment Runner")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name or 'all'")
    parser.add_argument("--method", type=str, required=True,
                        help="Method name or 'all'")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for training (cuda/cpu/mps)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = "cpu"

    datasets = list(DATASET_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]
    methods = ALL_METHODS if args.method == "all" else [args.method]

    total = len(datasets) * len(methods)
    completed = 0

    for ds in datasets:
        for method in methods:
            try:
                run_single_experiment(ds, method, args.device, args.epochs)
                completed += 1
                print(f"\n  [{completed}/{total}] Done.")
            except Exception as e:
                print(f"\n  ERROR: {ds} + {method}: {e}")
                import traceback
                traceback.print_exc()
                completed += 1

    print(f"\n{'='*60}")
    print(f"Phase 2 experiments complete: {completed}/{total}")
    print(f"Results in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
