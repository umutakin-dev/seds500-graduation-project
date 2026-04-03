"""
Statistical fidelity metrics for comparing real vs synthetic data distributions.

Computes per-column and aggregate metrics:
- Numerical: Wasserstein distance, KL divergence, KS test, mean/std diff
- Categorical: frequency difference (L1 norm)
- Overall: correlation matrix Frobenius norm difference

Usage:
    from distribution_metrics import compute_fidelity_metrics

    metrics = compute_fidelity_metrics(
        real_data, synthetic_data,
        num_col_indices=[0,1,2], cat_col_indices=[3,4],
        cat_cardinalities=[5, 3],
    )
"""

import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats
from scipy.spatial.distance import jensenshannon


def _wasserstein_per_column(real: np.ndarray, synthetic: np.ndarray) -> List[float]:
    """Wasserstein (earth mover's) distance per numerical column."""
    distances = []
    for j in range(real.shape[1]):
        d = stats.wasserstein_distance(real[:, j], synthetic[:, j])
        distances.append(float(d))
    return distances


def _ks_test_per_column(real: np.ndarray, synthetic: np.ndarray) -> List[Dict[str, float]]:
    """Kolmogorov-Smirnov test per numerical column."""
    results = []
    for j in range(real.shape[1]):
        stat, pval = stats.ks_2samp(real[:, j], synthetic[:, j])
        results.append({"statistic": float(stat), "p_value": float(pval)})
    return results


def _kl_divergence_per_column(real: np.ndarray, synthetic: np.ndarray, n_bins: int = 50) -> List[float]:
    """KL divergence per numerical column (binned approximation)."""
    divergences = []
    for j in range(real.shape[1]):
        # Use common bin edges from combined data
        combined = np.concatenate([real[:, j], synthetic[:, j]])
        bin_edges = np.histogram_bin_edges(combined, bins=n_bins)

        hist_real, _ = np.histogram(real[:, j], bins=bin_edges, density=True)
        hist_syn, _ = np.histogram(synthetic[:, j], bins=bin_edges, density=True)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        hist_real = hist_real + eps
        hist_syn = hist_syn + eps

        # Normalize to proper distributions
        hist_real = hist_real / hist_real.sum()
        hist_syn = hist_syn / hist_syn.sum()

        # Jensen-Shannon divergence (symmetric, bounded, more stable than KL)
        jsd = float(jensenshannon(hist_real, hist_syn) ** 2)  # squared to get divergence
        divergences.append(jsd)
    return divergences


def _mean_std_diff(real: np.ndarray, synthetic: np.ndarray) -> Dict[str, List[float]]:
    """Per-column mean and std absolute differences."""
    mean_diff = np.abs(real.mean(axis=0) - synthetic.mean(axis=0)).tolist()
    std_diff = np.abs(real.std(axis=0) - synthetic.std(axis=0)).tolist()
    return {"mean_diff": mean_diff, "std_diff": std_diff}


def _correlation_frobenius(real: np.ndarray, synthetic: np.ndarray) -> float:
    """Frobenius norm of correlation matrix difference (numerical columns only)."""
    if real.shape[1] < 2:
        return 0.0
    corr_real = np.corrcoef(real, rowvar=False)
    corr_syn = np.corrcoef(synthetic, rowvar=False)
    # Handle NaN from constant columns
    corr_real = np.nan_to_num(corr_real, nan=0.0)
    corr_syn = np.nan_to_num(corr_syn, nan=0.0)
    return float(np.linalg.norm(corr_real - corr_syn, "fro"))


def _categorical_frequency_diff(
    real_cat: np.ndarray,
    synthetic_cat: np.ndarray,
    cardinalities: List[int],
) -> List[float]:
    """L1 norm of frequency difference per categorical feature."""
    diffs = []
    for j, card in enumerate(cardinalities):
        # Count frequencies
        real_counts = np.bincount(real_cat[:, j].astype(int), minlength=card).astype(float)
        syn_counts = np.bincount(synthetic_cat[:, j].astype(int), minlength=card).astype(float)

        # Normalize to proportions
        real_freq = real_counts / real_counts.sum() if real_counts.sum() > 0 else real_counts
        syn_freq = syn_counts / syn_counts.sum() if syn_counts.sum() > 0 else syn_counts

        # L1 distance
        diffs.append(float(np.abs(real_freq - syn_freq).sum()))
    return diffs


def compute_fidelity_metrics(
    real_num: np.ndarray,
    synthetic_num: np.ndarray,
    real_cat: Optional[np.ndarray] = None,
    synthetic_cat: Optional[np.ndarray] = None,
    cat_cardinalities: Optional[List[int]] = None,
    num_col_names: Optional[List[str]] = None,
    cat_col_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive fidelity metrics between real and synthetic data.

    Args:
        real_num: Real numerical features (n_samples, n_num_features)
        synthetic_num: Synthetic numerical features
        real_cat: Real categorical indices (n_samples, n_cat_features), optional
        synthetic_cat: Synthetic categorical indices, optional
        cat_cardinalities: Number of categories per feature, optional
        num_col_names: Names for numerical columns
        cat_col_names: Names for categorical columns

    Returns:
        Dict with all metrics, per-column and aggregated
    """
    metrics = {}

    # --- Numerical metrics ---
    if real_num.shape[1] > 0:
        wasserstein = _wasserstein_per_column(real_num, synthetic_num)
        ks_tests = _ks_test_per_column(real_num, synthetic_num)
        jsd = _kl_divergence_per_column(real_num, synthetic_num)
        mean_std = _mean_std_diff(real_num, synthetic_num)
        corr_frob = _correlation_frobenius(real_num, synthetic_num)

        metrics["numerical"] = {
            "wasserstein": wasserstein,
            "ks_test": ks_tests,
            "jensen_shannon_divergence": jsd,
            "mean_diff": mean_std["mean_diff"],
            "std_diff": mean_std["std_diff"],
            "correlation_frobenius_norm": corr_frob,
            "n_columns": real_num.shape[1],
        }

        # Aggregated
        metrics["numerical"]["avg_wasserstein"] = float(np.mean(wasserstein))
        metrics["numerical"]["avg_jsd"] = float(np.mean(jsd))
        metrics["numerical"]["avg_ks_statistic"] = float(np.mean([k["statistic"] for k in ks_tests]))
        metrics["numerical"]["pct_ks_pass_005"] = float(
            np.mean([k["p_value"] > 0.05 for k in ks_tests]) * 100
        )

        if num_col_names:
            metrics["numerical"]["column_names"] = num_col_names

    # --- Categorical metrics ---
    if real_cat is not None and real_cat.shape[1] > 0 and cat_cardinalities:
        freq_diffs = _categorical_frequency_diff(real_cat, synthetic_cat, cat_cardinalities)
        metrics["categorical"] = {
            "frequency_l1_diff": freq_diffs,
            "avg_frequency_l1_diff": float(np.mean(freq_diffs)),
            "n_columns": real_cat.shape[1],
        }
        if cat_col_names:
            metrics["categorical"]["column_names"] = cat_col_names

    # --- Overall summary ---
    summary = {}
    if "numerical" in metrics:
        summary["avg_wasserstein"] = metrics["numerical"]["avg_wasserstein"]
        summary["avg_jsd"] = metrics["numerical"]["avg_jsd"]
        summary["correlation_frobenius"] = metrics["numerical"]["correlation_frobenius_norm"]
    if "categorical" in metrics:
        summary["avg_cat_freq_diff"] = metrics["categorical"]["avg_frequency_l1_diff"]
    metrics["summary"] = summary

    return metrics


def format_fidelity_table(metrics: Dict[str, Any]) -> str:
    """Format fidelity metrics as a markdown summary."""
    lines = []
    summary = metrics.get("summary", {})

    lines.append("### Statistical Fidelity Summary")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    for k, v in summary.items():
        lines.append(f"| {k} | {v:.4f} |")

    if "numerical" in metrics:
        n = metrics["numerical"]
        lines.append(f"\n**Numerical:** {n['n_columns']} columns, "
                     f"avg Wasserstein={n['avg_wasserstein']:.4f}, "
                     f"avg JSD={n['avg_jsd']:.4f}, "
                     f"KS pass rate (p>0.05)={n['pct_ks_pass_005']:.0f}%")

    if "categorical" in metrics:
        c = metrics["categorical"]
        lines.append(f"**Categorical:** {c['n_columns']} columns, "
                     f"avg freq L1 diff={c['avg_frequency_l1_diff']:.4f}")

    return "\n".join(lines)
