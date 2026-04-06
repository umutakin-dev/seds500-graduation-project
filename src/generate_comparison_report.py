"""
Phase 2 Comparison Report Generator.

Reads all RESULTS.json files from experiments/phase2/ and produces:
- Cross-dataset x cross-method comparison tables
- Ablation analysis (Our vs Vanilla TabDDPM)
- Scaling analysis (performance vs dimensionality)
- Figures (bar charts, line charts, heatmaps)
- Combined COMPARISON.md report

Usage:
    python src/generate_comparison_report.py
    python src/generate_comparison_report.py --output experiments/phase2/COMPARISON.md
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

PHASE2_DIR = Path(__file__).parent.parent / "experiments" / "phase2"

# Ordered by dimensionality for scaling analysis
DATASET_ORDER = [
    "iris", "california", "insurance", "maintenance",
    "steel", "bank", "credit", "supply_chain", "news", "adult", "ames",
]

METHOD_ORDER = ["our_tabddpm", "vanilla_tabddpm", "ctgan", "smogn"]

METHOD_LABELS = {
    "our_tabddpm": "Our TabDDPM",
    "vanilla_tabddpm": "Vanilla TabDDPM",
    "ctgan": "CTGAN",
    "smogn": "SMOGN",
}

DATASET_LABELS = {
    "iris": "Iris",
    "california": "California",
    "insurance": "Insurance",
    "maintenance": "Maintenance",
    "steel": "Steel",
    "bank": "Bank",
    "credit": "Credit",
    "supply_chain": "Supply Chain",
    "news": "News",
    "adult": "Adult",
    "ames": "Ames Housing",
}


# =============================================================================
# Load Results
# =============================================================================

def load_all_results() -> Dict[str, Dict[str, Any]]:
    """Load all RESULTS.json files. Returns {dataset_method: result_dict}."""
    results = {}
    for json_path in PHASE2_DIR.glob("*/RESULTS.json"):
        with open(json_path) as f:
            data = json.load(f)
        key = f"{data['dataset']}_{data['method']}"
        results[key] = data
    return results


def get_available_datasets(results: dict) -> List[str]:
    """Return datasets that have at least one result, in order."""
    available = set()
    for key in results:
        ds = key.rsplit("_", 1)[0]
        # Handle multi-word dataset names
        for d in DATASET_ORDER:
            if key.startswith(d + "_"):
                available.add(d)
                break
    return [d for d in DATASET_ORDER if d in available]


def get_result(results: dict, dataset: str, method: str) -> Optional[dict]:
    """Get result for a specific dataset+method combo."""
    key = f"{dataset}_{method}"
    return results.get(key)


# =============================================================================
# Build Comparison Tables
# =============================================================================

def build_utility_table(
    results: dict,
    datasets: List[str],
    scenario: str = "replacement",
) -> Dict[str, Dict[str, Optional[float]]]:
    """Build method x dataset table for a utility scenario.
    Returns {method: {dataset: pct_of_baseline or None}}."""
    table = {}
    for method in METHOD_ORDER:
        table[method] = {}
        for ds in datasets:
            r = get_result(results, ds, method)
            if r and "utility" in r:
                summary = r["utility"]["summary"]
                baseline_key = list(summary["baseline"].keys())[0]  # avg_r2 or avg_accuracy
                baseline_val = summary["baseline"][baseline_key]
                scenario_val = summary[scenario].get(baseline_key)
                pct = summary[scenario].get("pct_of_baseline")
                # Skip nonsensical results (negative baseline, extreme values)
                if baseline_val <= 0 or pct is None or abs(pct) > 200:
                    table[method][ds] = None
                else:
                    table[method][ds] = pct
            else:
                table[method][ds] = None
    return table


def build_fidelity_table(
    results: dict,
    datasets: List[str],
    metric: str = "avg_wasserstein",
) -> Dict[str, Dict[str, Optional[float]]]:
    """Build method x dataset table for a fidelity metric."""
    table = {}
    for method in METHOD_ORDER:
        table[method] = {}
        for ds in datasets:
            r = get_result(results, ds, method)
            if r and "fidelity" in r:
                summary = r["fidelity"].get("summary", {})
                val = summary.get(metric)
                if val is not None and not np.isinf(val) and abs(val) < 1e6:
                    table[method][ds] = float(val)
                else:
                    table[method][ds] = None
            else:
                table[method][ds] = None
    return table


def build_privacy_table(
    results: dict,
    datasets: List[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Build method x dataset table for privacy AUC."""
    table = {}
    for method in METHOD_ORDER:
        table[method] = {}
        for ds in datasets:
            r = get_result(results, ds, method)
            if r and "privacy" in r:
                auc = r["privacy"].get("attack_auc")
                if auc is not None:
                    table[method][ds] = float(auc)
                else:
                    table[method][ds] = None
            else:
                table[method][ds] = None
    return table


def get_dataset_dims(results: dict, datasets: List[str]) -> Dict[str, int]:
    """Get total dimensionality for each dataset."""
    dims = {}
    for ds in datasets:
        for method in METHOD_ORDER:
            r = get_result(results, ds, method)
            if r and "dataset_info" in r:
                dims[ds] = r["dataset_info"]["total_dims"]
                break
    return dims


def get_dataset_info_row(results: dict, ds: str) -> dict:
    """Get dataset info from any available result."""
    for method in METHOD_ORDER:
        r = get_result(results, ds, method)
        if r and "dataset_info" in r:
            return r["dataset_info"]
    return {}


# =============================================================================
# Format Tables as Markdown
# =============================================================================

def format_utility_md(
    table: Dict[str, Dict[str, Optional[float]]],
    datasets: List[str],
    dims: Dict[str, int],
    title: str,
) -> str:
    """Format utility table as markdown."""
    lines = [f"### {title}\n"]

    # Header
    header = "| Method |"
    separator = "| --- |"
    for ds in datasets:
        d = dims.get(ds, "?")
        header += f" {DATASET_LABELS.get(ds, ds)} ({d}d) |"
        separator += " --- |"
    header += " **Avg** |"
    separator += " --- |"
    lines.append(header)
    lines.append(separator)

    # Rows
    for method in METHOD_ORDER:
        row = f"| **{METHOD_LABELS.get(method, method)}** |"
        values = []
        for ds in datasets:
            val = table[method].get(ds)
            if val is not None:
                # Bold the best per column
                row += f" {val:.1f}% |"
                values.append(val)
            else:
                row += " — |"
        avg = np.mean(values) if values else 0
        row += f" **{avg:.1f}%** |"
        lines.append(row)

    # Best method per dataset
    row = "| **Best** |"
    for ds in datasets:
        best_method = None
        best_val = -999
        for method in METHOD_ORDER:
            val = table[method].get(ds)
            if val is not None and val > best_val:
                best_val = val
                best_method = method
        if best_method:
            label = METHOD_LABELS.get(best_method, best_method).split()[0]
            row += f" {label} |"
        else:
            row += " — |"
    row += " |"
    lines.append(row)

    return "\n".join(lines)


def format_ablation_md(
    table: Dict[str, Dict[str, Optional[float]]],
    datasets: List[str],
    dims: Dict[str, int],
) -> str:
    """Format ablation (Our vs Vanilla) as markdown."""
    lines = ["### Ablation: Our Improvements vs Vanilla TabDDPM\n"]

    header = "| |"
    separator = "| --- |"
    for ds in datasets:
        d = dims.get(ds, "?")
        header += f" {DATASET_LABELS.get(ds, ds)} ({d}d) |"
        separator += " --- |"
    header += " **Avg** |"
    separator += " --- |"
    lines.append(header)
    lines.append(separator)

    # Our row
    our_vals = []
    row = "| **Our TabDDPM** |"
    for ds in datasets:
        val = table["our_tabddpm"].get(ds)
        if val is not None:
            row += f" {val:.1f}% |"
            our_vals.append((ds, val))
        else:
            row += " — |"
    row += f" **{np.mean([v for _, v in our_vals]):.1f}%** |" if our_vals else " — |"
    lines.append(row)

    # Vanilla row
    van_vals = []
    row = "| **Vanilla TabDDPM** |"
    for ds in datasets:
        val = table["vanilla_tabddpm"].get(ds)
        if val is not None:
            row += f" {val:.1f}% |"
            van_vals.append((ds, val))
        else:
            row += " — |"
    row += f" **{np.mean([v for _, v in van_vals]):.1f}%** |" if van_vals else " — |"
    lines.append(row)

    # Delta row
    row = "| **Improvement** |"
    deltas = []
    for ds in datasets:
        our = table["our_tabddpm"].get(ds)
        van = table["vanilla_tabddpm"].get(ds)
        if our is not None and van is not None:
            delta = our - van
            deltas.append(delta)
            sign = "+" if delta >= 0 else ""
            row += f" {sign}{delta:.1f}% |"
        else:
            row += " — |"
    avg_delta = np.mean(deltas) if deltas else 0
    sign = "+" if avg_delta >= 0 else ""
    row += f" **{sign}{avg_delta:.1f}%** |"
    lines.append(row)

    return "\n".join(lines)


# =============================================================================
# Figures
# =============================================================================

def plot_replacement_comparison(
    table: Dict[str, Dict[str, Optional[float]]],
    datasets: List[str],
    dims: Dict[str, int],
    output_path: Path,
):
    """Bar chart comparing methods per dataset for replacement scenario."""
    if not HAS_PLOT:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(datasets))
    width = 0.2
    colors = ["#2196F3", "#64B5F6", "#FF9800", "#4CAF50"]

    for i, method in enumerate(METHOD_ORDER):
        vals = []
        for ds in datasets:
            v = table[method].get(ds)
            vals.append(v if v is not None else 0)
        ax.bar(x + i * width, vals, width, label=METHOD_LABELS[method], color=colors[i])

    ax.set_ylabel("% of Baseline")
    ax.set_title("Replacement Scenario: Train on Synthetic, Test on Real")
    ax.set_xticks(x + width * 1.5)
    labels = [f"{DATASET_LABELS.get(ds, ds)}\n({dims.get(ds, '?')}d)" for ds in datasets]
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(y=100, color="red", linestyle="--", alpha=0.5, label="Baseline (100%)")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(0, 115)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_scaling_curve(
    table: Dict[str, Dict[str, Optional[float]]],
    datasets: List[str],
    dims: Dict[str, int],
    output_path: Path,
):
    """Line chart: utility vs dimensionality per method."""
    if not HAS_PLOT:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"our_tabddpm": "#2196F3", "vanilla_tabddpm": "#64B5F6", "ctgan": "#FF9800", "smogn": "#4CAF50"}
    markers = {"our_tabddpm": "o", "vanilla_tabddpm": "s", "ctgan": "^", "smogn": "D"}

    for method in METHOD_ORDER:
        xs, ys = [], []
        for ds in datasets:
            d = dims.get(ds)
            v = table[method].get(ds)
            if d is not None and v is not None:
                xs.append(d)
                ys.append(v)
        if xs:
            ax.plot(xs, ys, marker=markers[method], label=METHOD_LABELS[method],
                    color=colors[method], linewidth=2, markersize=8)

    ax.set_xlabel("Total Dimensions (numerical + one-hot categorical)")
    ax.set_ylabel("Replacement Utility (% of Baseline)")
    ax.set_title("Scaling: How Methods Perform as Dimensionality Increases")
    ax.axhline(y=100, color="red", linestyle="--", alpha=0.3)
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_ablation_chart(
    table: Dict[str, Dict[str, Optional[float]]],
    datasets: List[str],
    dims: Dict[str, int],
    output_path: Path,
):
    """Bar chart showing improvement of Our vs Vanilla TabDDPM."""
    if not HAS_PLOT:
        return

    deltas = []
    labels = []
    for ds in datasets:
        our = table["our_tabddpm"].get(ds)
        van = table["vanilla_tabddpm"].get(ds)
        if our is not None and van is not None:
            deltas.append(our - van)
            labels.append(f"{DATASET_LABELS.get(ds, ds)}\n({dims.get(ds, '?')}d)")

    if not deltas:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2196F3" if d >= 0 else "#F44336" for d in deltas]
    ax.bar(range(len(deltas)), deltas, color=colors)
    ax.set_xticks(range(len(deltas)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Improvement (percentage points)")
    ax.set_title("Our Improvements vs Vanilla TabDDPM (Replacement Scenario)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    avg = np.mean(deltas)
    ax.axhline(y=avg, color="blue", linestyle="--", alpha=0.5, label=f"Average: {avg:+.1f}pp")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_heatmap(
    table: Dict[str, Dict[str, Optional[float]]],
    datasets: List[str],
    title: str,
    output_path: Path,
    vmin: float = 0,
    vmax: float = 110,
):
    """Heatmap of method x dataset matrix."""
    if not HAS_PLOT:
        return

    data = []
    for method in METHOD_ORDER:
        row = []
        for ds in datasets:
            v = table[method].get(ds)
            row.append(v if v is not None else np.nan)
        data.append(row)

    data = np.array(data)
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        data, annot=True, fmt=".1f", cmap="RdYlGn",
        xticklabels=[DATASET_LABELS.get(ds, ds) for ds in datasets],
        yticklabels=[METHOD_LABELS[m] for m in METHOD_ORDER],
        vmin=vmin, vmax=vmax, ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# Summary Statistics
# =============================================================================

def compute_win_counts(
    table: Dict[str, Dict[str, Optional[float]]],
    datasets: List[str],
) -> Dict[str, int]:
    """Count how many datasets each method wins on."""
    wins = {m: 0 for m in METHOD_ORDER}
    for ds in datasets:
        best_method = None
        best_val = -999
        for method in METHOD_ORDER:
            val = table[method].get(ds)
            if val is not None and val > best_val:
                best_val = val
                best_method = method
        if best_method:
            wins[best_method] += 1
    return wins


# =============================================================================
# Generate Full Report
# =============================================================================

def generate_report(output_path: Optional[Path] = None):
    """Generate the full comparison report."""
    print("Loading results...")
    results = load_all_results()
    print(f"  Found {len(results)} experiment results")

    datasets = get_available_datasets(results)
    dims = get_dataset_dims(results, datasets)
    print(f"  Datasets: {datasets}")
    print(f"  Dimensions: {dims}")

    # Build tables
    print("\nBuilding tables...")
    repl_table = build_utility_table(results, datasets, "replacement")
    aug_table = build_utility_table(results, datasets, "augmentation")
    fidelity_table = build_fidelity_table(results, datasets, "avg_wasserstein")
    privacy_table = build_privacy_table(results, datasets)

    # Generate figures
    fig_dir = PHASE2_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    print("\nGenerating figures...")
    plot_replacement_comparison(repl_table, datasets, dims, fig_dir / "replacement_comparison.png")
    plot_scaling_curve(repl_table, datasets, dims, fig_dir / "scaling_curve.png")
    plot_ablation_chart(repl_table, datasets, dims, fig_dir / "ablation.png")
    plot_heatmap(repl_table, datasets, "Replacement Utility (% of Baseline)", fig_dir / "heatmap_replacement.png")
    plot_heatmap(aug_table, datasets, "Augmentation Utility (% of Baseline)", fig_dir / "heatmap_augmentation.png", vmin=80, vmax=105)

    # Compute statistics
    repl_wins = compute_win_counts(repl_table, datasets)
    aug_wins = compute_win_counts(aug_table, datasets)

    # Average per method (excluding None)
    avg_repl = {}
    avg_aug = {}
    for method in METHOD_ORDER:
        repl_vals = [v for v in repl_table[method].values() if v is not None]
        aug_vals = [v for v in aug_table[method].values() if v is not None]
        avg_repl[method] = np.mean(repl_vals) if repl_vals else 0
        avg_aug[method] = np.mean(aug_vals) if aug_vals else 0

    # Build report
    if output_path is None:
        output_path = PHASE2_DIR / "COMPARISON.md"

    print(f"\nWriting report to {output_path}...")

    with open(output_path, "w") as f:
        f.write("# Phase 2 Experiment Comparison Report\n\n")
        f.write(f"**Experiments:** {len(results)} runs ({len(METHOD_ORDER)} methods × {len(datasets)} datasets)\n")
        f.write(f"**Methods:** {', '.join(METHOD_LABELS[m] for m in METHOD_ORDER)}\n\n")

        # Dataset overview
        f.write("## Dataset Overview\n\n")
        f.write("| # | Dataset | Samples | Num | Cat | Total Dims | Task |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for ds in datasets:
            info = get_dataset_info_row(results, ds)
            if info:
                f.write(f"| {info.get('id', '?')} | {info.get('name', ds)} | "
                        f"{info.get('n_train', '?')}+{info.get('n_test', '?')} | "
                        f"{info.get('d_numerical', '?')} | {info.get('d_categorical', '?')} | "
                        f"{info.get('total_dims', '?')} | {info.get('task', '?')} |\n")
        f.write("\n---\n\n")

        # Key findings
        f.write("## Key Findings\n\n")
        f.write(f"**Replacement Scenario Wins:** ")
        for m in METHOD_ORDER:
            f.write(f"{METHOD_LABELS[m]}: {repl_wins[m]}, ")
        f.write("\n\n")
        f.write(f"**Average Replacement Utility:**\n")
        for m in METHOD_ORDER:
            f.write(f"- {METHOD_LABELS[m]}: **{avg_repl[m]:.1f}%**\n")
        f.write(f"\n**Average Augmentation Utility:**\n")
        for m in METHOD_ORDER:
            f.write(f"- {METHOD_LABELS[m]}: **{avg_aug[m]:.1f}%**\n")
        f.write("\n---\n\n")

        # Replacement table
        f.write("## Utility — Replacement Scenario\n")
        f.write("*Train on synthetic data only, test on real data. Higher % = better.*\n\n")
        f.write(format_utility_md(repl_table, datasets, dims, "Replacement (% of Baseline)"))
        f.write("\n\n---\n\n")

        # Augmentation table
        f.write("## Utility — Augmentation Scenario\n")
        f.write("*Train on real + synthetic data, test on real data. Higher % = better.*\n\n")
        f.write(format_utility_md(aug_table, datasets, dims, "Augmentation (% of Baseline)"))
        f.write("\n\n---\n\n")

        # Ablation
        f.write("## Ablation: Our Improvements vs Vanilla TabDDPM\n")
        f.write("*Shows the impact of MinMaxScaler, outlier clipping, and capacity scaling.*\n\n")
        f.write(format_ablation_md(repl_table, datasets, dims))
        f.write("\n\n---\n\n")

        # Fidelity
        f.write("## Statistical Fidelity (Avg Wasserstein Distance)\n")
        f.write("*Lower = better distribution match.*\n\n")
        f.write("| Method |")
        for ds in datasets:
            f.write(f" {DATASET_LABELS.get(ds, ds)} |")
        f.write("\n| --- |")
        for _ in datasets:
            f.write(" --- |")
        f.write("\n")
        for method in METHOD_ORDER:
            f.write(f"| **{METHOD_LABELS[method]}** |")
            for ds in datasets:
                val = fidelity_table[method].get(ds)
                if val is not None:
                    f.write(f" {val:.4f} |")
                else:
                    f.write(" — |")
            f.write("\n")
        f.write("\n---\n\n")

        # Privacy table
        f.write("## Privacy (Membership Inference Attack AUC)\n")
        f.write("*AUC ~0.50 = safe (random guessing). AUC > 0.60 = privacy concern. AUC > 0.80 = critical leak (data is just copies).*\n\n")
        f.write("| Method |")
        for ds in datasets:
            f.write(f" {DATASET_LABELS.get(ds, ds)} |")
        f.write(" **Avg** |\n| --- |")
        for _ in datasets:
            f.write(" --- |")
        f.write(" --- |\n")
        for method in METHOD_ORDER:
            f.write(f"| **{METHOD_LABELS[method]}** |")
            vals = []
            for ds in datasets:
                val = privacy_table[method].get(ds)
                if val is not None:
                    # Color-code: bold if unsafe
                    if val > 0.60:
                        f.write(f" **{val:.4f}** |")
                    else:
                        f.write(f" {val:.4f} |")
                    vals.append(val)
                else:
                    f.write(" — |")
            avg = np.mean(vals) if vals else 0
            f.write(f" **{avg:.4f}** |\n")
        f.write("\n---\n\n")

        # Figures
        if HAS_PLOT:
            f.write("## Figures\n\n")
            f.write("![Replacement Comparison](figures/replacement_comparison.png)\n\n")
            f.write("![Scaling Curve](figures/scaling_curve.png)\n\n")
            f.write("![Ablation](figures/ablation.png)\n\n")
            f.write("![Heatmap Replacement](figures/heatmap_replacement.png)\n\n")
            f.write("![Heatmap Augmentation](figures/heatmap_augmentation.png)\n\n")

    print(f"\nReport generated: {output_path}")
    return output_path


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Phase 2 comparison report")
    parser.add_argument("--output", type=str, default=None, help="Output path for report")
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    generate_report(output)
