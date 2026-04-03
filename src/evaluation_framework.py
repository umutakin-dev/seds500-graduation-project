"""
Standardized evaluation framework for Phase 2 experiments.

Evaluates synthetic data quality via downstream ML task performance:
- Replacement scenario: train on synthetic, test on real
- Augmentation scenario: train on real+synthetic, test on real
- 3 models per task type (RF, GB, Ridge/LogReg)
- Reports absolute scores and % of baseline

Usage:
    from evaluation_framework import evaluate_synthetic_data

    results = evaluate_synthetic_data(
        X_real_train, y_real_train,
        X_real_test, y_real_test,
        X_synthetic, y_synthetic,
        task_type="regression",
    )
"""

import json
import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, f1_score, mean_squared_error


def _get_models(task_type: str) -> Dict[str, Any]:
    """Return 3 models appropriate for the task type."""
    if task_type == "regression":
        return {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Ridge": Ridge(alpha=1.0),
        }
    else:
        return {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        }


def _score(model, X_test, y_test, task_type: str) -> Dict[str, float]:
    """Compute metrics for a fitted model."""
    y_pred = model.predict(X_test)
    if task_type == "regression":
        return {
            "r2": float(r2_score(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        }
    else:
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        }


def _primary_metric(task_type: str) -> str:
    """Return the primary metric name for comparison."""
    return "r2" if task_type == "regression" else "accuracy"


def evaluate_synthetic_data(
    X_real_train: np.ndarray,
    y_real_train: np.ndarray,
    X_real_test: np.ndarray,
    y_real_test: np.ndarray,
    X_synthetic: np.ndarray,
    y_synthetic: np.ndarray,
    task_type: str = "regression",
) -> Dict[str, Any]:
    """
    Evaluate synthetic data quality via downstream ML performance.

    Returns:
        Dict with keys:
        - baseline: {model_name: {metric: value}}
        - replacement: {model_name: {metric: value}}
        - augmentation: {model_name: {metric: value}}
        - summary: {scenario: {avg_primary_metric, pct_of_baseline}}
    """
    primary = _primary_metric(task_type)
    results = {"task_type": task_type, "primary_metric": primary}

    # --- Baseline: train on real, test on real ---
    baseline_scores = {}
    for name, model in _get_models(task_type).items():
        model.fit(X_real_train, y_real_train)
        baseline_scores[name] = _score(model, X_real_test, y_real_test, task_type)
    results["baseline"] = baseline_scores

    # --- Replacement: train on synthetic, test on real ---
    replacement_scores = {}
    for name, model in _get_models(task_type).items():
        model.fit(X_synthetic, y_synthetic)
        replacement_scores[name] = _score(model, X_real_test, y_real_test, task_type)
    results["replacement"] = replacement_scores

    # --- Augmentation: train on real+synthetic, test on real ---
    X_aug = np.vstack([X_real_train, X_synthetic])
    y_aug = np.concatenate([y_real_train, y_synthetic])
    augmentation_scores = {}
    for name, model in _get_models(task_type).items():
        model.fit(X_aug, y_aug)
        augmentation_scores[name] = _score(model, X_real_test, y_real_test, task_type)
    results["augmentation"] = augmentation_scores

    # --- Summary ---
    baseline_avg = np.mean([s[primary] for s in baseline_scores.values()])
    replacement_avg = np.mean([s[primary] for s in replacement_scores.values()])
    augmentation_avg = np.mean([s[primary] for s in augmentation_scores.values()])

    results["summary"] = {
        "baseline": {
            f"avg_{primary}": float(baseline_avg),
        },
        "replacement": {
            f"avg_{primary}": float(replacement_avg),
            "pct_of_baseline": float(replacement_avg / baseline_avg * 100) if baseline_avg != 0 else 0.0,
        },
        "augmentation": {
            f"avg_{primary}": float(augmentation_avg),
            "pct_of_baseline": float(augmentation_avg / baseline_avg * 100) if baseline_avg != 0 else 0.0,
        },
    }

    return results


def format_results_table(results: Dict[str, Any]) -> str:
    """Format evaluation results as a markdown table."""
    task = results["task_type"]
    primary = results["primary_metric"]
    lines = []

    if task == "regression":
        metrics = ["r2", "rmse"]
        headers = ["Model", "R²", "RMSE"]
    else:
        metrics = ["accuracy", "f1_macro"]
        headers = ["Model", "Accuracy", "F1 (macro)"]

    for scenario in ["baseline", "replacement", "augmentation"]:
        lines.append(f"\n### {scenario.title()}")
        lines.append(f"| {' | '.join(headers)} |")
        lines.append(f"| {' | '.join(['---'] * len(headers))} |")
        for model_name, scores in results[scenario].items():
            vals = [f"{scores[m]:.4f}" for m in metrics]
            lines.append(f"| {model_name} | {' | '.join(vals)} |")

    # Summary
    lines.append("\n### Summary")
    summary = results["summary"]
    lines.append(f"| Scenario | Avg {primary.upper()} | % of Baseline |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| Baseline | {summary['baseline'][f'avg_{primary}']:.4f} | 100.0% |")
    lines.append(f"| Replacement | {summary['replacement'][f'avg_{primary}']:.4f} | {summary['replacement']['pct_of_baseline']:.1f}% |")
    lines.append(f"| Augmentation | {summary['augmentation'][f'avg_{primary}']:.4f} | {summary['augmentation']['pct_of_baseline']:.1f}% |")

    return "\n".join(lines)
