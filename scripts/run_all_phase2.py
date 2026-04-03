"""Run all Phase 2 experiments. Designed to run on the desktop GPU."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from run_experiment import run_single_experiment, ALL_METHODS
from datasets import DATASET_REGISTRY
import time

DATASETS = list(DATASET_REGISTRY.keys())
METHODS = ALL_METHODS  # our_tabddpm, vanilla_tabddpm, ctgan, smogn

total = len(DATASETS) * len(METHODS)
completed = 0
failed = []
start = time.time()

print(f"Phase 2: {len(METHODS)} methods x {len(DATASETS)} datasets = {total} experiments")
print(f"Methods: {METHODS}")
print(f"Datasets: {DATASETS}")
print(f"{'='*60}\n")

for ds in DATASETS:
    for method in METHODS:
        completed += 1
        print(f"\n[{completed}/{total}] {method} on {ds}")
        try:
            result = run_single_experiment(ds, method, device="cuda")
            summary = result["utility"]["summary"]
            repl_pct = summary["replacement"]["pct_of_baseline"]
            aug_pct = summary["augmentation"]["pct_of_baseline"]
            print(f"  >> Replacement: {repl_pct:.1f}% | Augmentation: {aug_pct:.1f}%")
        except Exception as e:
            print(f"  >> FAILED: {e}")
            failed.append(f"{ds}_{method}")
            import traceback
            traceback.print_exc()

elapsed = time.time() - start
print(f"\n{'='*60}")
print(f"Phase 2 complete: {completed-len(failed)}/{total} succeeded in {elapsed/3600:.1f}h")
if failed:
    print(f"Failed: {failed}")
