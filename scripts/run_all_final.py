"""Phase 2 final: all 11 datasets x 4 methods with privacy testing. Resumable."""
import sys, os, time, atexit
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from run_experiment import run_single_experiment, ALL_METHODS, is_experiment_complete
from datasets import DATASET_REGISTRY

LOCK_FILE = Path(__file__).parent.parent / "experiments" / "phase2" / ".running"

def cleanup():
    LOCK_FILE.unlink(missing_ok=True)

def main():
    # Lock file so we know it's running
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\nPID: {os.getpid()}\n")
    atexit.register(cleanup)

    DATASETS = list(DATASET_REGISTRY.keys())
    METHODS = ALL_METHODS

    # Find what's already done (with privacy)
    todo = []
    for ds in DATASETS:
        for method in METHODS:
            if not is_experiment_complete(ds, method):
                todo.append((ds, method))

    total_all = len(DATASETS) * len(METHODS)
    done_already = total_all - len(todo)

    print(f"Phase 2 final: {total_all} total, {done_already} already done, {len(todo)} remaining", flush=True)
    print(f"Includes: utility + fidelity + PRIVACY testing", flush=True)
    print(f"{'='*60}\n", flush=True)

    completed = 0
    failed = []
    start = time.time()

    for ds, method in todo:
        completed += 1
        print(f"\n[{completed}/{len(todo)}] {method} on {ds}", flush=True)
        try:
            result = run_single_experiment(ds, method, device="cuda")
            summary = result["utility"]["summary"]
            repl_pct = summary["replacement"]["pct_of_baseline"]
            aug_pct = summary["augmentation"]["pct_of_baseline"]
            priv = result.get("privacy", {})
            priv_auc = priv.get("attack_auc", "N/A")
            interp = priv.get("interpretation", "")
            if isinstance(priv_auc, float):
                print(f"  >> Repl: {repl_pct:.1f}% | Aug: {aug_pct:.1f}% | Privacy: {priv_auc:.4f} ({interp})", flush=True)
            else:
                print(f"  >> Repl: {repl_pct:.1f}% | Aug: {aug_pct:.1f}% | Privacy: {priv_auc}", flush=True)
        except Exception as e:
            print(f"  >> FAILED: {e}", flush=True)
            failed.append(f"{ds}_{method}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start
    print(f"\n{'='*60}", flush=True)
    print(f"Done: {completed-len(failed)}/{len(todo)} new + {done_already} previous = {total_all-len(failed)} total in {elapsed/3600:.1f}h", flush=True)
    if failed:
        print(f"Failed: {failed}", flush=True)

if __name__ == "__main__":
    main()
