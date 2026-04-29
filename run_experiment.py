"""
run_experiment.py

Entry point. Iterates over all (attacker x target x prompt) combinations,
runs each exchange, and saves results to JSONL + CSV.

Usage:
    python run_experiment.py [--dry-run] [--attacker-idx 0] [--target-idx 0]

The --attacker-idx / --target-idx flags let you run individual pairs in parallel
on the cluster (one job per pair).
"""

import json
import csv
import argparse
import os
from datetime import datetime
from pathlib import Path
from itertools import product

from config import (
    ATTACKER_MODELS, TARGET_MODELS, JUDGE_MODEL,
    JBB_BEHAVIORS_PATH, N_PROMPTS, MAX_TURNS, RESULTS_DIR
)
from attacker import Attacker
from defender import Defender
from judge import Judge
from experiment import run_exchange


def load_jbb_behaviors(path: str, n: int) -> list[dict]:
    """Load JBB behaviors from JSON file."""
    with open(path) as f:
        behaviors = json.load(f)
    return behaviors[:n]


def get_run_dir(run_id: str) -> Path:
    """Create and return the results directory for this run."""
    run_dir = Path(RESULTS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_result_jsonl(result_dict: dict, filepath: Path):
    """Append one result to a JSONL file (one JSON object per line)."""
    with open(filepath, "a") as f:
        f.write(json.dumps(result_dict) + "\n")


def save_summary_csv(results: list[dict], filepath: Path):
    """Write summary CSV with one row per (attacker, target, prompt) triple."""
    if not results:
        return

    # Flatten - exclude the verbose 'turns' list for the summary
    summary_keys = [k for k in results[0].keys() if k != "turns"]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_keys)
        writer.writeheader()
        for r in results:
            row = {k: r[k] for k in summary_keys}
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run only 1 prompt per pair to test the pipeline")
    parser.add_argument("--attacker-idx", type=int, default=None,
                        help="Run only this attacker index (for parallel cluster jobs)")
    parser.add_argument("--target-idx", type=int, default=None,
                        help="Run only this target index (for parallel cluster jobs)")
    parser.add_argument("--prompt-variant", type=int, default=1, choices=[1, 2],
                        help="Attacker system prompt variant (1 or 2)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print turn-by-turn progress")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = get_run_dir(run_id)
    raw_log_path = run_dir / "raw_logs.jsonl"
    summary_path = run_dir / "summary.csv"

    print(f"Run ID: {run_id}")
    print(f"Results dir: {run_dir}")

    # --- Load data ---
    n_prompts = 1 if args.dry_run else N_PROMPTS
    behaviors = load_jbb_behaviors(JBB_BEHAVIORS_PATH, n_prompts)
    print(f"Loaded {len(behaviors)} JBB behaviors")

    # --- Determine which attacker/target pairs to run ---
    attacker_list = ATTACKER_MODELS
    target_list = TARGET_MODELS

    if args.attacker_idx is not None:
        attacker_list = [ATTACKER_MODELS[args.attacker_idx]]
    if args.target_idx is not None:
        target_list = [TARGET_MODELS[args.target_idx]]

    # --- Initialise judge (shared across all runs) ---
    judge_model_id, judge_name, _ = JUDGE_MODEL
    judge = Judge(judge_model_id, judge_name)
    print(f"Judge: {judge_name}")

    all_results = []
    total = len(attacker_list) * len(target_list) * len(behaviors)
    completed = 0

    # --- Main loop ---
    for (a_id, a_name, a_size), (t_id, t_name, t_size) in product(attacker_list, target_list):

        print(f"\n{'='*60}")
        print(f"Attacker: {a_name} ({a_size}B) | Target: {t_name} ({t_size}B)")
        print(f"Size ratio: {a_size/t_size:.3f}x | log10: {__import__('math').log10(a_size/t_size):.3f}")
        print(f"{'='*60}")

        attacker = Attacker(a_id, a_name, a_size, prompt_variant=args.prompt_variant)
        defender = Defender(t_id, t_name, t_size)

        for task in behaviors:
            print(f"  [{completed+1}/{total}] Task {task['id']}: {task['behavior'][:60]}...")

            result = run_exchange(
                attacker=attacker,
                defender=defender,
                judge=judge,
                task=task,
                max_turns=MAX_TURNS,
                verbose=args.verbose,
            )

            result_dict = result.to_dict()
            all_results.append(result_dict)

            # Save incrementally (don't lose results if job is killed)
            save_result_jsonl(result_dict, raw_log_path)

            status = "SUCCESS" if result.attack_success else (
                "ATTACKER_REFUSED" if result.attacker_refused else "FAILED"
            )
            print(f"    → Harm: {result.harm_score:.1f} | {status} | {result.duration_seconds:.1f}s")
            completed += 1

    # --- Save summary CSV ---
    save_summary_csv(all_results, summary_path)
    print(f"\nDone. {completed} exchanges completed.")
    print(f"Raw logs: {raw_log_path}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()