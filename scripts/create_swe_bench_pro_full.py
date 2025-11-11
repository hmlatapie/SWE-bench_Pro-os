#!/usr/bin/env python3
import json
import random
import click
from datasets import load_dataset

@click.command()
@click.option("--split", default="test", show_default=True, help="HF split to use")
@click.option("--output", default="swe_bench_pro_full.jsonl", show_default=True, help="Output JSONL path")
@click.option("--num-tests", type=int, default=None, help="Limit number of rows (random sample)")
@click.option("--instance-id", default=None, help="Filter to a single instance_id (overrides --num-tests)")
@click.option("--seed", type=int, default=42, show_default=True, help="Sampling seed when --num-tests is used")
def main(split, output, num_tests, instance_id, seed):
    """
    Generate a JSONL compatible with sweap_pro_eval_modal.py to avoid CSV quoting issues.
    Ensures keys match what the evaluator reads later: fail_to_pass / pass_to_pass (lowercase).
    """
    print(f"Loading ScaleAI/SWE-bench_Pro split={split} ...")
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split=split)

    # Convert to python list of dicts
    rows = [dict(r) for r in ds]

    # Optional filters
    if instance_id:
        rows = [r for r in rows if str(r.get("instance_id")) == instance_id]
        if not rows:
            print(f"No rows match instance_id={instance_id}")
            return
    elif num_tests:
        random.Random(seed).shuffle(rows)
        rows = rows[:num_tests]

    # Map / normalize keys the evaluator actually uses
    def norm(row):
        return {
            # required by the evaluator
            "instance_id": row.get("instance_id", ""),
            "before_repo_set_cmd": row.get("before_repo_set_cmd", row.get("repo_cmd", "")),
            "selected_test_files_to_run": row.get("selected_test_files_to_run", row.get("test_files", [])),
            "base_commit": row.get("base_commit", ""),

            # NOTE: evaluator later reads LOWERCASE versions:
            "fail_to_pass": row.get("fail_to_pass", row.get("FAIL_TO_PASS", [])),
            "pass_to_pass": row.get("pass_to_pass", row.get("PASS_TO_PASS", [])),

            # optional but helpful for image resolution
            "repo": row.get("repo", ""),
        }

    out_rows = [norm(r) for r in rows]

    # Write JSONL
    with open(output, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_rows)} rows to {output}")

if __name__ == "__main__":
    main()

