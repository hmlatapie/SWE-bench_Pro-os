#!/usr/bin/env python3
import json
import random
from typing import List, Dict, Any, Optional

import click
from datasets import load_dataset

def _list_to_str(value) -> str:
    """
    Ensure fields the evaluator eval()s are serialized as a string list literal.
    Using JSON-style double quotes keeps it safe and eval()-compatible.
    """
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return "[]"
    # Already a string from source; trust but verify minimal structure
    s = str(value).strip()
    return s if (s.startswith("[") and s.endswith("]")) else "[]"

@click.command()
@click.option("--split", default="test", show_default=True, help="HF split to use")
@click.option("--output", default="swe_bench_pro_full.jsonl", show_default=True, help="Output JSONL path")
@click.option("--patch-output", default="gold_patches.json", show_default=True, help="Where to write gold patches JSON")
@click.option("--patch-prefix", default="gold", show_default=True, help="Prefix recorded in each patch record")
@click.option("--num-tests", type=int, default=None, help="Limit number of rows (random sample)")
@click.option("--instance-id", default=None, help="Filter to a single instance_id (overrides --num-tests)")
@click.option("--seed", type=int, default=42, show_default=True, help="Sampling seed when --num-tests is used")
def main(split: str, output: str, patch_output: str, patch_prefix: str,
         num_tests: Optional[int], instance_id: Optional[str], seed: int):
    """
    Generate:
      1) swe_bench_pro_full.jsonl for sweap_pro_eval_modal.py (--raw_sample_path)
      2) gold_patches.json for sweap_pro_eval_modal.py (--patch_path)

    JSONL keeps lists/newlines intact and matches what the evaluator expects.
    We REQUIRE the HF dataset to have a 'patch' field for gold diffs.
    """
    print(f"Loading ScaleAI/SWE-bench_Pro split={split} ...")
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split=split)
    rows: List[Dict[str, Any]] = [dict(r) for r in ds]

    # Filter/sampling
    if instance_id:
        rows = [r for r in rows if str(r.get("instance_id")) == instance_id]
        if not rows:
            raise SystemExit(f"No rows match instance_id={instance_id}")
    elif num_tests:
        rnd = random.Random(seed)
        rnd.shuffle(rows)
        rows = rows[:num_tests]

    if not rows:
        raise SystemExit("No rows to write after filtering.")

    # Hard-require 'patch' key to avoid guesswork
    if "patch" not in rows[0]:
        keys = sorted(rows[0].keys())
        raise SystemExit(
            "Expected 'patch' field not found in dataset rows. "
            f"Available keys include: {keys[:20]} ..."
        )

    # Build JSONL (raw samples) — ensure list-like fields are STRINGIFIED list literals (eval-friendly)
    out_samples: List[Dict[str, Any]] = []
    for r in rows:
        out_samples.append({
            "instance_id": r.get("instance_id", ""),
            "before_repo_set_cmd": r.get("before_repo_set_cmd", r.get("repo_cmd", "")),
            "selected_test_files_to_run": _list_to_str(r.get("selected_test_files_to_run", r.get("test_files", []))),
            "base_commit": r.get("base_commit", ""),
            # evaluator later reads lowercase keys:
            "fail_to_pass": _list_to_str(r.get("fail_to_pass", r.get("FAIL_TO_PASS", []))),
            "pass_to_pass": _list_to_str(r.get("pass_to_pass", r.get("PASS_TO_PASS", []))),
            "repo": r.get("repo", ""),
        })

    with open(output, "w", encoding="utf-8") as f:
        for row in out_samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out_samples)} rows to {output}")

    # Build patches JSON (gold)
    patches = [
        {
            "instance_id": str(r.get("instance_id", "")).strip(),
            "patch": str(r.get("patch", "")),
            "prefix": patch_prefix,
        }
        for r in rows
        if r.get("patch")
    ]
    if not patches:
        raise SystemExit("No gold patches found (rows had empty 'patch' values).")

    with open(patch_output, "w", encoding="utf-8") as f:
        json.dump(patches, f)
    print(f"Wrote {len(patches)} patches to {patch_output}")

if __name__ == "__main__":
    main()

