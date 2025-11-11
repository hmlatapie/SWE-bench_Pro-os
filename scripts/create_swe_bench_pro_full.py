#!/usr/bin/env python3
import click
import pandas as pd
from datasets import load_dataset

@click.command()
@click.option(
    "--split",
    default="test",
    show_default=True,
    help="Dataset split to download from HuggingFace (default: test)",
)
@click.option(
    "--output",
    default="swe_bench_pro_full.csv",
    show_default=True,
    help="Output CSV file path",
)
@click.option(
    "--num-tests",
    type=int,
    default=None,
    help="Limit the number of rows to include (e.g., 10 for quick testing)",
)
@click.option(
    "--instance-id",
    default=None,
    help="Filter to a single instance_id (takes precedence over --num-tests)",
)
def main(split: str, output: str, num_tests: int, instance_id: str):
    """
    Generate a CSV compatible with sweap_pro_eval_modal.py from the
    HuggingFace dataset ScaleAI/SWE-bench_Pro.

    Examples:
        python create_swe_bench_pro_csv.py --num-tests 10
        python create_swe_bench_pro_csv.py --instance-id instance_django__django-12345
    """

    click.echo(f"🔍 Loading SWE-bench_Pro dataset split: {split}")
    dataset = load_dataset("ScaleAI/SWE-bench_Pro", split=split)

    click.echo("🧩 Converting dataset to DataFrame...")
    df = pd.DataFrame(dataset)

    expected_cols = [
        "instance_id",
        "before_repo_set_cmd",
        "selected_test_files_to_run",
        "base_commit",
        "base_dockerfile",
        "instance_dockerfile",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
    ]

    # Try to normalize column names
    rename_map = {}
    for col in df.columns:
        lower = col.lower()
        if "instance_id" in lower:
            rename_map[col] = "instance_id"
        elif "before_repo_set_cmd" in lower or "repo_cmd" in lower:
            rename_map[col] = "before_repo_set_cmd"
        elif "selected_test_files_to_run" in lower or "test_files" in lower:
            rename_map[col] = "selected_test_files_to_run"
        elif "base_commit" in lower:
            rename_map[col] = "base_commit"
        elif "fail_to_pass" in lower:
            rename_map[col] = "FAIL_TO_PASS"
        elif "pass_to_pass" in lower:
            rename_map[col] = "PASS_TO_PASS"

    df = df.rename(columns=rename_map)

    # Keep only columns that exist in expected set
    df_filtered = df[[c for c in expected_cols if c in df.columns]].copy()

    # Fill missing expected columns with empty strings
    for c in expected_cols:
        if c not in df_filtered.columns:
            df_filtered[c] = ""

    # Apply filters
    if instance_id:
        click.echo(f"🎯 Filtering to single instance_id: {instance_id}")
        df_filtered = df_filtered[df_filtered["instance_id"] == instance_id]
        if df_filtered.empty:
            click.echo(f"❌ No matching instance found for: {instance_id}")
            return
    elif num_tests is not None:
        click.echo(f"📏 Limiting to first {num_tests} tests")
        df_filtered = df_filtered.head(num_tests)

    click.echo(f"📝 Writing {len(df_filtered)} rows to {output}")
    df_filtered.to_csv(output, index=False)
    click.echo("✅ Done! CSV ready for sweap_pro_eval_modal.py")

if __name__ == "__main__":
    main()

