#!/usr/bin/env python3
import click
import pandas as pd
from datasets import load_dataset

@click.command()
@click.option(
    "--split",
    default="test",
    help="Dataset split to download from HuggingFace (default: test)",
)
@click.option(
    "--output",
    default="swe_bench_pro_full.csv",
    show_default=True,
    help="Output CSV file path",
)
def main(split: str, output: str):
    """
    Generate swe_bench_pro_full.csv from the HuggingFace dataset ScaleAI/SWE-bench_Pro.
    """

    click.echo(f"🔍 Loading SWE-bench_Pro dataset split: {split}")
    dataset = load_dataset("ScaleAI/SWE-bench_Pro", split=split)

    click.echo("🧩 Converting dataset to DataFrame...")
    df = pd.DataFrame(dataset)

    # The evaluation script expects the following columns:
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

    # Try to map existing columns to the expected ones
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

    # Keep only relevant columns
    df_filtered = df[[c for c in expected_cols if c in df.columns]].copy()

    # Fill missing expected columns with empty strings
    for c in expected_cols:
        if c not in df_filtered.columns:
            df_filtered[c] = ""

    click.echo(f"📝 Writing {len(df_filtered)} rows to {output}")
    df_filtered.to_csv(output, index=False)
    click.echo("✅ Done! CSV ready for sweap_pro_eval_modal.py")

if __name__ == "__main__":
    main()

