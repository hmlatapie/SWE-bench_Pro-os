./swe_bench_pro_eval.py \
    --raw_sample_path=swe_bench_pro_full.jsonl \
    --patch_path=gold_patches.json \
    --output_dir=/tmp/ \
    --scripts_dir=run_scripts \
    --num_workers=1 \
    --dockerhub_username=jefzda \
    --use_local_docker

