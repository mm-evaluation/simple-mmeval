export PYTHONPATH=./:$PYTHONPATH
#
# The committed outputs under work_dirs/examples/hf_dataset/ are exactly what
# this script produces: a deterministic 20-sample subset selected with the
# framework's native sampling flags (--sample_num/--sample_order/--sample_seed,
# see docs/en/USAGE.md "Run a subset of samples"). Score the output
# credential-free with:
#   PYTHONPATH=. python3 mmeval/score.py --out_dir work_dirs/examples/hf_dataset/MMBench_en_V11 \
#       --matching_order exact,template --no_score_resume
# (the explicit rule chain avoids constructing the LLM matcher that MMBench's
# llm_extract protocol would otherwise require judge API credentials for)

python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset mmeval_hf@mm-eval/MMBench-V11 \
    --subset en \
    --split test \
    --sample_num 20 \
    --sample_order random \
    --sample_seed 42 \
    --out_dir work_dirs/examples/hf_dataset/MMBench_en_V11 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1
