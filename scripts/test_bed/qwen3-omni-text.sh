export PYTHONPATH=./:$PYTHONPATH

# Qwen3-Omni Text-Only Tests
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-Omni-30B-A3B-Instruct-text \
    --model_name_or_path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --gpu_per_parallel 2 \
    --parallel_per_task 1

python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-Omni-30B-A3B-Thinking-text \
    --model_name_or_path Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --gpu_per_parallel 2 \
    --parallel_per_task 1

