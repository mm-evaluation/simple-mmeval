export PYTHONPATH=./:$PYTHONPATH

# Qwen3-VL Text-Only Tests
python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-2B-Instruct-text \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 8

python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-4B-Instruct-text \
    --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 8

python mmeval/run.py \
    --infile test_bed/modality_test/task/no_media.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-8B-Instruct-text \
    --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 8

