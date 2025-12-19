export PYTHONPATH=./:$PYTHONPATH

# Qwen3-Omni Chat Test
python mmeval/run.py \
    --infile test_bed/modality_test/task/chat.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-Omni-30B-A3B-Instruct-chat-test \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

