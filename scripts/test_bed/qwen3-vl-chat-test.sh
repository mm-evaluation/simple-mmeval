export PYTHONPATH=./:$PYTHONPATH

# Qwen3-VL Chat Test
python mmeval/run.py \
    --infile test_bed/modality_test/task/chat.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-VL-2B-Instruct-chat-test \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/Qwen3-VL-4B-Instruct-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/Qwen3-VL-8B-Instruct-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/Qwen3-VL-32B-Instruct-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path Qwen/Qwen3-VL-32B-Instruct \
#     --gpu_per_parallel 2 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512
