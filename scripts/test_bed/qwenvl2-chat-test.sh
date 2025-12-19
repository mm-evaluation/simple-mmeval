export PYTHONPATH=./:$PYTHONPATH

# Qwen2-VL Chat Test
python mmeval/run.py \
    --infile test_bed/modality_test/task/chat.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen2-VL-2B-Instruct-chat-test \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/Qwen2-VL-7B-Instruct-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/Qwen2-VL-72B-Instruct-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path Qwen/Qwen2-VL-72B-Instruct \
#     --gpu_per_parallel 4 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512
