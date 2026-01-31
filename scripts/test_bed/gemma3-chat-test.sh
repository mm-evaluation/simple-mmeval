export PYTHONPATH=./:$PYTHONPATH

# Gemma3 Chat Test
python mmeval/run.py \
    --infile test_bed/modality_test/task/chat.json \
    --dataset local@json \
    --out_dir work_dirs/gemma-3-4b-it-chat-test \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path google/gemma-3-4b-it \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/gemma-3-12b-it-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path google/gemma-3-12b-it \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/gemma-3-27b-it-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path google/gemma-3-27b-it \
#     --gpu_per_parallel 4 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

