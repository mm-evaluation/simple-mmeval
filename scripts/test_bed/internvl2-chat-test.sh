export PYTHONPATH=./:$PYTHONPATH

# InternVL2 Chat Test
python mmeval/run.py \
    --infile test_bed/modality_test/task/chat.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2-1B-chat-test \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2-1B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/InternVL2-2B-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/InternVL2-2B \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/InternVL2-4B-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/InternVL2-4B \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/InternVL2-8B-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/InternVL2-8B \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/InternVL2-26B-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/InternVL2-26B \
#     --gpu_per_parallel 2 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/InternVL2-40B-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/InternVL2-40B \
#     --gpu_per_parallel 2 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512
