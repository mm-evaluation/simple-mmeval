export PYTHONPATH=./:$PYTHONPATH

# InternVL3 Chat Test
python mmeval/run.py \
    --infile test_bed/modality_test/task/chat.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL3-1B-Instruct-chat-test \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL3-1B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/InternVL3-2B-Instruct-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/InternVL3-2B-Instruct \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/InternVL3-8B-Instruct-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/InternVL3-8B-Instruct \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/InternVL3-14B-Instruct-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/InternVL3-14B-Instruct \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/InternVL3-38B-Instruct-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/InternVL3-38B-Instruct \
#     --gpu_per_parallel 2 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/InternVL3-78B-Instruct-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/InternVL3-78B-Instruct \
#     --gpu_per_parallel 4 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512
