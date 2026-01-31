export PYTHONPATH=./:$PYTHONPATH

# InternVL-Chat V1 Chat Test
python mmeval/run.py \
    --infile test_bed/modality_test/task/chat.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL-Chat-V1-1-chat-test \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL-Chat-V1-1 \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/InternVL-Chat-V1-2-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/InternVL-Chat-V1-2 \
#     --gpu_per_parallel 2 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/InternVL-Chat-V1-2-Plus-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/InternVL-Chat-V1-2-Plus \
#     --gpu_per_parallel 2 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

