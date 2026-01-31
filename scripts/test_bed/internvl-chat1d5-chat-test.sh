export PYTHONPATH=./:$PYTHONPATH

# InternVL-Chat V1.5 Chat Test
python mmeval/run.py \
    --infile test_bed/modality_test/task/chat.json \
    --dataset local@json \
    --out_dir work_dirs/Mini-InternVL-Chat-2B-V1-5-chat-test \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/Mini-InternVL-Chat-2B-V1-5 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/Mini-InternVL-Chat-4B-V1-5-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/Mini-InternVL-Chat-4B-V1-5 \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/InternVL-Chat-V1-5-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/InternVL-Chat-V1-5 \
#     --gpu_per_parallel 2 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

