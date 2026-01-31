export PYTHONPATH=./:$PYTHONPATH

# InternVL3.5 Chat Test
python mmeval/run.py \
    --infile test_bed/modality_test/task/chat.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL3_5-1B-chat-test \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL3_5-1B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/InternVL3_5-2B-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/InternVL3_5-2B \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/InternVL3_5-4B-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/InternVL3_5-4B \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/InternVL3_5-8B-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/InternVL3_5-8B \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/InternVL3_5-14B-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/InternVL3_5-14B \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/InternVL3_5-38B-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path OpenGVLab/InternVL3_5-38B \
#     --gpu_per_parallel 2 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512
