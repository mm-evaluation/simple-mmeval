export PYTHONPATH=./:$PYTHONPATH

# LLaVA OneVision 1.5 Chat Test
python mmeval/run.py \
    --infile test_bed/modality_test/task/chat.json \
    --dataset local@json \
    --out_dir work_dirs/LLaVA-OneVision-1.5-0.5B-Instruct-chat-test \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path lmms-lab/LLaVA-OneVision-1.5-0.5B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/LLaVA-OneVision-1.5-3B-Instruct-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path lmms-lab/LLaVA-OneVision-1.5-3B-Instruct \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/LLaVA-OneVision-1.5-8B-Instruct-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512
