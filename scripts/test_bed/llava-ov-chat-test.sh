export PYTHONPATH=./:$PYTHONPATH

# LLaVA OneVision Chat Test
python mmeval/run.py \
    --infile test_bed/modality_test/task/chat.json \
    --dataset local@json \
    --out_dir work_dirs/llava-onevision-qwen2-0.5b-ov-hf-chat-test \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/llava-onevision-qwen2-7b-ov-hf-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path llava-hf/llava-onevision-qwen2-7b-ov-hf \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/chat.json \
#     --dataset local@json \
#     --out_dir work_dirs/llava-onevision-qwen2-7b-ov-chat-hf-chat-test \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path llava-hf/llava-onevision-qwen2-7b-ov-chat-hf \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

