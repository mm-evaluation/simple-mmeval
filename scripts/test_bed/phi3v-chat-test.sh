export PYTHONPATH=./:$PYTHONPATH

# Phi-3.5 Vision Chat Test
python mmeval/run.py \
    --infile test_bed/modality_test/task/chat.json \
    --dataset local@json \
    --out_dir work_dirs/Phi-3.5-vision-instruct-chat-test \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path microsoft/Phi-3.5-vision-instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/chat.json \
    --dataset local@json \
    --out_dir work_dirs/Phi-3-vision-128k-instruct-chat-test \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path microsoft/Phi-3-vision-128k-instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

