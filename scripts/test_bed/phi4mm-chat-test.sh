export PYTHONPATH=./:$PYTHONPATH

# Phi-4 Multimodal Chat Test
python mmeval/run.py \
    --infile test_bed/modality_test/task/chat.json \
    --dataset local@json \
    --out_dir work_dirs/Phi-4-multimodal-instruct-chat-test \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path microsoft/Phi-4-multimodal-instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

