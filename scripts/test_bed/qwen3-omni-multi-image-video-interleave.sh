export PYTHONPATH=./:$PYTHONPATH

# Qwen3-Omni Models
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-Omni-30B-A3B-Instruct-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --gpu_per_parallel 2 \
    --parallel_per_task 4 \
    --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/multi_image_video_interleave.json \
#     --dataset local@json \
#     --out_dir work_dirs/Qwen3-Omni-30B-A3B-Thinking-multi-image-video-interleave \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path Qwen/Qwen3-Omni-30B-A3B-Thinking \
#     --gpu_per_parallel 2 \
#     --parallel_per_task 4 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/multi_image_video_interleave.json \
#     --dataset local@json \
#     --out_dir work_dirs/Qwen3-Omni-30B-A3B-Captioner-multi-image-video-interleave \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path Qwen/Qwen3-Omni-30B-A3B-Captioner \
#     --gpu_per_parallel 2 \
#     --parallel_per_task 4 \
#     --max_new_tokens 512


