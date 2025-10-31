export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/glm-41v-9b-thinking-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path zai-org/GLM-4.1V-9B-Thinking \
    --gpu_per_parallel 1 \
    --parallel_per_task 1
