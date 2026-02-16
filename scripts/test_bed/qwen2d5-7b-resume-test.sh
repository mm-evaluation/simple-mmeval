export PYTHONPATH=./:$PYTHONPATH

# Resume test with existing cache.db
# The cache.db contain 7 samples (eval-id 0-6)
python mmeval/run.py \
    --dataset local@json \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --img_dir test_bed/modality_test/media/512 \
    --out_dir work_dirs/Qwen2.5-VL-7B-Instruct-hf-resume-test \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --resume
