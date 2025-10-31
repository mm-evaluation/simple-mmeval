export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/mPLUG-Owl3-7B-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path mPLUG/mPLUG-Owl3-7B-241101 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/mPLUG-Owl3-2B-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path mPLUG/mPLUG-Owl3-2B-241014 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1
