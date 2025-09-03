export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/single_video_start.json \
    --dataset local@json \
    --out_dir work_dirs/aria-single-video-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path rhymes-ai/Aria \
    --gpu_per_parallel 2 \
    --parallel_per_task 1