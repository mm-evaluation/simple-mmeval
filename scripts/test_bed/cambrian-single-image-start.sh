export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/cambrian-8b-single-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path nyu-visionx/cambrian-8b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python mmeval/run.py \
    --infile test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/cambrian-13b-single-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path nyu-visionx/cambrian-13b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python mmeval/run.py \
    --infile test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/cambrian-34b-single-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path nyu-visionx/cambrian-34b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python mmeval/run.py \
    --infile test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/cambrian-phi3-3b-single-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path nyu-visionx/cambrian-phi3-3b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1