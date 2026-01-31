export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/SmolVLM-Instruct-multi-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path HuggingFaceTB/SmolVLM-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 8 \
    --circular False \
    --resize 2048

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/SmolVLM-Instruct-DPO-multi-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path HuggingFaceTB/SmolVLM-Instruct-DPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 8 \
    --circular False \
    --resize 2048

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/SmolVLM-Synthetic-multi-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path HuggingFaceTB/SmolVLM-Synthetic \
    --gpu_per_parallel 1 \
    --parallel_per_task 8 \
    --circular False \
    --resize 2048