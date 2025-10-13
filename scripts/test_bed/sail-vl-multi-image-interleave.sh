export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/SAIL-VL-2B-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path BytedanceDouyinContent/SAIL-VL-2B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/SAIL-VL-4B-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path BytedanceDouyinContent/SAIL-VL-4B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/SAIL-VL-8B-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path BytedanceDouyinContent/SAIL-VL-8B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/SAIL-VL-16B-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path BytedanceDouyinContent/SAIL-VL-16B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512
