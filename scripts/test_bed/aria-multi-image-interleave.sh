export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/aria-multi_image_interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path rhymes-ai/Aria \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512