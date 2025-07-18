export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/image-qca.json \
    --dataset local@json \
    --out_dir test_bed/test_blip2opt_score \
    --img_dir test_bed \
    --model_name_or_path blip2-opt-2.7b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 