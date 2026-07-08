export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset mmeval_hf@mm-eval/MMBench-en-V11 \
    --split test \
    --out_dir work_dirs/examples/hf_dataset/MMBench_en_V11 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1