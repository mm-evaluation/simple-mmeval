#!/bin/bash

export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/multi-image-video-interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-Omni-30B-A3B-Instruct-multi-image-video-interleave \
    --img_dir test_bed \
    --model_name_or_path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python mmeval/run.py \
    --infile test_bed/multi-image-video-interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Qwen3-Omni-30B-A3B-Thinking-multi-image-video-interleave \
    --img_dir test_bed \
    --model_name_or_path Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --gpu_per_parallel 1 \
    --parallel_per_task 1
