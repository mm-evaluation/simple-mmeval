export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile /data/ztw/data/cogcontrol4/cogcontrol4_frame/p0/cogcontrol4_total_p0.json \
    --dataset local@json \
    --out_dir work_dirs/llava-next-interleave-qwen-0.5b-test \
    --img_dir /data/ztw/data/cogcontrol4/cogcontrol4_media \
    --model_name_or_path lmms-lab/llava-next-interleave-qwen-0.5b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/multi_image_interleave.json \
#     --dataset local@json \
#     --out_dir work_dirs/llava-next-interleave-qwen-7b-multi-image-interleave \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path lmms-lab/llava-next-interleave-qwen-7b \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512

# python mmeval/run.py \
#     --infile test_bed/modality_test/task/multi_image_interleave.json \
#     --dataset local@json \
#     --out_dir work_dirs/llava-next-interleave-qwen-7b-dpo-multi-image-interleave \
#     --img_dir test_bed/modality_test/media/448 \
#     --model_name_or_path lmms-lab/llava-next-interleave-qwen-7b-dpo \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1 \
#     --max_new_tokens 512