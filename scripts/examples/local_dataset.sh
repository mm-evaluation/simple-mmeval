export PYTHONPATH=./:$PYTHONPATH

# Inference-only example. These inputs are modality / text smoke samples
# (tests/samples/*.json) that carry NO ground truth, so scoring them would be
# 100% invalid and verify nothing — no score.json fixture is committed and the
# `invalid == 0` fixture gate (see hf_dataset.sh) does not apply here. The
# committed result.json shows the inference output format for local@json.

python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset local@json \
    --infile tests/samples/multi_image_video_interleave.json \
    --img_dir tests/media/448 \
    --out_dir work_dirs/examples/local_dataset/modality_test \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset local@json \
    --infile tests/samples/no_media.json \
    --out_dir work_dirs/examples/local_dataset/text_test \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

