SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/ovis2/Ovis2-1B-multi-image-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path AIDC-AI/Ovis2-1B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 2048

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/ovis2/Ovis2-2B-multi-image-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path AIDC-AI/Ovis2-2B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 2048

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/ovis2/Ovis2-4B-multi-image-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path AIDC-AI/Ovis2-4B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 2048

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/ovis2/Ovis2-8B-multi-image-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path AIDC-AI/Ovis2-8B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 2048

