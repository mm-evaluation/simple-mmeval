SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MMEVAL_DIR="$SCRIPT_DIR/../../../simple-mmeval-model-dev"
RESULT_DIR="$SCRIPT_DIR/../.."
export PYTHONPATH="$MMEVAL_DIR:$PYTHONPATH"
cd "$MMEVAL_DIR"

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/glm-4d1v/GLM-4.1V-9B-Thinking-multi-image-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path THUDM/GLM-4.1V-9B-Thinking \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 2048

python $MMEVAL_DIR/mmeval/run.py \
    --infile $MMEVAL_DIR/tests/samples/multi_image_interleave.json \
    --dataset local@json \
    --out_dir $RESULT_DIR/work_dirs/glm-4d1v/GLM-4.1V-9B-Base-multi-image-interleave \
    --img_dir $MMEVAL_DIR/tests/media/448 \
    --model_name_or_path THUDM/GLM-4.1V-9B-Base \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 2048

