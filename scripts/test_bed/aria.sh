export PYTHONPATH=./:$PYTHONPATH

# Set HuggingFace cache to current project directory
export HF_HOME=/scratch/bbkc/boqiny2/simple-mmeval/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/bbkc/boqiny2/simple-mmeval/.cache/huggingface/datasets

python mmeval/run.py \
    --infile test_bed/image.json \
    --dataset local@json \
    --out_dir test_bed/test_aria \
    --img_dir test_bed \
    --model_name_or_path rhymes-ai/Aria \
    --gpu_per_parallel 2 \
    --parallel_per_task 1