export PYTHONPATH=./:$PYTHONPATH

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /MLLM_Eval/envs/vllm

python mmeval/infer/qwen2d5.py \
    --infile test_bed/image.json \
    --dataset local@json \
    --sample_mode first \
    --sample_number 5 \
    --out_dir work_dirs/test_dataloader/test_qwen2d5_local \
    --img_dir test_bed \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct

python mmeval/infer/qwen2d5.py \
    --dataset MME \
    --sample_mode first \
    --sample_number 5 \
    --out_dir work_dirs/test_dataloader/test_qwen2d5_mme \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct

python mmeval/infer/qwen2d5.py \
    --dataset MM-Vet \
    --sample_mode first \
    --sample_number 5 \
    --out_dir work_dirs/test_dataloader/test_qwen2d5_mm_vet \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct

# python mmeval/infer/qwen2d5.py \
#     --dataset SEED-Bench \
#     --sample_mode first \
#     --sample_number 5 \
#     --out_dir work_dirs/test_dataloader/test_qwen2d5_seed_bench \
#     --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct
