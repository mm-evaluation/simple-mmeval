export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset mmeval_hf@mm-eval/MMMU-pro \
    --split standard_4_options \
    --out_dir work_dirs/Qwen2.5-VL-3B-Instruct/MMMU-pro \
    --gpu_per_parallel 1 \
    --no_conda \
    --parallel_per_task 8



python mmeval/run.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset mmeval_hf@mm-eval/MMMU \
    --split validation \
    --out_dir work_dirs/Qwen2.5-VL-3B-Instruct/MMMU \
    --gpu_per_parallel 1 \
    --no_conda \
    --parallel_per_task 8

python mmeval/run.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset mmeval_hf@mm-eval/DynaMath \
    --split test \
    --out_dir work_dirs/Qwen2.5-VL-3B-Instruct/DynaMath \
    --gpu_per_parallel 1 \
    --no_conda \
    --parallel_per_task 8

# python mmeval/run.py \
#     --model_name_or_path /home/tiger/yijiangli/project/opsd/OPSD/work_dirs/opsd_vlm_swa/qwen25vl3b_1epochs_lr1e5_bs4_fixteach_aug_unsup_open-mm-reasoner-74k@10k/Qwen2.5-VL-3B-Instruct-merged \
#     --dataset mmeval_hf@mm-eval/MMMU_Pro \
#     --split test \
#     --out_dir work_dirs/opsd_vlm_swa/qwen25vl3b_1epochs_lr1e5_bs4_fixteach_aug_unsup_open-mm-reasoner-74k@10k \
#     --gpu_per_parallel 1 \
#     --no_conda \
#     --parallel_per_task 8