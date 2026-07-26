export PYTHONPATH=./:$PYTHONPATH
export DATASET_DIR=./datasets

# Inference examples (produce result.json) for several VLMEvalKit datasets.
# LLaVABench is additionally SCORED here with an open-weight local judge: its
# official protocol is the `llm-judge` path, and the `local` provider runs it
# with no API key (see run_score_evalkit_llm.sh / llm_judge_api_reference.md for
# the API judge). The reference answer is in the `gpt4_ans` field, so scoring
# passes `--score_gt_field gpt4_ans`.
#
# IMPORTANT — this is a DEGRADED approximation, not LLaVABench's official number:
# the official metric is a GPT-graded RELATIVE score (assistant vs reference on a
# 1-10 scale, per category, with the image caption given to the judge), whereas
# our llm-judge emits a BINARY correctness verdict and the caption is not fed to
# the judge. Treat the committed accuracy as a per-sample binary rate under an
# open judge, not as the official LLaVABench score. See docs/en/OPEN_JUDGE.md.
# The other datasets below are inference-only.

# LLaVABench — inference
python mmeval/run.py \
    --dataset evalkit@LLaVABench \
    --out_dir work_dirs/examples/evalkit_dataset/LLaVABench \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# LLaVABench — score with a local open-weight judge (deterministic, greedy)
python mmeval/score.py \
    --score_out_dir work_dirs/examples/evalkit_dataset/LLaVABench \
    --score_pipeline llm-judge \
    --score_gt_field gpt4_ans \
    --judge_provider local --judge_model Qwen/Qwen2.5-7B-Instruct

# Fixture gate: a committed judge fixture must be a genuine graded run.
python3 - <<'PY'
import json
s = json.load(open("work_dirs/examples/evalkit_dataset/LLaVABench/score.json"))["summary"]
assert s["invalid"] == 0, f"fixture has {s['invalid']} invalid samples"
assert s["llm_errors"] == 0, f"fixture has {s['llm_errors']} llm_errors"
print(f"LLaVABench fixture OK: {s['total']} samples, invalid=0, "
      f"llm_errors=0, accuracy={s['accuracy']:.2f}")
PY

# MicroBench
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset evalkit@MicroBench \
    --out_dir work_dirs/examples/evalkit_dataset/MicroBench \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# MMMB
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset evalkit@MMMB \
    --out_dir work_dirs/examples/evalkit_dataset/MMMB \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# 3DSRBench (URL)
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset https://huggingface.co/datasets/mm-eval/VLMEvalKit/resolve/main/3DSRBench.tsv \
    --out_dir work_dirs/examples/evalkit_dataset/3DSRBench_url \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# MM-Math (Local Path)
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset datasets/MM-Math.tsv \
    --out_dir work_dirs/examples/evalkit_dataset/MM_Math_local \
    --gpu_per_parallel 1 \
    --parallel_per_task 1
