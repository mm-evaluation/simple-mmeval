export PYTHONPATH=./:$PYTHONPATH
#
# The committed outputs under work_dirs/examples/hf_dataset/ are exactly what
# this script produces: a deterministic 20-sample subset of the MMBench dev
# split (dev carries answers; the test split withholds them), selected with
# the framework's native sampling flags — see docs/en/USAGE.md "Run a subset
# of samples".

python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset mmeval_hf@mm-eval/MMBench-V11 \
    --subset en \
    --split dev \
    --sample_num 20 \
    --sample_order random \
    --sample_seed 42 \
    --out_dir work_dirs/examples/hf_dataset/MMBench_en_V11 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# Score the run. The explicit rule pipeline keeps this credential-free
# (MMBench's declared pipeline includes an LLM judge stage, which needs API
# keys); the override is recorded in score.json's knob_sources.
python mmeval/score.py \
    --score_out_dir work_dirs/examples/hf_dataset/MMBench_en_V11 \
    --score_result_glob 'result.json' \
    --score_pipeline exact-match,rule-match \
    --no_score_resume

# Fixture gate: every sample must be gradable — a fixture whose rows are all
# invalid would exercise nothing but the missing-gt path.
python3 - <<'PY'
import json
s = json.load(open("work_dirs/examples/hf_dataset/MMBench_en_V11/score.json"))["summary"]
assert s["invalid"] == 0, f"fixture has {s['invalid']} invalid samples"
print(f"fixture OK: {s['total']} samples, invalid=0, accuracy={s['accuracy']:.2f}")
PY
