export PYTHONPATH=./:$PYTHONPATH
#
# NOTE on committed example outputs: the result.json/score.json checked into
# work_dirs/examples/ are the outputs of these commands as written — the HF
# example (Test 4) selects a deterministic 20-sample subset of the answered
# dev split with the framework's native sampling flags
# (--sample_num/--sample_order/--sample_seed, see docs/en/USAGE.md "Run a
# subset of samples"), scores it, and asserts every sample is gradable.
# Score the local-JSON outputs (Tests 1-2) with:
#   PYTHONPATH=. python3 mmeval/score.py --score_out_dir <out_dir> --no_score_resume

# Test 1: Local JSON with template file path
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset local@json \
    --infile tests/samples/template.json \
    --img_dir tests/media/448 \
    --out_dir work_dirs/examples/prompt_template/local_template_file_path \
    --template mmeval/data/default_template.txt \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# Test 2: Local JSON with inline template string
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset local@json \
    --infile tests/samples/template.json \
    --img_dir tests/media/448 \
    --out_dir work_dirs/examples/prompt_template/local_template_string \
    --template '{{ question }}{% if options %}
Choices:
{% for k, v in options.items() %}({{ k }}) {{ v }}{% if not loop.last %}
{% endif %}{% endfor %}{% endif %}{% if hint %}
Note: {{ hint }}{% endif %}' \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# Test 3: TSV with template
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset evalkit@MMBench_dev_en \
    --out_dir work_dirs/examples/prompt_template/tsv_template \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# Test 4: HuggingFace MMBench-V11 (en subset), 20-sample subset via native
# sampling. The dev split carries answers (test withholds them), so the
# committed fixture exercises real grading.
python mmeval/run.py \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --dataset mmeval_hf@mm-eval/MMBench-V11 \
    --subset en \
    --split dev \
    --sample_num 20 \
    --sample_order random \
    --sample_seed 42 \
    --out_dir work_dirs/examples/prompt_template/hf_template \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# Score Test 4 and gate the fixture: every sample must be gradable.
python mmeval/score.py \
    --score_out_dir work_dirs/examples/prompt_template/hf_template \
    --score_result_glob 'result.json' \
    --score_pipeline exact-match,rule-match \
    --no_score_resume
python3 - <<'PY'
import json
s = json.load(open("work_dirs/examples/prompt_template/hf_template/score.json"))["summary"]
assert s["invalid"] == 0, f"fixture has {s['invalid']} invalid samples"
print(f"fixture OK: {s['total']} samples, invalid=0, accuracy={s['accuracy']:.2f}")
PY


