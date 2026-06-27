#!/usr/bin/env bash
#
# Evaluate Qwen3-VL-4B-Instruct on mm-eval/BLINK and mm-eval/MMStar.
#
#   - Datasets (mm-eval HuggingFace format, NOT VLMEvalKit):
#       * mm-eval/BLINK   -> all 14 sub-task configs, split=val
#       * mm-eval/MMStar  -> single default config,   split=val
#   - Generation: VLMEvalKit Qwen3-VL setup -> temperature=0.01, top_k=20,
#                 top_p=0.8, do_sample=true; max_new_tokens=4096 (override via env)
#   - Matching:   cascade exact -> template -> llm (judge only resolves leftovers)
#   - GPUs:       parallel_per_task=8 (8 cards), the two datasets run in parallel
#
# Usage:
#   OPENAI_API_KEY=sk-... bash scripts/examples/run_qwen3vl4b_blink_mmstar.sh
#
# Optional env overrides:
#   MODEL_NAME_OR_PATH   (default: Qwen/Qwen3-VL-4B-Instruct)
#   OUT_DIR              (default: work_dirs/Qwen3-VL-4B-Instruct_llmjudge)
#   PARALLEL_PER_TASK    (default: 8)      # GPU workers per dataset run
#   GPU_PER_PARALLEL     (default: 1)
#   JUDGE_PROVIDER       (default: openai) # openai | azure_openai
#   JUDGE_MODEL          (default: gpt-5)
#   SCORE_PARALLEL       (default: 8)      # sample workers for the judge
#   OPENAI_BASE_URL      (optional, for OpenAI-compatible endpoints)

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="./:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Reduce CUDA allocator fragmentation -> lower peak memory, more headroom for
# high oversubscription (e.g. BLINK at 6 shards/GPU with heavy multi-image KV).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
OUT_DIR="${OUT_DIR:-work_dirs/Qwen3-VL-4B-Instruct_llmjudge}"
GPU_PER_PARALLEL="${GPU_PER_PARALLEL:-1}"

# Physical GPUs to use (comma-separated). Default: all 8.
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

# Oversubscription: how many shards to run *per physical GPU* for each dataset.
# The scheduler builds its GPU pool from CUDA_VISIBLE_DEVICES entries (opaque
# strings), so repeating each GPU id N times launches N shards on that card.
# MMStar is the long pole -> 4 shards/GPU (32 total). BLINK is tiny -> 1/GPU.
# Peak model copies/GPU when both run together = MMStar + BLINK (~5), ~50GB on
# 80GB cards for this 4B model. Lower these if you hit OOM.
# At 16k context each model copy needs ~13GB worst-case (8.3GB weights + ~2.4GB
# KV@16k + overhead); safe ceiling ~6 copies/GPU. MMStar (big single job) gets
# 4/GPU; BLINK runs alongside at 2/GPU (its configs are small) -> 6/GPU total.
SHARDS_PER_GPU_MMSTAR="${SHARDS_PER_GPU_MMSTAR:-4}"
SHARDS_PER_GPU_BLINK="${SHARDS_PER_GPU_BLINK:-2}"

# Build a repeated CUDA_VISIBLE_DEVICES string and matching parallel_per_task.
# args: <n_repeat> -> echoes "cvd_string parallel_count"
build_pool() {
  local n_repeat="$1"
  local cvd=""
  local count=0
  local r g
  IFS=',' read -ra _gpus <<< "${GPUS}"
  for ((r = 0; r < n_repeat; r++)); do
    for g in "${_gpus[@]}"; do
      cvd+="${g},"
      count=$((count + 1))
    done
  done
  echo "${cvd%,} ${count}"
}

read -r MMSTAR_CVD MMSTAR_PARALLEL < <(build_pool "${SHARDS_PER_GPU_MMSTAR}")
read -r BLINK_CVD  BLINK_PARALLEL  < <(build_pool "${SHARDS_PER_GPU_BLINK}")

# Generation hyper-parameters
# Follows VLMEvalKit's Qwen3-VL wrapper defaults (temperature=0.01, top_k=20,
# top_p=0.8, do_sample=True). max_new_tokens=8192: best speed/correctness balance.
# At 4096 ~9.7% of MMStar responses truncated mid-reasoning; 8k lets the vast
# majority of long CoT finish while keeping runtime ~2x faster than 16k (16k
# measured ~2-5h for MMStar due to batch-1 HF generation on very long sequences).
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
# Defaults = VLMEvalKit Qwen3-VL setup; override via env for other models
# (Qwen2.5-VL: TEMPERATURE=0.01 TOP_K=1 TOP_P=0.001 MAX_NEW_TOKENS=2048).
TEMPERATURE="${TEMPERATURE:-0.01}"
TOP_K="${TOP_K:-20}"
TOP_P="${TOP_P:-0.8}"

# Prompt template (Jinja2). Aligns with VLMEvalKit's ImageMCQDataset by appending
# "Please select the correct answer from the options above." to the question.
# Set TEMPLATE="" to disable and fall back to the dataset/default template.
TEMPLATE="${TEMPLATE:-scripts/templates/vlmevalkit_mcq.txt}"

# Matching cascade: robust template (VLMEvalKit-style can_infer letter extraction)
# -> LLM for leftovers. `exact` dropped: the robust template subsumes it (template
# alone 0.6347 == exact+template 0.6360 on MMStar after the \boxed{} patch).
# LLM stage options: `llm-judge` (LLM judges correctness, our approach) or
# `llm-match` (LLM extracts the option letter then rule-based compare, VLMEvalKit).
MATCHING_ORDER="${MATCHING_ORDER:-template,llm-match}"

# LLM-judge config (Azure OpenAI, gpt-5.4-mini reasoning model)
JUDGE_PROVIDER="${JUDGE_PROVIDER:-azure_openai}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-mini-2026-03-17}"
SCORE_PARALLEL="${SCORE_PARALLEL:-8}"
SCORE_OUTPUT_NAME="score_llm.json"

# Cap concurrent LLM judge/match calls (decoupled from SCORE_PARALLEL) and retry
# with backoff — avoids API quota saturation that silently degrades verdicts.
export JUDGE_MAX_CONCURRENCY="${JUDGE_MAX_CONCURRENCY:-4}"
export JUDGE_MAX_RETRY="${JUDGE_MAX_RETRY:-3}"

# --- Azure OpenAI judge credentials (consumed by mmeval/scoring/match/llm.py) ---
export AZURE_OPENAI_KEY="${AZURE_OPENAI_KEY:-kEQGmoirhe9XZdDOCmE5MxAL3vDx2ViT_GPT_AK}"
export AZURE_OPENAI_DEPLOYNAME="${AZURE_OPENAI_DEPLOYNAME:-gpt-5.4-mini-2026-03-17}"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-02-01}"
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-2048}"
# Azure resource endpoint (internal modelhub gateway; verified working).
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl}"

# All 14 BLINK sub-tasks (HF configs of mm-eval/BLINK)
BLINK_CONFIGS=(
  Art_Style
  Counting
  Forensic_Detection
  Functional_Correspondence
  IQ_Test
  Jigsaw
  Multi_view_Reasoning
  Object_Localization
  Relative_Depth
  Relative_Reflectance
  Semantic_Correspondence
  Spatial_Relation
  Visual_Correspondence
  Visual_Similarity
)

echo "============================================================"
echo "[CONFIG] model            : ${MODEL_NAME_OR_PATH}"
echo "[CONFIG] out_dir          : ${OUT_DIR}"
echo "[CONFIG] gen              : max_new_tokens=${MAX_NEW_TOKENS}, top_k=${TOP_K}, top_p=${TOP_P}, temperature=${TEMPERATURE}, do_sample=true (VLMEvalKit Qwen3-VL setup)"
echo "[CONFIG] gpu_per_parallel : ${GPU_PER_PARALLEL}"
echo "[CONFIG] GPUs             : ${GPUS}"
echo "[CONFIG] MMStar shards   : ${MMSTAR_PARALLEL} (${SHARDS_PER_GPU_MMSTAR}/GPU)"
echo "[CONFIG] BLINK shards    : ${BLINK_PARALLEL} (${SHARDS_PER_GPU_BLINK}/GPU)"
echo "[CONFIG] template         : ${TEMPLATE:-<dataset/default>}"
echo "[CONFIG] matching         : ${MATCHING_ORDER} (judge: ${JUDGE_PROVIDER} / ${JUDGE_MODEL})"
echo "[CONFIG] BLINK configs    : ${#BLINK_CONFIGS[@]} sub-tasks (split=val)"
echo "[CONFIG] MMStar           : default (split=val)"
echo "============================================================"

mkdir -p "${OUT_DIR}"

# Shared generation flags for run.py (parallel_per_task / CVD set per-dataset)
GEN_ARGS=(
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top_k "${TOP_K}"
  --top_p "${TOP_P}"
  --do_sample
  --gpu_per_parallel "${GPU_PER_PARALLEL}"
  --no_conda
)
if [ -n "${TEMPLATE}" ]; then
  GEN_ARGS+=(--template "${TEMPLATE}")
fi

# ---------------------------------------------------------------------------
# Phase 1: Inference  (BLINK group and MMStar group run in parallel)
# ---------------------------------------------------------------------------
run_blink() {
  local n=${#BLINK_CONFIGS[@]}
  local i=0
  for cfg in "${BLINK_CONFIGS[@]}"; do
    i=$((i + 1))
    echo "[$(date '+%H:%M:%S')] [BLINK ${i}/${n}] START ${cfg}"
    CUDA_VISIBLE_DEVICES="${BLINK_CVD}" python mmeval/run.py \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" --model_series "${MODEL_SERIES:-}" \
      --dataset "mmeval_hf@mm-eval/BLINK:${cfg}" \
      --split val \
      --out_dir "${OUT_DIR}/BLINK/${cfg}" \
      --parallel_per_task "${BLINK_PARALLEL}" \
      "${GEN_ARGS[@]}"
    echo "[$(date '+%H:%M:%S')] [BLINK ${i}/${n}] DONE  ${cfg} (exit=$?)"
  done
}

run_mmstar() {
  echo "[$(date '+%H:%M:%S')] [MMStar] START"
  CUDA_VISIBLE_DEVICES="${MMSTAR_CVD}" python mmeval/run.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" --model_series "${MODEL_SERIES:-}" \
    --dataset "mmeval_hf@mm-eval/MMStar" \
    --split val \
    --out_dir "${OUT_DIR}/MMStar" \
    --parallel_per_task "${MMSTAR_PARALLEL}" \
    "${GEN_ARGS[@]}"
  echo "[$(date '+%H:%M:%S')] [MMStar] DONE (exit=$?)"
}

if [ "${SEQUENTIAL:-0}" = "1" ]; then
  # Run MMStar then BLINK sequentially (no memory stacking) — needed when full-res
  # multi-image BLINK samples are too heavy to overlap with MMStar.
  echo "[PHASE 1] Inference (sequential: MMStar then BLINK) ..."
  run_mmstar > "${OUT_DIR}/infer_mmstar.log" 2>&1; RC_MMSTAR=$?
  run_blink  > "${OUT_DIR}/infer_blink.log"  2>&1; RC_BLINK=$?
else
  echo "[PHASE 1] Inference (two datasets in parallel) ..."
  run_blink  > "${OUT_DIR}/infer_blink.log"  2>&1 &
  PID_BLINK=$!
  run_mmstar > "${OUT_DIR}/infer_mmstar.log" 2>&1 &
  PID_MMSTAR=$!
  wait "${PID_MMSTAR}"; RC_MMSTAR=$?
  wait "${PID_BLINK}";  RC_BLINK=$?
fi
echo "[PHASE 1] inference done (blink rc=${RC_BLINK}, mmstar rc=${RC_MMSTAR})"
echo "          logs: ${OUT_DIR}/infer_blink.log , ${OUT_DIR}/infer_mmstar.log"

# ---------------------------------------------------------------------------
# Phase 2: Scoring  (LLM judge only)
# ---------------------------------------------------------------------------
if [[ "${JUDGE_PROVIDER}" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY must be set for the LLM judge (provider=openai)."
  exit 1
fi
if [[ "${JUDGE_PROVIDER}" == "azure_openai" ]]; then
  if [[ -z "${AZURE_OPENAI_KEY:-}" || -z "${AZURE_OPENAI_ENDPOINT:-}" ]]; then
    echo "[ERROR] AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT must be set for the azure_openai judge."
    exit 1
  fi
fi

echo "[PHASE 2] Scoring with matching cascade (matching_order=${MATCHING_ORDER}) ..."
python mmeval/score.py \
  --out_dir "${OUT_DIR}" \
  --score_result_glob '**/result.json' \
  --matching_order "${MATCHING_ORDER}" \
  --judge_provider "${JUDGE_PROVIDER}" \
  --judge_model "${JUDGE_MODEL}" \
  --score_output_name "${SCORE_OUTPUT_NAME}" \
  --parallel_per_task "${SCORE_PARALLEL}" \
  --score_save_freq 20

# ---------------------------------------------------------------------------
# Phase 3: Aggregate + report (BLINK = mean over 14 sub-tasks)
# ---------------------------------------------------------------------------
echo "[PHASE 3] Aggregating results ..."
python - "${OUT_DIR}" "${SCORE_OUTPUT_NAME}" <<'PY'
import glob, json, os, sys

out_dir, score_name = sys.argv[1], sys.argv[2]

def acc(path):
    with open(path) as f:
        return json.load(f)["summary"]

# BLINK: average accuracy across sub-task configs
blink_rows, blink_correct, blink_total = [], 0, 0
for p in sorted(glob.glob(os.path.join(out_dir, "BLINK", "*", score_name))):
    cfg = os.path.basename(os.path.dirname(p))
    s = acc(p)
    blink_rows.append((cfg, s["accuracy"], s["correct"], s["total"]))
    blink_correct += s["correct"]; blink_total += s["total"]

print("\n================ BLINK (split=val) ================")
for cfg, a, c, t in blink_rows:
    print(f"  {cfg:<26} acc={a:.4f}  ({c}/{t})")
if blink_rows:
    macro = sum(r[1] for r in blink_rows) / len(blink_rows)
    micro = blink_correct / blink_total if blink_total else 0.0
    print(f"  {'-'*40}")
    print(f"  BLINK score (micro, pooled {blink_correct}/{blink_total}): {micro:.4f}  <- reported metric")
    print(f"  BLINK macro-avg (mean of {len(blink_rows)} tasks)   : {macro:.4f}  (reference)")
else:
    print("  (no BLINK score files found)")

mm = os.path.join(out_dir, "MMStar", score_name)
print("\n================ MMStar (split=val) ===============")
if os.path.exists(mm):
    s = acc(mm)
    print(f"  MMStar acc={s['accuracy']:.4f}  ({s['correct']}/{s['total']})")
else:
    print("  (no MMStar score file found)")
print("===================================================\n")
PY

echo "[DONE] Scores written as ${SCORE_OUTPUT_NAME} under ${OUT_DIR}/<dataset>/..."
