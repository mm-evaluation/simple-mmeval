#!/usr/bin/env bash
#
# Run Qwen3-VL-4B-Instruct on a suite of mm-eval HF MCQ benchmarks, aligned with
# VLMEvalKit (prompt template + vision params + generation), and score each with
# the robust template -> llm-match cascade. One dataset at a time, all GPUs each.
#
# Usage: OPENAI/AZURE creds in env, then: bash scripts/examples/run_qwen3vl4b_mcq_suite.sh
#
# Each spec is "name:config:split". Edit DATASETS to add/remove.

set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="./:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
OUT_DIR="${OUT_DIR:-work_dirs/Qwen3-VL-4B-Instruct_llmjudge}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SHARDS_PER_GPU="${SHARDS_PER_GPU:-7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPLATE="${TEMPLATE:-scripts/templates/vlmevalkit_mcq.txt}"
MATCHING_ORDER="${MATCHING_ORDER:-template,llm-match}"
SCORE_PARALLEL="${SCORE_PARALLEL:-16}"

# LLM-match / judge (Azure) — capped concurrency avoids quota-saturation degradation.
export JUDGE_MAX_CONCURRENCY="${JUDGE_MAX_CONCURRENCY:-4}"
export JUDGE_MAX_RETRY="${JUDGE_MAX_RETRY:-3}"
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-2048}"
export AZURE_OPENAI_KEY="${AZURE_OPENAI_KEY:?set AZURE_OPENAI_KEY in the environment}"
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:?set AZURE_OPENAI_ENDPOINT in the environment}"
export AZURE_OPENAI_DEPLOYNAME="${AZURE_OPENAI_DEPLOYNAME:-gpt-5.4-mini-2026-03-17}"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-02-01}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-azure_openai}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-mini-2026-03-17}"

# name:config:split  (config "default" -> omit from dataset spec)
DATASETS=(
  "RealWorldQA:default:test"
  "LogicVista:default:test"
  "VStarBench:default:test"
  # HRBench: 4K/8K full-res images run ~11 min/sample (~13-14h total) in this HF
  # batch-1 setup. Enabled for an overnight run; resumes from cache.
  "HRBench4K:default:hrbench_4k"
  "HRBench8K:default:hrbench_8k"
  "MMBench:en:dev"                      # original MMBench V1.0 dev (4329)
  "MMBench-V11:en:dev"                  # MMBench V1.1 dev (4876) — version-aligned w/ official
  "ScienceQA-IMG:default:validation"   # image-only subset (2097), aligns with VLMEvalKit
  # "SEEDBench:default:test"           # 14k samples (~10h) — skipped per request; enable to run
)

# Build a repeated CUDA_VISIBLE_DEVICES (SHARDS_PER_GPU copies of each GPU).
IFS=',' read -ra _gpus <<< "${GPUS}"
CVD=""; NSHARD=0
for ((r = 0; r < SHARDS_PER_GPU; r++)); do for g in "${_gpus[@]}"; do CVD+="${g},"; NSHARD=$((NSHARD + 1)); done; done
CVD="${CVD%,}"

# Generation params default to the VLMEvalKit Qwen3-VL setup; override via env for
# other models (e.g. Qwen2.5-VL: TEMPERATURE=0.01 TOP_K=1 TOP_P=0.001 MAX_NEW_TOKENS=2048).
TEMPERATURE="${TEMPERATURE:-0.01}"
TOP_K="${TOP_K:-20}"
TOP_P="${TOP_P:-0.8}"
GEN_ARGS=( --max_new_tokens "${MAX_NEW_TOKENS}" --temperature "${TEMPERATURE}" --top_k "${TOP_K}" --top_p "${TOP_P}" --do_sample
           --gpu_per_parallel 1 --parallel_per_task "${NSHARD}" --no_conda --template "${TEMPLATE}" )

echo "[CONFIG] model=${MODEL_NAME_OR_PATH} shards=${NSHARD} (${SHARDS_PER_GPU}/GPU) max_new_tokens=${MAX_NEW_TOKENS}"
echo "[CONFIG] template=${TEMPLATE} matching=${MATCHING_ORDER} judge=${JUDGE_PROVIDER}/${JUDGE_MODEL}"
echo "[CONFIG] datasets: ${DATASETS[*]}"
mkdir -p "${OUT_DIR}"

for spec in "${DATASETS[@]}"; do
  IFS=':' read -r name cfg split <<< "${spec}"
  [ "${SKIP_HRBENCH:-0}" = "1" ] && [[ "${name}" == HRBench* ]] && continue
  ds_spec="mmeval_hf@mm-eval/${name}"
  [ "${cfg}" != "default" ] && ds_spec="${ds_spec}:${cfg}"
  odir="${OUT_DIR}/${name}"
  echo "============================================================"
  echo "[$(date '+%H:%M:%S')] START ${name} (${ds_spec} split=${split})"
  if [ -f "${odir}/result.json" ]; then
    echo "  result.json exists -> skip inference"
  else
    CUDA_VISIBLE_DEVICES="${CVD}" python mmeval/run.py \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" --model_series "${MODEL_SERIES:-}" --dataset "${ds_spec}" --split "${split}" \
      --out_dir "${odir}" "${GEN_ARGS[@]}" > "${OUT_DIR}/infer_${name}.log" 2>&1
    echo "  inference exit=$? ($(date '+%H:%M:%S'))"
  fi
  if [ -f "${odir}/result.json" ]; then
    python mmeval/score.py --out_dir "${odir}" --score_result_glob 'result.json' \
      --matching_order "${MATCHING_ORDER}" --judge_provider "${JUDGE_PROVIDER}" --judge_model "${JUDGE_MODEL}" \
      --score_output_name score_llm.json --parallel_per_task "${SCORE_PARALLEL}" --no_score_resume \
      > "${OUT_DIR}/score_${name}.log" 2>&1
    acc=$(python -c "import json;print('%.4f (%d/%d)'%((lambda s:(s['accuracy'],s['correct'],s['total']))(json.load(open('${odir}/score_llm.json'))['summary'])))" 2>/dev/null)
    echo "  [${name}] acc=${acc}"
  fi
done

echo "============================================================"
echo "[SUITE DONE] summary:"
for spec in "${DATASETS[@]}"; do
  IFS=':' read -r name cfg split <<< "${spec}"
  [ "${SKIP_HRBENCH:-0}" = "1" ] && [[ "${name}" == HRBench* ]] && continue
  f="${OUT_DIR}/${name}/score_llm.json"
  [ -f "${f}" ] && python -c "import json;s=json.load(open('${f}'))['summary'];print('  %-14s %.4f (%d/%d)'%('${name}',s['accuracy'],s['correct'],s['total']))" 2>/dev/null || echo "  ${name}: (no score)"
done
echo "[DONE]"
