#!/usr/bin/env bash
#
# HRBench re-run at spg=1 (1 model instance per GPU, 8 shards) for step65.
# The othernode script used spg=2 (16 shards) which OOM'd: HRBench 4K/8K high-res
# images need ~30 GiB for a single attention op, and 2 instances/GPU don't fit.
# spg=1 gives each shard the full 79 GiB -> no OOM. Datasets already cached -> offline.
# Skips a dataset whose result.json already exists.   Usage: bash run_hrbench_spg1.sh
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="./:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL="work_dirs/_vopd_link/Qwen3-VL-4B-Instruct"
OUT_DIR="work_dirs/Vision-OPD-Qwen3-VL-4B-step65"
TEMPLATE="scripts/templates/vlmevalkit_mcq.txt"

export JUDGE_MAX_CONCURRENCY=4 JUDGE_MAX_RETRY=3 JUDGE_MAX_TOKENS=2048
# Set AZURE_OPENAI_KEY / AZURE_OPENAI_ENDPOINT in your environment before running.
export AZURE_OPENAI_KEY="${AZURE_OPENAI_KEY:?set AZURE_OPENAI_KEY in the environment}"
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:?set AZURE_OPENAI_ENDPOINT in the environment}"
export AZURE_OPENAI_DEPLOYNAME="gpt-5.4-mini-2026-03-17"
export AZURE_OPENAI_API_VERSION="2024-02-01"

MNT=4096 TEMP=0.01 TOPK=20 TOPP=0.8

# name  config  split        shards_per_gpu
# bf16 + flash-attn keeps attention memory O(N); spg=3 (24 shards) overlaps decode
# (bandwidth-bound, leaves SMs idle) across samples. spg=3 is the safe ceiling: 8K
# contexts (~39-49k tokens) push KV-cache peak to ~20-27GB/instance -> ~60-75GB/GPU.
HR_SPECS=(
  "HRBench4K default hrbench_4k 3"
  "HRBench8K default hrbench_8k 3"
)

run_one() {
  local name="$1" cfg="$2" split="$3" spg="$4"
  local odir="${OUT_DIR}/${name}"
  if [ -f "${odir}/result.json" ]; then echo "[skip] ${name}: result.json exists"; return; fi
  # fresh: clear any partial shard state from the failed spg=2 run
  rm -rf "${odir}"; mkdir -p "${odir}"

  local ds="mm-eval/${name}"; [ "${cfg}" != "default" ] && ds="${ds}:${cfg}"
  local cvd; cvd=$(python3 -c "print(','.join(['0','1','2','3','4','5','6','7']*${spg}))")
  local nsh=$(( spg * 8 ))
  echo "[$(date '+%H:%M')] ${name}: inference (${nsh} shards, ${spg}/GPU)"
  # flash_attention_2: HRBench images are uncapped high-res (~20k+ vision tokens);
  # sdpa materializes a ~30 GiB attention matrix -> OOM even at 1 inst/GPU. Flash-attn
  # is O(N) memory, keeps full resolution (no eval-quality loss), and is faster.
  # dtype bfloat16: required by flash-attn (the ckpt's "auto" was loading as fp32).
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="${cvd}" python mmeval/run.py \
    --model_name_or_path "${MODEL}" --dataset "mmeval_hf@${ds}" --split "${split}" \
    --out_dir "${odir}" --max_new_tokens "${MNT}" --temperature "${TEMP}" --top_k "${TOPK}" \
    --top_p "${TOPP}" --do_sample --dtype bfloat16 --attn_implementation flash_attention_2 \
    --gpu_per_parallel 1 --parallel_per_task "${nsh}" --no_conda \
    --template "${TEMPLATE}" > "${OUT_DIR}/infer_${name}.log" 2>&1
  echo "  inference exit=$?"

  if [ -f "${odir}/result.json" ]; then
    HF_HUB_OFFLINE=1 python mmeval/score.py --out_dir "${odir}" --score_result_glob 'result.json' \
      --matching_order template,llm-match --judge_provider azure_openai --judge_model gpt-5.4-mini-2026-03-17 \
      --score_output_name score_llm.json --parallel_per_task 16 --no_score_resume \
      > "${OUT_DIR}/score_${name}.log" 2>&1
    python -c "import json;s=json.load(open('${odir}/score_llm.json'))['summary'];print('  [%s] acc=%.4f (%d/%d)'%('${name}',s['accuracy'],s['correct'],s['total']))"
  else
    echo "  [err] ${name}: no result.json after inference"
  fi
}

echo "############ HRBench re-run (spg=1) — step65 ############"
for spec in "${HR_SPECS[@]}"; do run_one $spec; done
echo "[HRBENCH SPG1 DONE]"
