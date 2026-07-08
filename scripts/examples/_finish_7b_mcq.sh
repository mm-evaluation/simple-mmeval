#!/usr/bin/env bash
# Finish the 2 remaining 7B MCQ (MMBench-V11, ScienceQA-IMG) on GPU1-7 (3/GPU=21 shards),
# avoiding GPU0 (leaked/zombie memory from BLINK). Then write the 7B done-marker so
# chain_tail2 advances. Datasets are cached -> offline.
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="./:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export JUDGE_MAX_CONCURRENCY=4 JUDGE_MAX_RETRY=3 JUDGE_MAX_TOKENS=2048
# Set AZURE_OPENAI_KEY / AZURE_OPENAI_ENDPOINT in your environment before running.
export AZURE_OPENAI_KEY="${AZURE_OPENAI_KEY:?set AZURE_OPENAI_KEY in the environment}"
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:?set AZURE_OPENAI_ENDPOINT in the environment}"
export AZURE_OPENAI_DEPLOYNAME="gpt-5.4-mini-2026-03-17"
export AZURE_OPENAI_API_VERSION="2024-02-01"

MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
OUT="work_dirs/Qwen2.5-VL-7B-Instruct_aligned"
CVD="1,2,3,4,5,6,7,1,2,3,4,5,6,7,1,2,3,4,5,6,7"   # 3/GPU on GPU1-7 = 21 shards
NSH=21

run_one() {  # name cfg split
  local name="$1" cfg="$2" split="$3" odir="${OUT}/$1"
  [ -f "${odir}/result.json" ] && { echo "[skip] ${name}"; return; }
  local ds="mm-eval/${name}"; [ "${cfg}" != "default" ] && ds="${ds}:${cfg}"
  echo "[$(date '+%H:%M')] ${name}: inference (21 shards, 3/GPU on GPU1-7)"
  CUDA_VISIBLE_DEVICES="${CVD}" python mmeval/run.py \
    --model_name_or_path "${MODEL}" --dataset "mmeval_hf@${ds}" --split "${split}" \
    --out_dir "${odir}" --max_new_tokens 2048 --temperature 0.01 --top_k 1 --top_p 0.001 \
    --do_sample --gpu_per_parallel 1 --parallel_per_task "${NSH}" --no_conda \
    --template scripts/templates/vlmevalkit_mcq.txt > "${OUT}/infer_${name}.log" 2>&1
  echo "  infer exit=$?"
  if [ -f "${odir}/result.json" ]; then
    python mmeval/score.py --out_dir "${odir}" --score_result_glob 'result.json' \
      --matching_order template,llm-match --judge_provider azure_openai --judge_model gpt-5.4-mini-2026-03-17 \
      --score_output_name score_llm.json --parallel_per_task 16 --no_score_resume > "${OUT}/score_${name}.log" 2>&1
    python -c "import json;s=json.load(open('${odir}/score_llm.json'))['summary'];print('  [%s] acc=%.4f (%d/%d)'%('${name}',s['accuracy'],s['correct'],s['total']))"
  fi
}

run_one MMBench-V11 en dev
run_one ScienceQA-IMG default validation

# wait until BLINK is fully scored (Visual_Similarity), then write the 7B done-marker
until [ -f "${OUT}/BLINK/Visual_Similarity/score_llm.json" ]; do sleep 20; done
echo "[ALL DONE] Qwen2.5-VL-7B-Instruct full suite -> ${OUT}" >> work_dirs/qwen25vl7b_all.log
echo "[finish_7b $(date '+%H:%M')] 7B done, marker written."
