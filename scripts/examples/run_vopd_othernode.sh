#!/usr/bin/env bash
#
# CROSS-NODE runner: Vision-OPD-Qwen3-VL-4B-step65 remaining work.
# Run this on a SECOND node (8 GPUs) while node-1 finishes Qwen2.5-VL-7B.
# Everything lives on shared /mnt, so results land in the same OUT_DIR for comparison.
#
#   Stage A (4 MCQ that 429'd):  VStarBench, MMBench, MMBench-V11, ScienceQA-IMG
#   Stage B (HRBench, optional): HRBench4K, HRBench8K   (set RUN_HRBENCH=1; ~13-14h)
#
# Self-contained: pre-downloads each dataset in a SINGLE process (avoids the 429 that
# concurrent shards triggered), then runs offline. Skips any dataset whose result.json
# already exists. Usage:   bash scripts/examples/run_vopd_othernode.sh
#                          RUN_HRBENCH=1 bash scripts/examples/run_vopd_othernode.sh

set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="./:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL="work_dirs/_vopd_link/Qwen3-VL-4B-Instruct"   # symlink -> global_step_65 merged ckpt (~9.65GB)
OUT_DIR="work_dirs/Vision-OPD-Qwen3-VL-4B-step65"    # shared; results land here
TEMPLATE="scripts/templates/vlmevalkit_mcq.txt"

# Azure judge (template,llm-match). Capped concurrency avoids quota-saturation degradation.
export JUDGE_MAX_CONCURRENCY=4 JUDGE_MAX_RETRY=3 JUDGE_MAX_TOKENS=2048
export AZURE_OPENAI_KEY="kEQGmoirhe9XZdDOCmE5MxAL3vDx2ViT_GPT_AK"
export AZURE_OPENAI_ENDPOINT="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl"
export AZURE_OPENAI_DEPLOYNAME="gpt-5.4-mini-2026-03-17"
export AZURE_OPENAI_API_VERSION="2024-02-01"

# Qwen3-VL generation setup (matches base/Vision-OPD on node-1).
MNT=4096 TEMP=0.01 TOPK=20 TOPP=0.8

# specs: "name config split shards_per_gpu"   (config "default" -> omitted from dataset id)
MCQ_SPECS=(
  "VStarBench   default test       3"
  "MMBench      en      dev        3"
  "MMBench-V11  en      dev        3"
  "ScienceQA-IMG default validation 3"
)
HR_SPECS=(
  "HRBench4K    default hrbench_4k 2"
  "HRBench8K    default hrbench_8k 2"
)

run_one() {
  local name="$1" cfg="$2" split="$3" spg="$4"
  local odir="${OUT_DIR}/${name}"
  if [ -f "${odir}/result.json" ]; then echo "[skip] ${name}: result.json exists"; return; fi

  local ds="mm-eval/${name}"; [ "${cfg}" != "default" ] && ds="${ds}:${cfg}"
  # --- pre-download (single process, ONLINE) so concurrent shards don't 429 ---
  echo "[$(date '+%H:%M')] ${name}: pre-download (single proc)"
  HF_HUB_OFFLINE=0 python - "$name" "$cfg" "$split" <<'PY'
import sys, os
os.environ.pop("HF_HUB_OFFLINE", None)
from datasets import load_dataset
name, cfg, split = sys.argv[1], sys.argv[2], sys.argv[3]
kw = {} if cfg == "default" else {"name": cfg}
ds = load_dataset(f"mm-eval/{name}", split=split, **kw)
print(f"  cached {name}: {len(ds)} rows", flush=True)
PY
  [ $? -ne 0 ] && { echo "[err] ${name}: download failed"; return; }

  local cvd; cvd=$(python3 -c "print(','.join(['0','1','2','3','4','5','6','7']*${spg}))")
  local nsh=$(( spg * 8 ))
  echo "[$(date '+%H:%M')] ${name}: inference (${nsh} shards, ${spg}/GPU)"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="${cvd}" python mmeval/run.py \
    --model_name_or_path "${MODEL}" --dataset "mmeval_hf@${ds}" --split "${split}" \
    --out_dir "${odir}" --max_new_tokens "${MNT}" --temperature "${TEMP}" --top_k "${TOPK}" \
    --top_p "${TOPP}" --do_sample --gpu_per_parallel 1 --parallel_per_task "${nsh}" --no_conda \
    --template "${TEMPLATE}" > "${OUT_DIR}/infer_${name}.log" 2>&1
  echo "  inference exit=$?"

  if [ -f "${odir}/result.json" ]; then
    HF_HUB_OFFLINE=1 python mmeval/score.py --out_dir "${odir}" --score_result_glob 'result.json' \
      --matching_order template,llm-match --judge_provider azure_openai --judge_model gpt-5.4-mini-2026-03-17 \
      --score_output_name score_llm.json --parallel_per_task 16 --no_score_resume \
      > "${OUT_DIR}/score_${name}.log" 2>&1
    python -c "import json;s=json.load(open('${odir}/score_llm.json'))['summary'];print('  [%s] acc=%.4f (%d/%d)'%('${name}',s['accuracy'],s['correct'],s['total']))"
  fi
}

echo "############ Vision-OPD (other node): Stage A — 4 MCQ ############"
for spec in "${MCQ_SPECS[@]}"; do run_one $spec; done

if [ "${RUN_HRBENCH:-0}" = "1" ]; then
  echo "############ Vision-OPD (other node): Stage B — HRBench ############"
  for spec in "${HR_SPECS[@]}"; do run_one $spec; done
fi
echo "[VOPD OTHER-NODE DONE]"
