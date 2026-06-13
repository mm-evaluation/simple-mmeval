#!/usr/bin/env bash
# FINAL stage (run LAST): HRBench4K + HRBench8K for both models.
#   - Qwen3-VL-4B-Instruct (base)  : HRBench4K resumes from cache (~705/800) + HRBench8K
#   - Qwen2.5-VL-7B-Instruct        : HRBench4K + HRBench8K
# Full-resolution images are ~11 min/sample -> this is a long (multi-hour to ~day) run.
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="./:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export JUDGE_MAX_CONCURRENCY=4 JUDGE_MAX_RETRY=3 JUDGE_MAX_TOKENS=2048
export AZURE_OPENAI_KEY="kEQGmoirhe9XZdDOCmE5MxAL3vDx2ViT_GPT_AK"
export AZURE_OPENAI_ENDPOINT="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl"
export AZURE_OPENAI_DEPLOYNAME="gpt-5.4-mini-2026-03-17"
export AZURE_OPENAI_API_VERSION="2024-02-01"

# args: <model> <out_dir> <shards_per_gpu> <max_new_tokens> <top_k> <top_p>
run_model_hrbench() {
  local model="$1" out="$2" spg="$3" mnt="$4" topk="$5" topp="$6"
  # GPU0 hosts a ~37GB vLLM co-tenant (PID 28104, run_dynamic.py target-util 75) — skip it
  # so full-res HRBench shards don't OOM contending with it. Use GPU1-7 only.
  local cvd; cvd=$(python3 -c "print(','.join(['1','2','3','4','5','6','7']*${spg}))")
  local nsh=$(( spg * 7 ))
  for spec in "HRBench4K hrbench_4k" "HRBench8K hrbench_8k"; do
    set -- $spec; local name=$1 split=$2
    echo "[$(date '+%m-%d %H:%M')] ${out##*/} : ${name}"
    if [ ! -f "${out}/${name}/result.json" ]; then
      CUDA_VISIBLE_DEVICES="${cvd}" python mmeval/run.py \
        --model_name_or_path "${model}" --dataset "mmeval_hf@mm-eval/${name}" --split "${split}" \
        --out_dir "${out}/${name}" --max_new_tokens "${mnt}" --temperature 0.01 --top_k "${topk}" \
        --top_p "${topp}" --do_sample --gpu_per_parallel 1 --parallel_per_task "${nsh}" --no_conda \
        --template scripts/templates/vlmevalkit_mcq.txt > "${out}/infer_${name}.log" 2>&1
    fi
    if [ -f "${out}/${name}/result.json" ]; then
      python mmeval/score.py --out_dir "${out}/${name}" --score_result_glob 'result.json' \
        --matching_order template,llm-match --judge_provider azure_openai --judge_model gpt-5.4-mini-2026-03-17 \
        --score_output_name score_llm.json --parallel_per_task 16 --no_score_resume > "${out}/score_${name}.log" 2>&1
      python -c "import json;s=json.load(open('${out}/${name}/score_llm.json'))['summary'];print('  [${name}] acc=%.4f (%d/%d)'%(s['accuracy'],s['correct'],s['total']))"
    fi
  done
}

echo "############ HRBench FINAL 1/3: Qwen3-VL-4B (base) ############"
run_model_hrbench "Qwen/Qwen3-VL-4B-Instruct" "work_dirs/Qwen3-VL-4B-Instruct_llmjudge" 3 4096 20 0.8

echo "############ HRBench FINAL 2/3: Vision-OPD-Qwen3-VL-4B-step65 ############"
# symlink basename = Qwen3-VL-4B-Instruct so get_series resolves qwen3_vl; merged ckpt ~9.65GB -> 2/GPU.
run_model_hrbench "work_dirs/_vopd_link/Qwen3-VL-4B-Instruct" "work_dirs/Vision-OPD-Qwen3-VL-4B-step65" 2 4096 20 0.8

echo "############ HRBench FINAL 3/3: Qwen2.5-VL-7B ############"
run_model_hrbench "Qwen/Qwen2.5-VL-7B-Instruct" "work_dirs/Qwen2.5-VL-7B-Instruct_aligned" 2 2048 1 0.001

echo "[HRBENCH FINAL DONE]"
