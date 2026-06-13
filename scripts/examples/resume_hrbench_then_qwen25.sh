#!/usr/bin/env bash
# Resume Qwen3-VL-4B HRBench (4K from cache + 8K), then run Qwen2.5-VL-7B full suite.
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="./:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export JUDGE_MAX_CONCURRENCY=4
export JUDGE_MAX_RETRY=3
export JUDGE_MAX_TOKENS=2048
export AZURE_OPENAI_KEY="kEQGmoirhe9XZdDOCmE5MxAL3vDx2ViT_GPT_AK"
export AZURE_OPENAI_ENDPOINT="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl"
export AZURE_OPENAI_DEPLOYNAME="gpt-5.4-mini-2026-03-17"
export AZURE_OPENAI_API_VERSION="2024-02-01"

d="work_dirs/Qwen3-VL-4B-Instruct_llmjudge"
CVD=$(python3 -c "print(','.join(['0','1','2','3','4','5','6','7']*4))")   # 4/GPU = 32 shards

for spec in "HRBench4K hrbench_4k" "HRBench8K hrbench_8k"; do
  set -- $spec; name=$1; split=$2
  echo "[$(date '+%H:%M:%S')] HRBench: ${name} (split=${split})"
  if [ ! -f "$d/$name/result.json" ]; then
    CUDA_VISIBLE_DEVICES="$CVD" python mmeval/run.py \
      --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
      --dataset "mmeval_hf@mm-eval/${name}" --split "${split}" --out_dir "$d/$name" \
      --max_new_tokens 4096 --temperature 0.01 --top_k 20 --top_p 0.8 --do_sample \
      --gpu_per_parallel 1 --parallel_per_task 32 --no_conda \
      --template scripts/templates/vlmevalkit_mcq.txt > "$d/infer_${name}.log" 2>&1
  fi
  if [ -f "$d/$name/result.json" ]; then
    python mmeval/score.py --out_dir "$d/$name" --score_result_glob 'result.json' \
      --matching_order template,llm-match --judge_provider azure_openai --judge_model gpt-5.4-mini-2026-03-17 \
      --score_output_name score_llm.json --parallel_per_task 16 --no_score_resume > "$d/score_${name}.log" 2>&1
    python -c "import json;s=json.load(open('$d/$name/score_llm.json'))['summary'];print('  [${name}] acc=%.4f (%d/%d)'%(s['accuracy'],s['correct'],s['total']))"
  fi
done

echo "[$(date '+%H:%M:%S')] Qwen3-4B HRBench done -> launching Qwen2.5-VL-7B full suite"
bash scripts/examples/run_qwen25vl7b_all.sh > work_dirs/qwen25vl7b_all.log 2>&1
echo "[ALL DONE]"
