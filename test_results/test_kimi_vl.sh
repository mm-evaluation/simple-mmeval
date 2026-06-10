#!/usr/bin/env bash
# Smoke test for kimi_vl (model: moonshotai/Kimi-VL-A3B-Instruct).
set -euo pipefail

REPO_DIR=${REPO_DIR:-/raid/ztw/simple-mmeval-skills-dev}
OUT_ROOT=${OUT_ROOT:-/raid/ztw/simple-mmeval-test-result/work_dirs/kimi_vl/$(date +%Y-%m-%d)}
GPU=${GPU:-0}
MODEL_ID=moonshotai/Kimi-VL-A3B-Instruct

cd "$REPO_DIR"
mkdir -p "$OUT_ROOT"

for spec in "no_media|32|" "single_image_start|64|tests/media/448"; do
  IFS='|' read -r sample maxtok imgdir <<< "$spec"
  out="$OUT_ROOT/$sample"
  extra=()
  [[ -n "$imgdir" ]] && extra=(--img_dir "$imgdir")
  PYTHONPATH="$REPO_DIR" ENV_DIR=/raid/ztw/envs CUDA_VISIBLE_DEVICES="$GPU" \
    /raid/ztw/envs/kimi_vl/bin/python mmeval/run.py \
      --model_name_or_path "$MODEL_ID" \
      --dataset local@json --infile tests/samples/$sample.json \
      --out_dir "$out" \
      --gpu_per_parallel 1 --parallel_per_task 1 --max_new_tokens "$maxtok" \
      "${extra[@]}"
done
