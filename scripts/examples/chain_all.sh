#!/usr/bin/env bash
# Full autonomous chain: Vision-OPD step65 (running) -> Qwen2.5-VL-7B suite -> HRBench (3 models, LAST).
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${ROOT_DIR}"

echo "[chain $(date '+%m-%d %H:%M')] waiting for Vision-OPD step65 to finish..."
until grep -q "VISION-OPD step65 DONE" "work_dirs/vision_opd_step65.log" 2>/dev/null; do sleep 60; done

echo "[chain $(date '+%m-%d %H:%M')] Vision-OPD done -> Qwen2.5-VL-7B suite"
sleep 20
bash scripts/examples/run_qwen25vl7b_all.sh > work_dirs/qwen25vl7b_all.log 2>&1

echo "[chain $(date '+%m-%d %H:%M')] 7B suite done -> HRBench FINAL (Qwen3-VL-4B + Vision-OPD-step65 + Qwen2.5-VL-7B)"
sleep 20
bash scripts/examples/run_hrbench_final.sh > work_dirs/hrbench_final.log 2>&1

echo "[chain $(date '+%m-%d %H:%M')] ALL DONE"
