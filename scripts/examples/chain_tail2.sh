#!/usr/bin/env bash
# Recovery tail (replaces chain_all after the 429 break):
#   wait for Qwen2.5-VL-7B suite -> Vision-OPD MCQ redo (4 datasets that 429'd) -> HRBench (3 models).
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${ROOT_DIR}"
export HF_HUB_OFFLINE=1

echo "[tail $(date '+%m-%d %H:%M')] waiting for Qwen2.5-VL-7B suite to finish..."
until grep -q "\[ALL DONE\] Qwen2.5-VL-7B" "work_dirs/qwen25vl7b_all.log" 2>/dev/null; do sleep 60; done
echo "[tail $(date '+%m-%d %H:%M')] 7B done."

# Vision-OPD MCQ redo: the suite skips datasets whose result.json exists, so this only
# re-runs the 4 that failed (VStarBench, MMBench, MMBench-V11, ScienceQA-IMG). Offline = cached.
echo "[tail $(date '+%m-%d %H:%M')] Vision-OPD MCQ redo (offline)..."
sleep 20
SKIP_HRBENCH=1 SHARDS_PER_GPU=3 \
  MODEL_NAME_OR_PATH="work_dirs/_vopd_link/Qwen3-VL-4B-Instruct" \
  OUT_DIR="work_dirs/Vision-OPD-Qwen3-VL-4B-step65" \
  bash scripts/examples/run_qwen3vl4b_mcq_suite.sh > work_dirs/vision_opd_mcq_redo.log 2>&1
echo "[tail $(date '+%m-%d %H:%M')] Vision-OPD MCQ redo done."

echo "[tail $(date '+%m-%d %H:%M')] waiting for HRBench dataset pre-download to finish..."
until grep -q "ALL HRBENCH CACHED" "work_dirs/dl_hrbench.log" 2>/dev/null; do sleep 30; done
echo "[tail $(date '+%m-%d %H:%M')] HRBench datasets cached -> HRBench FINAL (3 models)..."
sleep 20
bash scripts/examples/run_hrbench_final.sh > work_dirs/hrbench_final.log 2>&1
echo "[tail $(date '+%m-%d %H:%M')] ALL DONE"
