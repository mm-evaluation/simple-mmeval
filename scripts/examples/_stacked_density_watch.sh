#!/usr/bin/env bash
# One-shot watcher: when MMBench (the in-flight 2/GPU dataset) finishes, restart the
# remaining MCQ (MMBench-V11, ScienceQA-IMG) at 3/GPU on GPU1-7 (21 shards), then write
# the 7B done-marker so chain_tail2 proceeds. Keeps GPU0 clear for BLINK's last task.
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${ROOT_DIR}"
OUT="work_dirs/Qwen2.5-VL-7B-Instruct_aligned"

echo "[watch $(date '+%H:%M')] waiting for MMBench result.json (current 2/GPU run to finish it)..."
until [ -f "${OUT}/MMBench/result.json" ]; do sleep 20; done
echo "[watch $(date '+%H:%M')] MMBench done -> kill 2/GPU stacked, relaunch remaining at 3/GPU"

# kill the current stacked suite tree (it would otherwise do MMBench-V11/ScienceQA at 2/GPU)
for p in $(pgrep -f run_qwen3vl4b_mcq_suite.sh); do kill "$p" 2>/dev/null; done
sleep 3
for p in $(pgrep -f run_qwen3vl4b_mcq_suite.sh); do kill -9 "$p" 2>/dev/null; done
sleep 2

# BLINK is done by now, so use ALL 8 GPUs at 3/GPU = 24 shards. Auto-skips the 4 datasets
# that already have result.json -> runs only MMBench-V11 + ScienceQA-IMG. Offline (cached).
GPUS=0,1,2,3,4,5,6,7 SHARDS_PER_GPU=3 SKIP_HRBENCH=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
  OUT_DIR="${OUT}" \
  TEMPERATURE=0.01 TOP_K=1 TOP_P=0.001 MAX_NEW_TOKENS=2048 HF_HUB_OFFLINE=1 \
  bash scripts/examples/run_qwen3vl4b_mcq_suite.sh > work_dirs/qwen25_mcq_dense.log 2>&1
echo "[watch $(date '+%H:%M')] dense MCQ done."

# wait for BLINK to be fully complete (14/14) too
until [ "$(ls ${OUT}/BLINK/ 2>/dev/null | wc -l)" -ge 14 ]; do sleep 20; done

# write the 7B done-marker so chain_tail2 advances to Vision-OPD redo + HRBench
echo "[ALL DONE] Qwen2.5-VL-7B-Instruct full suite -> ${OUT}" >> work_dirs/qwen25vl7b_all.log
echo "[watch $(date '+%H:%M')] 7B fully done, marker written."
