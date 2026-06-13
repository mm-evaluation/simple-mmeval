#!/usr/bin/env bash
# Test Vision-OPD-Qwen3-VL-4B global_step_65 on the same VLMEvalKit-aligned suite
# as the base Qwen3-VL-4B-Instruct (excluding HRBench), to compare deltas.
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${ROOT_DIR}"

# Symlink whose basename ("Qwen3-VL-4B-Instruct") makes get_series resolve qwen3_vl.
export MODEL_NAME_OR_PATH="work_dirs/_vopd_link/Qwen3-VL-4B-Instruct"
export OUT_DIR="work_dirs/Vision-OPD-Qwen3-VL-4B-step65"
# Generation/vision/matching all default to the base Qwen3-VL-4B aligned setup.

echo "############ Vision-OPD step65: MMStar + BLINK ############"
# Sequential (no memory stacking). This merged checkpoint is ~9.65GB (bigger than the
# 8.3GB base), so use conservative shards: MMStar 4/GPU, BLINK 1/GPU (full-res multi-image).
SEQUENTIAL=1 SHARDS_PER_GPU_MMSTAR=4 SHARDS_PER_GPU_BLINK=1 \
  bash scripts/examples/run_qwen3vl4b_blink_mmstar.sh

echo "############ Vision-OPD step65: MCQ suite (no HRBench) ############"
SKIP_HRBENCH=1 SHARDS_PER_GPU=3 bash scripts/examples/run_qwen3vl4b_mcq_suite.sh

echo "[VISION-OPD step65 DONE] -> ${OUT_DIR}"
