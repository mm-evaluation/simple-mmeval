#!/usr/bin/env bash
#
# Full mm-eval suite for Qwen2.5-VL-7B-Instruct, aligned with VLMEvalKit:
#   - vision: processor defaults (qwenvl2d5.py no longer caps max_pixels)
#   - generation: temperature=0.01, top_k=1, top_p=0.001, max_new_tokens=2048
#   - prompt: VLMEvalKit ImageMCQDataset template; matching: template,llm-match
# Conservative shard counts (7B weights ~16GB) so an unattended run won't OOM.

set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${ROOT_DIR}"

export MODEL_NAME_OR_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
export OUT_DIR="work_dirs/Qwen2.5-VL-7B-Instruct_aligned"
# VLMEvalKit Qwen2.5-VL generation setup.
export TEMPERATURE=0.01
export TOP_K=1
export TOP_P=0.001
export MAX_NEW_TOKENS=2048

echo "############################################################"
echo "# Qwen2.5-VL-7B: MMStar + BLINK"
echo "############################################################"
# 7B (~16GB) + full-res vision: run MMStar/BLINK SEQUENTIALLY (no memory stacking),
# MMStar 2/GPU, BLINK 1/GPU (multi-image is heavy).
SEQUENTIAL=1 SHARDS_PER_GPU_MMSTAR=2 SHARDS_PER_GPU_BLINK=1 \
  bash scripts/examples/run_qwen3vl4b_blink_mmstar.sh

echo "############################################################"
echo "# Qwen2.5-VL-7B: MCQ suite (RealWorldQA/LogicVista/VStarBench/MMBench/MMBench-V11/ScienceQA-IMG) + HRBench"
echo "############################################################"
# 7B: 2/GPU. HRBench skipped — at 7B + full-res it would take days; enable separately
# with SKIP_HRBENCH=0 if needed.
SKIP_HRBENCH="${SKIP_HRBENCH:-1}" SHARDS_PER_GPU=2 \
  bash scripts/examples/run_qwen3vl4b_mcq_suite.sh

echo "[ALL DONE] Qwen2.5-VL-7B-Instruct full suite -> ${OUT_DIR}"
