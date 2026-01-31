#!/bin/bash
# Run all chat-test scripts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running all chat-test scripts..."

# Qwen series
echo "=== Running Qwen2-VL chat test ==="
bash "$SCRIPT_DIR/qwenvl2-chat-test.sh"

echo "=== Running Qwen2.5-VL chat test ==="
bash "$SCRIPT_DIR/qwenvl2d5-chat-test.sh"

echo "=== Running Qwen2.5-Omni chat test ==="
bash "$SCRIPT_DIR/qwenvl2d5-omni-chat-test.sh"

echo "=== Running Qwen3-VL chat test ==="
bash "$SCRIPT_DIR/qwen3-vl-chat-test.sh"

echo "=== Running Qwen3-Omni chat test ==="
bash "$SCRIPT_DIR/qwen3-omni-chat-test.sh"

# LLaVA-OneVision series
echo "=== Running LLaVA-OneVision chat test ==="
bash "$SCRIPT_DIR/llava-ov-chat-test.sh"

# echo "=== Running LLaVA-OneVision-1.5 chat test ==="
# bash "$SCRIPT_DIR/llava-ov-1d5-chat-test.sh"

# InternVL series
echo "=== Running InternVL-Chat chat test ==="
bash "$SCRIPT_DIR/internvl-chat-chat-test.sh"

echo "=== Running InternVL-Chat V1.5 chat test ==="
bash "$SCRIPT_DIR/internvl-chat1d5-chat-test.sh"

echo "=== Running InternVL2 chat test ==="
bash "$SCRIPT_DIR/internvl2-chat-test.sh"

echo "=== Running InternVL2.5 chat test ==="
bash "$SCRIPT_DIR/internvl2d5-chat-test.sh"

echo "=== Running InternVL3 chat test ==="
bash "$SCRIPT_DIR/internvl3-chat-test.sh"

echo "=== Running InternVL3.5 chat test ==="
bash "$SCRIPT_DIR/internvl3d5-chat-test.sh"

echo "All chat-test scripts completed!"

