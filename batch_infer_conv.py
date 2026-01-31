#!/usr/bin/env python3
"""
Unified batch inference script for all models in the registry.
Supports filtering by model series and GPU count.
"""

import json
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import sys
import os

# Import registry
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mmeval.registry import series_mapping

# Model configurations with GPU requirements, HF paths, sanity check status, supported modalities, and series information
# Based on registry - sanity_check means NOT commented in registry
# Modalities determined from modality_test scripts
# Series information matches registry series_mapping
MODEL_CONFIGS = {
    
    # Gemma3 series - SANITY_CHECK
    "gemma-3-4b-it": {"gpu_requirement": 1, "hf_path": "google/gemma-3-4b-it", "sanity_check": True, "modalities": ["multi_image_interleave"], "series": "gemma3", "model_type": "instruct", "quantization": False},
    "gemma-3-12b-it": {"gpu_requirement": 1, "hf_path": "google/gemma-3-12b-it", "sanity_check": True, "modalities": ["multi_image_interleave"], "series": "gemma3", "model_type": "instruct", "quantization": False},
    "gemma-3-27b-it": {"gpu_requirement": 4, "hf_path": "google/gemma-3-27b-it", "sanity_check": True, "modalities": ["multi_image_interleave"], "series": "gemma3", "model_type": "instruct", "quantization": False},
    
    # Phi3v series - NO SANITY_CHECK
    "Phi-3-vision-128k-instruct": {"gpu_requirement": 8, "hf_path": "microsoft/Phi-3-vision-128k-instruct", "sanity_check": False, "modalities": ["multi_image_interleave"], "series": "phi3v", "model_type": "instruct", "quantization": False},

    # Phi3d5 series - NO SANITY_CHECK
    "Phi-3.5-vision-instruct": {"gpu_requirement": 8, "hf_path": "microsoft/Phi-3.5-vision-instruct", "sanity_check": False, "modalities": ["multi_image_interleave"], "series": "phi3d5", "model_type": "instruct", "quantization": False},
    
    # Phi4mm series - NO SANITY_CHECK
    "Phi-4-multimodal-instruct": {"gpu_requirement": 1, "hf_path": "microsoft/Phi-4-multimodal-instruct", "sanity_check": False, "modalities": ["multi_image_interleave"], "series": "phi4mm", "model_type": "instruct", "quantization": False},
  
    # QwenVL2 series - NO SANITY_CHECK
    "Qwen2-VL-2B-Instruct": {"gpu_requirement": 1, "hf_path": "Qwen/Qwen2-VL-2B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwenvl2", "model_type": "instruct", "quantization": False},
    "Qwen2-VL-7B-Instruct": {"gpu_requirement": 1, "hf_path": "Qwen/Qwen2-VL-7B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwenvl2", "model_type": "instruct", "quantization": False},
    "Qwen2-VL-72B-Instruct": {"gpu_requirement": 4, "hf_path": "Qwen/Qwen2-VL-72B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwenvl2", "model_type": "instruct", "quantization": False},

    # QwenVL2.5 series - NO SANITY_CHECK
    "Qwen2.5-VL-3B-Instruct": {"gpu_requirement": 1, "hf_path": "Qwen/Qwen2.5-VL-3B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwenvl2d5", "model_type": "instruct", "quantization": False},
    "Qwen2.5-VL-7B-Instruct": {"gpu_requirement": 1, "hf_path": "Qwen/Qwen2.5-VL-7B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwenvl2d5", "model_type": "instruct", "quantization": False},
    "Qwen2.5-VL-32B-Instruct": {"gpu_requirement": 2, "hf_path": "Qwen/Qwen2.5-VL-32B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwenvl2d5", "model_type": "instruct", "quantization": False},
    "Qwen2.5-VL-72B-Instruct": {"gpu_requirement": 4, "hf_path": "Qwen/Qwen2.5-VL-72B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwenvl2d5", "model_type": "instruct", "quantization": False},

    # Qwen3-VL series - NO SANITY_CHECK
    "Qwen3-VL-2B-Instruct": {"gpu_requirement": 1, "hf_path": "Qwen/Qwen3-VL-2B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwen3_vl", "model_type": "instruct", "quantization": False},
    "Qwen3-VL-4B-Instruct": {"gpu_requirement": 1, "hf_path": "Qwen/Qwen3-VL-4B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwen3_vl", "model_type": "instruct", "quantization": False},
    "Qwen3-VL-8B-Instruct": {"gpu_requirement": 1, "hf_path": "Qwen/Qwen3-VL-8B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwen3_vl", "model_type": "instruct", "quantization": False},
    "Qwen3-VL-32B-Instruct": {"gpu_requirement": 8, "hf_path": "Qwen/Qwen3-VL-32B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwen3_vl", "model_type": "instruct", "quantization": False},
    # "Qwen3-VL-2B-Thinking": {"gpu_requirement": 1, "hf_path": "Qwen/Qwen3-VL-2B-Thinking", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwen3_vl", "model_type": "reasoning", "quantization": False},
    # "Qwen3-VL-4B-Thinking": {"gpu_requirement": 1, "hf_path": "Qwen/Qwen3-VL-4B-Thinking", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwen3_vl", "model_type": "reasoning", "quantization": False},
    # "Qwen3-VL-8B-Thinking": {"gpu_requirement": 1, "hf_path": "Qwen/Qwen3-VL-8B-Thinking", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwen3_vl", "model_type": "reasoning", "quantization": False},
    # "Qwen3-VL-32B-Thinking": {"gpu_requirement":8, "hf_path": "Qwen/Qwen3-VL-32B-Thinking", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwen3_vl", "model_type": "reasoning", "quantization": False},
    # "Qwen3-VL-30B-A3B-Instruct": {"gpu_requirement": 2, "hf_path": "Qwen/Qwen3-VL-30B-A3B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwen3_vl", "model_type": "instruct", "quantization": False},
    # "Qwen3-VL-30B-A3B-Thinking": {"gpu_requirement": 2, "hf_path": "Qwen/Qwen3-VL-30B-A3B-Thinking", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwen3_vl", "model_type": "reasoning", "quantization": False},
    # "Qwen3-VL-235B-A22B-Instruct": {"gpu_requirement": 8, "hf_path": "Qwen/Qwen3-VL-235B-A22B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwen3_vl", "model_type": "instruct", "quantization": False},
    # "Qwen3-VL-235B-A22B-Thinking": {"gpu_requirement": 8, "hf_path": "Qwen/Qwen3-VL-235B-A22B-Thinking", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "qwen3_vl", "model_type": "reasoning", "quantization": False},

    # InternVL2.5 series - SANITY_CHECK
    "InternVL2_5-1B": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL2_5-1B", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl2d5", "model_type": "instruct", "quantization": False},
    "InternVL2_5-2B": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL2_5-2B", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl2d5", "model_type": "instruct", "quantization": False},
    "InternVL2_5-4B": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL2_5-4B", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl2d5", "model_type": "instruct", "quantization": False},
    "InternVL2_5-8B": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL2_5-8B", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl2d5", "model_type": "instruct", "quantization": False},
    "InternVL2_5-26B": {"gpu_requirement": 2, "hf_path": "OpenGVLab/InternVL2_5-26B", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl2d5", "model_type": "instruct", "quantization": False},
    "InternVL2_5-38B": {"gpu_requirement": 2, "hf_path": "OpenGVLab/InternVL2_5-38B", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl2d5", "model_type": "instruct", "quantization": False},
    "InternVL2_5-78B": {"gpu_requirement": 4, "hf_path": "OpenGVLab/InternVL2_5-78B", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl2d5", "model_type": "instruct", "quantization": False},
    # "InternVL2_5-1B-MPO": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL2_5-1B-MPO", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl2d5", "model_type": "instruct", "quantization": False},
    # "InternVL2_5-2B-MPO": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL2_5-2B-MPO", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl2d5", "model_type": "instruct", "quantization": False},
    # "InternVL2_5-4B-MPO": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL2_5-4B-MPO", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl2d5", "model_type": "instruct", "quantization": False},
    # "InternVL2_5-8B-MPO": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL2_5-8B-MPO", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl2d5", "model_type": "instruct", "quantization": False},
    # "InternVL2_5-26B-MPO": {"gpu_requirement": 2, "hf_path": "OpenGVLab/InternVL2_5-26B-MPO", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl2d5", "model_type": "instruct", "quantization": False},
    # "InternVL2_5-38B-MPO": {"gpu_requirement": 2, "hf_path": "OpenGVLab/InternVL2_5-38B-MPO", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl2d5", "model_type": "instruct", "quantization": False},
    # "InternVL2_5-78B-MPO": {"gpu_requirement": 4, "hf_path": "OpenGVLab/InternVL2_5-78B-MPO", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl2d5", "model_type": "instruct", "quantization": False},
    
    # InternVL3 series - SANITY_CHECK
    # "InternVL3-1B": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3-1B", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "reasoning", "quantization": False},
    # "InternVL3-2B": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3-2B", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "reasoning", "quantization": False},
    # "InternVL3-8B": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3-8B", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "reasoning", "quantization": False},
    # "InternVL3-9B": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3-9B", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "reasoning", "quantization": False},
    # "InternVL3-14B": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3-14B", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "reasoning", "quantization": False},
    # "InternVL3-38B": {"gpu_requirement": 2, "hf_path": "OpenGVLab/InternVL3-38B", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "reasoning", "quantization": False},
    # "InternVL3-78B": {"gpu_requirement": 4, "hf_path": "OpenGVLab/InternVL3-78B", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "reasoning", "quantization": False},
    "InternVL3-1B-Instruct": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3-1B-Instruct", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "instruct", "quantization": False},
    "InternVL3-2B-Instruct": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3-2B-Instruct", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "instruct", "quantization": False},
    "InternVL3-8B-Instruct": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3-8B-Instruct", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "instruct", "quantization": False},
    "InternVL3-9B-Instruct": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3-9B-Instruct", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "instruct", "quantization": False},
    "InternVL3-14B-Instruct": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3-14B-Instruct", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "instruct", "quantization": False},
    "InternVL3-38B-Instruct": {"gpu_requirement": 2, "hf_path": "OpenGVLab/InternVL3-38B-Instruct", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "instruct", "quantization": False},
    "InternVL3-78B-Instruct": {"gpu_requirement": 4, "hf_path": "OpenGVLab/InternVL3-78B-Instruct", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "instruct", "quantization": False},
    # "InternVL3-1B-Pretrained": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3-1B-Pretrained", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "base", "quantization": False},
    # "InternVL3-2B-Pretrained": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3-2B-Pretrained", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "base", "quantization": False},
    # "InternVL3-8B-Pretrained": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3-8B-Pretrained", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "base", "quantization": False},
    # "InternVL3-9B-Pretrained": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3-9B-Pretrained", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "base", "quantization": False},
    # "InternVL3-14B-Pretrained": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3-14B-Pretrained", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "base", "quantization": False},
    # "InternVL3-38B-Pretrained": {"gpu_requirement": 2, "hf_path": "OpenGVLab/InternVL3-38B-Pretrained", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "base", "quantization": False},
    # "InternVL3-78B-Pretrained": {"gpu_requirement": 4, "hf_path": "OpenGVLab/InternVL3-78B-Pretrained", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "internvl3", "model_type": "base", "quantization": False},
    
    # # InternVL3.5 series - NO SANITY_CHECK
    # "InternVL3_5-1B": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3_5-1B", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "reasoning", "quantization": False},
    # "InternVL3_5-2B": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3_5-2B", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "reasoning", "quantization": False},
    # "InternVL3_5-4B": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3_5-4B", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "reasoning", "quantization": False},
    # "InternVL3_5-8B": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3_5-8B", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "reasoning", "quantization": False},
    # "InternVL3_5-14B": {"gpu_requirement": 4, "hf_path": "OpenGVLab/InternVL3_5-14B", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "reasoning", "quantization": False},
    # "InternVL3_5-GPT-OSS-20B-A4B-Preview": {"gpu_requirement": 4, "hf_path": "OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "reasoning", "quantization": False},
    # "InternVL3_5-30B-A3B": {"gpu_requirement": 4, "hf_path": "OpenGVLab/InternVL3_5-30B-A3B", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "reasoning", "quantization": False},
    # "InternVL3_5-38B": {"gpu_requirement": 4, "hf_path": "OpenGVLab/InternVL3_5-38B", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "reasoning", "quantization": False},
    # # "InternVL3_5-241B-A28B": {"gpu_requirement": 8, "hf_path": "OpenGVLab/InternVL3_5-241B-A28B", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "reasoning", "quantization": False},
    # "InternVL3_5-1B-MPO": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3_5-1B-MPO", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "reasoning", "quantization": False},
    # "InternVL3_5-2B-MPO": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3_5-2B-MPO", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "reasoning", "quantization": False},
    # "InternVL3_5-4B-MPO": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3_5-4B-MPO", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "reasoning", "quantization": False},
    # "InternVL3_5-8B-MPO": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3_5-8B-MPO", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "reasoning", "quantization": False},
    # "InternVL3_5-14B-MPO": {"gpu_requirement": 4, "hf_path": "OpenGVLab/InternVL3_5-14B-MPO", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "reasoning", "quantization": False},
    # "InternVL3_5-30B-A3B-MPO": {"gpu_requirement": 4, "hf_path": "OpenGVLab/InternVL3_5-30B-A3B-MPO", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "reasoning", "quantization": False},
    # "InternVL3_5-38B-MPO": {"gpu_requirement": 4, "hf_path": "OpenGVLab/InternVL3_5-38B-MPO", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "reasoning", "quantization": False},
    # # "InternVL3_5-241B-A28B-MPO": {"gpu_requirement": 8, "hf_path": "OpenGVLab/InternVL3_5-241B-A28B-MPO", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "reasoning", "quantization": False},
    # "InternVL3_5-1B-Pretrained": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3_5-1B-Pretrained", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "base", "quantization": False},
    # "InternVL3_5-2B-Pretrained": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3_5-2B-Pretrained", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "base", "quantization": False},
    # "InternVL3_5-4B-Pretrained": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3_5-4B-Pretrained", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "base", "quantization": False},
    # "InternVL3_5-8B-Pretrained": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3_5-8B-Pretrained", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "base", "quantization": False},
    # "InternVL3_5-14B-Pretrained": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3_5-14B-Pretrained", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "base", "quantization": False},
    # "InternVL3_5-30B-A3B-Pretrained": {"gpu_requirement": 2, "hf_path": "OpenGVLab/InternVL3_5-30B-A3B-Pretrained", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "base", "quantization": False},
    # "InternVL3_5-38B-Pretrained": {"gpu_requirement": 2, "hf_path": "OpenGVLab/InternVL3_5-38B-Pretrained", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "base", "quantization": False},
    # # "InternVL3_5-241B-A28B-Pretrained": {"gpu_requirement": 8, "hf_path": "OpenGVLab/InternVL3_5-241B-A28B-Pretrained", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "base", "quantization": False},
    "InternVL3_5-1B-Instruct": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3_5-1B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "instruct", "quantization": False},
    "InternVL3_5-2B-Instruct": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3_5-2B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "instruct", "quantization": False},
    "InternVL3_5-4B-Instruct": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3_5-4B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "instruct", "quantization": False},
    "InternVL3_5-8B-Instruct": {"gpu_requirement": 1, "hf_path": "OpenGVLab/InternVL3_5-8B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "instruct", "quantization": False},
    "InternVL3_5-14B-Instruct": {"gpu_requirement": 4, "hf_path": "OpenGVLab/InternVL3_5-14B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "instruct", "quantization": False},
    "InternVL3_5-30B-A3B-Instruct": {"gpu_requirement": 4, "hf_path": "OpenGVLab/InternVL3_5-30B-A3B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "instruct", "quantization": False},
    "InternVL3_5-38B-Instruct": {"gpu_requirement": 4, "hf_path": "OpenGVLab/InternVL3_5-38B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "instruct", "quantization": False},
    #"InternVL3_5-241B-A28B-Instruct": {"gpu_requirement": 8, "hf_path": "OpenGVLab/InternVL3_5-241B-A28B-Instruct", "sanity_check": False, "modalities": ["multi_image_video_interleave"], "series": "internvl3d5", "model_type": "instruct", "quantization": False},

    # LLaVA OneVision series - SANITY_CHECK
    # "llava-onevision-qwen2-0.5b-si-hf": {"gpu_requirement": 1, "hf_path": "llava-hf/llava-onevision-qwen2-0.5b-si-hf", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "llava_ov", "model_type": "instruct", "quantization": False},
    # "llava-onevision-qwen2-7b-si-hf": {"gpu_requirement": 1, "hf_path": "llava-hf/llava-onevision-qwen2-7b-si-hf", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "llava_ov", "model_type": "instruct", "quantization": False},
    # "llava-onevision-qwen2-72b-si-hf": {"gpu_requirement": 8, "hf_path": "llava-hf/llava-onevision-qwen2-72b-si-hf", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "llava_ov", "model_type": "instruct", "quantization": False},
    # "llava-onevision-qwen2-0.5b-ov-hf": {"gpu_requirement": 1, "hf_path": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "llava_ov", "model_type": "instruct", "quantization": False},
    # "llava-onevision-qwen2-7b-ov-hf": {"gpu_requirement": 1, "hf_path": "llava-hf/llava-onevision-qwen2-7b-ov-hf", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "llava_ov", "model_type": "instruct", "quantization": False},
    # "llava-onevision-qwen2-72b-ov-hf": {"gpu_requirement": 8, "hf_path": "llava-hf/llava-onevision-qwen2-72b-ov-hf", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "llava_ov", "model_type": "instruct", "quantization": False},
    # # "llava-onevision-qwen2-7b-ov-chat-hf": {"gpu_requirement": 1, "hf_path": "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "llava_ov", "model_type": "instruct", "quantization": False},
    # "llava-onevision-qwen2-72b-ov-chat-hf": {"gpu_requirement": 8, "hf_path": "llava-hf/llava-onevision-qwen2-72b-ov-chat-hf", "sanity_check": True, "modalities": ["multi_image_video_interleave"], "series": "llava_ov", "model_type": "instruct", "quantization": False},
}


def get_available_gpu_count() -> int:
    """Get the number of available GPUs, respecting CUDA_VISIBLE_DEVICES."""
    # First check CUDA_VISIBLE_DEVICES - this takes priority
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible:
        # Count comma-separated GPU IDs
        gpu_ids = [gpu.strip() for gpu in cuda_visible.split(',') if gpu.strip()]
        return len(gpu_ids)
    
    # If CUDA_VISIBLE_DEVICES is not set, use nvidia-smi to count all GPUs
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                              capture_output=True, text=True, check=True)
        gpu_lines = [line for line in result.stdout.strip().split('\n') if line.strip()]
        return len(gpu_lines)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Default fallback
        return 1


def calculate_gpu_allocation(available_gpus: int, gpu_requirement: int) -> Dict:
    """Calculate optimal GPU allocation for model inference.
    
    Args:
        available_gpus: Number of available GPUs
        gpu_requirement: Number of GPUs required per model instance (80GB VRAM each)
        
    Returns:
        Dictionary with allocation details:
        - can_run: bool, whether model can run
        - parallel_tasks: int, number of parallel tasks
        - gpus_per_task: int, GPUs allocated per task
        - warning: str, warning message if any
    """
    if gpu_requirement <= 0:
        return {
            "can_run": True,
            "parallel_tasks": 1,
            "gpus_per_task": 1,
            "warning": None
        }
    
    # Case 1: Not enough GPUs - skip model with warning
    if available_gpus < gpu_requirement:
        return {
            "can_run": False,
            "parallel_tasks": 0,
            "gpus_per_task": 0,
            "warning": f"Insufficient GPUs: need {gpu_requirement}, have {available_gpus}"
        }
    
    # Case 2: Enough for exactly one instance but not parallel
    if available_gpus < gpu_requirement * 2:
        return {
            "can_run": True,
            "parallel_tasks": 1,
            "gpus_per_task": available_gpus,  # Use all available GPUs
            "warning": None
        }
    
    # Case 3: Enough for parallel inference
    parallel_tasks = available_gpus // gpu_requirement
    remainder_gpus = available_gpus % gpu_requirement
    
    if remainder_gpus == 0:
        # Perfect division - no remainder
        gpus_per_task = gpu_requirement
    else:
        # Distribute remainder evenly across tasks
        extra_per_task = remainder_gpus // parallel_tasks
        gpus_per_task = gpu_requirement + extra_per_task
        
        # If there's still remainder after even distribution, we keep the base allocation
        # The mmeval script will handle the actual GPU assignment
        if remainder_gpus % parallel_tasks > 0:
            # For simplicity, we'll use base + average extra
            gpus_per_task = gpu_requirement + (remainder_gpus // parallel_tasks)


    return {
        "can_run": True,
        "parallel_tasks": parallel_tasks,
        "gpus_per_task": gpus_per_task,
        "warning": None
}


class BatchInferenceRunner:
    def __init__(self, model_type: str = "instruct", model_quantization: str = "non-quantization"):
        self.model_type = model_type
        self.model_quantization = model_quantization
        self.base_dir = Path(__file__).parent
        self.available_gpus = get_available_gpu_count()
        self.results = []
        
        print(f"Batch inference initialized - Available GPUs: {self.available_gpus}")
        
    def log_and_print(self, message: str, level: str = "INFO"):
        """Log message to console."""
        if level.upper() == "ERROR":
            print(f"ERROR: {message}")
        elif level.upper() == "WARNING":
            print(f"WARNING: {message}")
        else:
            print(message)
        
    def get_modality_paths(self, dataset: str, modality: str, prompt: str) -> tuple[str, str]:
        """Get infile and img_dir paths based on modality mapping for any dataset."""
        # Generic modality to path mapping
        modality_mapping = {
            "single_image_start": ("frame", "single_image"),
            "single_video_start": ("video", "single_video"),
            "multi_video_interleave": ("video", "total"),
            "multi_image_video_interleave": ("video", "total"),
            "multi_image_interleave": ("frame", "total"),
            "multi_image_start": ("frame", "total"),
        }
        
        if modality not in modality_mapping:
            raise ValueError(f"Unsupported modality: {modality}")
            
        subdir_type, file_type = modality_mapping[modality]
        infile_path = f"/data/ztw/data/{dataset}/{dataset}_{subdir_type}/{prompt}/{dataset}_{file_type}_{prompt}.json"
        img_dir_path = f"/data/ztw/data/{dataset}/{dataset}_media"
        
        return infile_path, img_dir_path
        
    def get_model_series(self, model_name: str) -> Optional[str]:
        """Get model series from MODEL_CONFIGS."""
        if model_name in MODEL_CONFIGS:
            return MODEL_CONFIGS[model_name].get("series")
        return None
    
    def is_model_sanity_check(self, model_name: str) -> bool:
        """Check if model is in sanity check (not commented in registry)."""
        if model_name in MODEL_CONFIGS:
            return MODEL_CONFIGS[model_name].get("sanity_check", False)
        return False
    
    def get_model_modalities(self, model_name: str) -> List[str]:
        """Get supported modalities for a model."""
        if model_name in MODEL_CONFIGS:
            return MODEL_CONFIGS[model_name].get("modalities", [])
        return []
    
    def get_model_type(self, model_name: str) -> str:
        """Get model type (instruct, base, or reasoning)."""
        if model_name in MODEL_CONFIGS:
            return MODEL_CONFIGS[model_name].get("model_type", "instruct")
        return "instruct"
    
    def is_base_model(self, model_name: str) -> bool:
        """Check if model is a base model."""
        model_type = self.get_model_type(model_name)
        return model_type == "base"
    
    def is_quantized_model(self, model_name: str) -> bool:
        """Check if model is a quantized model."""
        if model_name in MODEL_CONFIGS:
            return MODEL_CONFIGS[model_name].get("quantization", False)
        return False
    
    def filter_models(self, 
                     series_filter: Optional[List[str]] = None,
                     modality_filter: Optional[str] = None,
                     model_status: str = "sanity_check",
                     model_type: str = "instruct",
                     model_quantization: str = "non-quantization") -> List[str]:
        """Filter models based on criteria.
        
        Args:
            series_filter: Filter by model series
            modality_filter: Filter by supported modality
            model_status: Which models to include - "sanity_check", "no_sanity_check", or "all"
            model_type: Which model types to include - "base", "instruct", "reasoning", "non-base", or "all"
            model_quantization: Filter by quantization - "quantization", "non-quantization", or "all"
            
        Returns:
            List of filtered model names
        """
        filtered_models = []
        
        for model_name, config in MODEL_CONFIGS.items():
            # Filter by model status (sanity_check/no_sanity_check/all)
            is_sanity_check = self.is_model_sanity_check(model_name)
            if model_status == "sanity_check" and not is_sanity_check:
                continue
            elif model_status == "no_sanity_check" and is_sanity_check:
                continue
            # If model_status == "all", include both sanity_check and no_sanity_check models
            
            # Filter by model type
            current_model_type = self.get_model_type(model_name)
            if model_type == "base" and current_model_type != "base":
                continue
            elif model_type == "instruct" and current_model_type != "instruct":
                continue
            elif model_type == "reasoning" and current_model_type != "reasoning":
                continue
            elif model_type == "non-base" and current_model_type == "base":
                continue
            # If model_type == "all", include all model types
                
            # Filter by quantization
            is_quantized = self.is_quantized_model(model_name)
            if model_quantization == "quantization" and not is_quantized:
                continue
            elif model_quantization == "non-quantization" and is_quantized:
                continue
            # If model_quantization == "all", include both quantized and non-quantized models
                
            if series_filter:
                model_series = self.get_model_series(model_name)
                if model_series not in series_filter:
                    continue
            
            if modality_filter:
                model_modalities = self.get_model_modalities(model_name)
                if modality_filter not in model_modalities:
                    continue
                
            filtered_models.append(model_name)
        
        return filtered_models
    
    def get_command_args(self, model_name: str, mode: str, modality: str, prompt: str, gpu_allocation: Dict) -> List[str]:
        """Generate command arguments for a model."""
        config = MODEL_CONFIGS[model_name]
        out_dir = self.base_dir / "work_dirs" / mode / f"{model_name}-{mode.replace('_', '-')}"
        
        python_exe = "/data/ztw/mmeval_envs/test/bin/python"
        args = [
            python_exe, "mmeval/run.py",
            "--out_dir", str(out_dir),
            "--model_name_or_path", config["hf_path"],
            "--gpu_per_parallel", str(gpu_allocation["gpus_per_task"]),
            "--parallel_per_task", str(gpu_allocation["parallel_tasks"]),
            "--max_new_tokens", "512"
        ]
        
        if mode == "sanity_check":
            args.extend([
                "--dataset", "mmeval_hf@mm-eval/MMBench-en",
                "--split", "test"
            ])
        elif mode == "no_media":
            # Special case for no_media test - use the test_bed/modality_test/task/no_media.json file
            infile_path = str(self.base_dir / "test_bed" / "modality_test" / "task" / "no_media.json")
            args.extend([
                "--infile", infile_path,
                "--dataset", "local@json"
                # Note: no --img_dir needed for text-only tasks
            ])
        else:
            # For any dataset mode (69_Control, conservation_text, etc.)
            infile_path, img_dir_path = self.get_modality_paths(mode, modality, prompt)
            args.extend([
                "--infile", infile_path,
                "--dataset", "local@json",
                "--img_dir", img_dir_path
            ])
        
        return args
    
    def run_model(self, model_name: str, mode: str, modality: str, prompt: str) -> Optional[Dict]:
        """Run inference for a single model with specific configuration."""
        config = MODEL_CONFIGS[model_name]
        
        # Calculate GPU allocation
        gpu_allocation = calculate_gpu_allocation(self.available_gpus, config["gpu_requirement"])
        
        # Check if model can run
        if not gpu_allocation["can_run"]:
            self.log_and_print(f"⚠️  Skipping {model_name}: {gpu_allocation['warning']}", "WARNING")
            return None
        
        # Check if result file already exists and is valid
        result_file = self.base_dir / "work_dirs" / mode / f"{model_name}-{mode.replace('_', '-')}" / "result.json"
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    existing_results = json.load(f)
                # Skip if file exists and is not empty (i.e., not just [])
                if existing_results and existing_results != []:
                    self.log_and_print(f"⏭️  Skipping {model_name}: valid results already exist", "INFO")
                    return {
                        "model_name": model_name,
                        "hf_path": config["hf_path"],
                        "mode": mode,
                        "modality": modality,
                        "prompt": prompt,
                        "status": "skipped",
                        "error_message": "Valid results already exist",
                        "gpu_requirement": config["gpu_requirement"]
                    }
            except (json.JSONDecodeError, Exception) as e:
                # If we can't read the file, continue with inference
                self.log_and_print(f"⚠️  Could not read existing result file: {e}. Re-running inference.", "WARNING")
        
        self.log_and_print(f"Starting inference for {model_name} (mode: {mode}, modality: {modality}, prompt: {prompt})...")
        self.log_and_print(f"  GPU allocation: {gpu_allocation['gpus_per_task']} GPUs per task × {gpu_allocation['parallel_tasks']} parallel tasks")
        
        start_time = datetime.now()
        result = {
            "model_name": model_name,
            "hf_path": config["hf_path"],
            "mode": mode,
            "modality": modality,
            "prompt": prompt,
            "start_time": start_time.isoformat(),
            "status": "unknown",
            "error_message": None,
            "gpu_requirement": config["gpu_requirement"],
            "parallel_tasks": gpu_allocation["parallel_tasks"],
            "gpus_per_task": gpu_allocation["gpus_per_task"]
        }
        
        try:
            # Ensure work directory exists
            work_dir = self.base_dir / "work_dirs" / mode
            work_dir.mkdir(parents=True, exist_ok=True)
            
            args = self.get_command_args(model_name, mode, modality, prompt, gpu_allocation)
            self.log_and_print(f"Command: {' '.join(args)}")
            
            # Set up environment with flexible variable handling
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.base_dir) + ":" + env.get('PYTHONPATH', '')
            
            # Add environment variables with defaults if not already set
            env.setdefault('HF_TOKEN', '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')
            env.setdefault('HF_HOME', '/data/.cache/hf_home')
            env.setdefault('ENV_DIR', '/data/ztw/mmeval_envs')
            
            # Run process with real-time output
            process = subprocess.Popen(args, cwd=self.base_dir, stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT, text=True, env=env, 
                                     bufsize=1, universal_newlines=True)
            
            stdout_lines = []
            self.log_and_print(f"📋 === mmeval output for {model_name} ===")
            
            # Read output line by line and display in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    stdout_lines.append(line)
                    # Display real-time output with prefix
                    print(f"[mmeval] {line}")
            
            process.wait()
            self.log_and_print(f"📋 === End mmeval output for {model_name} ===")
            
            result["status"] = "success" if process.returncode == 0 else "failed"
            result["stdout"] = '\n'.join(stdout_lines)
            
            if process.returncode != 0:
                result["error_message"] = f"Process exited with code {process.returncode}"
                self.log_and_print(f"❌ {model_name} failed", "ERROR")
            else:
                # Check if results were generated
                result_file = self.base_dir / "work_dirs" / mode / f"{model_name}-{mode.replace('_', '-')}" / "result.json"
                if result_file.exists():
                    self.log_and_print(f"✅ {model_name} completed successfully")
                else:
                    result["status"] = "failed"
                    result["error_message"] = "No result.json file found"
                    self.log_and_print(f"⚠️  {model_name} completed but no results found", "WARNING")
                
        except Exception as e:
            result["status"] = "error"
            result["error_message"] = str(e)
            self.log_and_print(f"💥 {model_name} error: {str(e)}", "ERROR")
        
        finally:
            end_time = datetime.now()
            result["end_time"] = end_time.isoformat()
            result["duration_seconds"] = (end_time - start_time).total_seconds()
        
        return result
    
    def run_combinations(self, modes: List[str], user_modalities: List[str], prompts: List[str], models: List[str]) -> None:
        """Run batch inference for models, each using only its supported modalities."""
        from itertools import product
        
        # Set up logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.base_dir / f"batch_log_{timestamp}.json"
        
        # Generate model-specific tasks
        tasks = []
        for mode in modes:
            for prompt in prompts:
                for model_name in models:
                    if model_name not in MODEL_CONFIGS:
                        self.log_and_print(f"⚠️  Unknown model: {model_name}", "WARNING")
                        continue
                        
                    model_config = MODEL_CONFIGS[model_name]
                    supported_modalities = model_config.get("modalities", [])
                    
                    if mode in ["sanity_check", "no_media"]:
                        # For sanity_check and no_media modes, modality is not relevant
                        tasks.append((mode, "multi_image_interleave", prompt, model_name))
                    else:
                        # For dataset modes, find intersection of user modalities and model supported modalities
                        if user_modalities:
                            # User specified modalities - use intersection with model's supported modalities
                            compatible_modalities = list(set(user_modalities) & set(supported_modalities))
                        else:
                            # No user modalities specified - use all model's supported modalities
                            compatible_modalities = supported_modalities
                        
                        # Add tasks for each compatible modality
                        for modality in compatible_modalities:
                            tasks.append((mode, modality, prompt, model_name))
        
        total_tasks = len(tasks)
        unique_models = len(set(model for _, _, _, model in tasks))
        
        self.log_and_print(f"Starting batch inference:")
        self.log_and_print(f"  Modes: {modes}")
        self.log_and_print(f"  User modalities: {user_modalities if user_modalities else 'Auto (model-specific)'}")
        self.log_and_print(f"  Prompts: {prompts}")
        self.log_and_print(f"  Models: {unique_models} models")
        self.log_and_print(f"  Total tasks: {total_tasks}")
        
        completed_tasks = 0
        skipped_tasks = 0
        
        for task_idx, (mode, modality, prompt, model_name) in enumerate(tasks, 1):
            self.log_and_print(f"\n[Task {task_idx}/{total_tasks}] {model_name} (mode: {mode}, modality: {modality}, prompt: {prompt})")
            
            result = self.run_model(model_name, mode, modality, prompt)
            if result is not None:
                self.results.append(result)
                completed_tasks += 1
            else:
                skipped_tasks += 1
            
            # Save results after each task
            self.save_results(log_file)
        
        self.print_summary(completed_tasks, skipped_tasks, total_tasks)
    
    def save_results(self, log_file: Path) -> None:
        """Save results to log file."""
        with open(log_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": self.results
            }, f, indent=2)
    
    def print_summary(self, completed_tasks: int, skipped_tasks: int, total_tasks: int) -> None:
        """Print execution summary."""
        if completed_tasks == 0 and skipped_tasks == 0:
            print("No tasks were processed.")
            return
            
        success = sum(1 for r in self.results if r["status"] == "success")
        failed = sum(1 for r in self.results if r["status"] == "failed")
        error = sum(1 for r in self.results if r["status"] == "error")
        total_duration = sum(r.get("duration_seconds", 0) for r in self.results)
        
        print(f"\n{'='*50}")
        print(f"BATCH INFERENCE SUMMARY")
        print(f"{'='*50}")
        print(f"Total tasks planned: {total_tasks}")
        print(f"⚠️  Skipped (insufficient GPU): {skipped_tasks}")
        print(f"🔄 Completed: {completed_tasks}")
        print(f"✅ Success: {success}")
        print(f"❌ Failed: {failed}")
        print(f"💥 Error: {error}")
        
        if completed_tasks > 0:
            print(f"📊 Success rate: {success/completed_tasks*100:.1f}%")
        
        print(f"⏱️  Total duration: {total_duration:.1f}s ({total_duration/60:.1f}m)")
        
        if failed > 0 or error > 0:
            print(f"\n❌ Failed tasks:")
            for result in self.results:
                if result["status"] != "success":
                    duration = result.get("duration_seconds", 0)
                    error_msg = result.get("error_message", "Unknown error")
                    config = f"{result['mode']}/{result['modality']}/{result['prompt']}"
                    print(f"  - {result['model_name']} ({config}): {result['status']} ({duration:.1f}s)")
                    if error_msg and len(error_msg) < 200:
                        print(f"    Error: {error_msg}")
        
        print(f"\n🎉 Batch inference completed!")


def main():
    parser = argparse.ArgumentParser(description="Batch inference runner for all models")
    parser.add_argument("--mode", nargs="+", default=["sanity_check"], 
                       help="Inference modes (sanity_check, no_media, or dataset names like 69_Control, conservation_text)")
    parser.add_argument("--modality", nargs="*",
                       choices=["single_image_start", "single_video_start", "multi_video_interleave", 
                               "multi_image_video_interleave", "multi_image_interleave", "multi_image_start"], 
                       help="Modality types - automatically filters models to only run those supporting these modalities")
    parser.add_argument("--prompt", nargs="+", default=["p0"], 
                       help="Prompt types (p0, p1, p2, etc.)")
    parser.add_argument("--series", nargs="+", help="Filter by model series")
    parser.add_argument("--model-status", choices=["sanity_check", "no_sanity_check", "all"], default="sanity_check",
                       help="Which models to run: 'sanity_check' (default), 'no_sanity_check', or 'all'")
    parser.add_argument("--model-type", choices=["base", "instruct", "reasoning", "non-base", "all"], default="instruct",
                       help="Filter by model type: 'base' (only base), 'instruct' (only instruct), 'reasoning' (only reasoning), 'non-base' (all non-base models), 'all' (all model types)")
    parser.add_argument("--model-quantization", choices=["quantization", "non-quantization", "all"], default="non-quantization",
                       help="Filter by quantization: 'quantization' (only quantized), 'non-quantization' (only non-quantized), 'all' (all models)")
    parser.add_argument("--list-model", action="store_true", 
                       help="List available models and exit")
    parser.add_argument("--model", nargs="*", help="Specific models to run")
    parser.add_argument("--reverse", action="store_true",
                       help="Reverse the order of model execution")
    
    args = parser.parse_args()
    
    runner = BatchInferenceRunner(args.model_type, args.model_quantization)
    
    # Handle modalities - simplified logic
    modalities = args.modality if args.modality else []
    
    if args.list_model:
        print("Available models:")
        
        # Get models to list
        if args.model:
            filtered_models = args.model
        else:
            # For listing, use first modality for filtering if multiple specified
            modality_filter = modalities[0] if len(modalities) == 1 else None
            filtered_models = runner.filter_models(
                series_filter=args.series,
                modality_filter=modality_filter,
                model_status=args.model_status,
                model_type=args.model_type,
                model_quantization=args.model_quantization
            )
        
        # Group filtered models by series
        series_models = {}
        for model in filtered_models:
            model_series = runner.get_model_series(model)
            if model_series:
                if model_series not in series_models:
                    series_models[model_series] = []
                series_models[model_series].append(model)
            else:
                # Handle disabled models
                if "disabled" not in series_models:
                    series_models["disabled"] = []
                series_models["disabled"].append(model)
        
        for series, models in series_models.items():
            print(f"\n{series}:")
            for model in models:
                config = MODEL_CONFIGS.get(model, {})
                status = "✅" if runner.is_model_sanity_check(model) else "❌"
                gpu_requirement = config.get("gpu_requirement", "?")
                modalities_list = config.get("modalities", [])
                model_series = config.get("series", "unknown")
                model_type = config.get("model_type", "unknown")
                is_quantized = config.get("quantization", False)
                modality_str = ", ".join(modalities_list) if modalities_list else "unknown"
                quant_str = "Quantized" if is_quantized else "Standard"
                print(f"  {status} {model} (GPU: {gpu_requirement}, Type: {model_type}, {quant_str}, Series: {model_series}, Modalities: {modality_str})")
        return
    
    # Get models to run
    if args.model:
        models_to_run = args.model
    else:
        # For filtering, don't use modality filter to get all compatible models
        models_to_run = runner.filter_models(
            series_filter=args.series,
            modality_filter=None,  # Let run_combinations handle modality filtering
            model_status=args.model_status,
            model_type=args.model_type,
            model_quantization=args.model_quantization
        )
    
    if not models_to_run:
        print("No models match the specified criteria.")
        return
    
    # Reverse model order if requested
    if args.reverse:
        models_to_run = list(reversed(models_to_run))
    
    runner.run_combinations(args.mode, modalities, args.prompt, models_to_run)


if __name__ == "__main__":
    main()
