"""GPU discovery and VRAM-aware allocation helpers for batch inference.

These helpers let ``mmeval/batch_infer.py`` decide, for each model, how many
GPUs a single inference shard needs and how many shards can run in parallel,
based on an estimate of the model's peak VRAM usage.
"""

import math
import os
import re
import subprocess

# Bytes per parameter for common weight dtypes (KV cache is always fp16/bf16).
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "int8": 1, "int4": 0.5, "fp8": 1}

DEFAULT_IMAGE_TOKENS = 1024
DEFAULT_TEXT_TOKENS = 2048
DEFAULT_MAX_CONTEXT = 32768
VRAM_OVERHEAD_GB = 3.0          # activations, fragmentation, runtime overhead
VRAM_FALLBACK_GB_PER_GPU = 40.0  # used when parameter_count is unknown
VRAM_USABLE_FRACTION = 0.9       # leave headroom on each GPU
DETECTED_VRAM_FALLBACK_GB = 80.0  # assumed per-GPU VRAM when detection fails


def _get_visible_devices() -> str:
    """Return CUDA_VISIBLE_DEVICES, falling back to NVIDIA_VISIBLE_DEVICES."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"].strip()
    return os.environ.get("NVIDIA_VISIBLE_DEVICES", "").strip()


def _visibility_explicitly_disabled() -> bool:
    """True when CUDA_VISIBLE_DEVICES is set to empty (CUDA = disable all GPUs)."""
    return "CUDA_VISIBLE_DEVICES" in os.environ and not os.environ["CUDA_VISIBLE_DEVICES"].strip()


def _nvidia_smi_vram(*extra_args: str) -> float | None:
    """Query nvidia-smi for the minimum per-GPU VRAM in GiB. None on failure."""
    try:
        out = subprocess.run(
            ["nvidia-smi", *extra_args,
             "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            vals = [int(v) for v in out.stdout.strip().split("\n") if v.strip()]
            if vals:
                return min(vals) / 1024  # MiB -> GiB
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _parse_mig_vram_gb() -> float | None:
    """Parse the smallest MIG slice VRAM from nvidia-smi -L profiles (e.g. '1g.10gb')."""
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            sizes = re.findall(r"\d+g\.(\d+)gb", out.stdout, re.IGNORECASE)
            if sizes:
                return float(min(int(s) for s in sizes))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_gpu_indices() -> list[str]:
    """Return the visible GPU tokens as strings (respects CUDA_VISIBLE_DEVICES)."""
    if _visibility_explicitly_disabled():
        return []
    cvd = _get_visible_devices()
    if cvd and cvd.lower() != "all":
        # Opaque tokens: numeric ids and GPU/MIG UUIDs are both supported.
        return [x.strip() for x in cvd.split(",") if x.strip()]

    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return []
        lines = [ln.strip() for ln in out.stdout.strip().splitlines() if ln.strip()]
        # MIG mode: CUDA sees MIG instances, not parent GPUs.
        mig_lines = [ln for ln in lines if "MIG" in ln and "Device" in ln]
        if mig_lines:
            return [str(i) for i in range(len(mig_lines))]
        gpu_count = len([ln for ln in lines if ln.startswith("GPU") and ":" in ln])
        return [str(i) for i in range(gpu_count)]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def get_gpu_vram_gb() -> float:
    """Detect the minimum per-GPU VRAM in GiB, falling back to a safe default."""
    if _visibility_explicitly_disabled():
        return 0.0
    cvd = _get_visible_devices()
    if cvd and cvd.lower() != "all":
        cvd_norm = ",".join(x.strip() for x in cvd.split(",") if x.strip())
        detected = _nvidia_smi_vram("-i", cvd_norm)
        if detected is not None:
            return detected
        # -i can fail for MIG UUIDs; try MIG profiles, then the safe fallback.
        return _parse_mig_vram_gb() or DETECTED_VRAM_FALLBACK_GB
    return _nvidia_smi_vram() or DETECTED_VRAM_FALLBACK_GB


def estimate_max_tokens(model: dict, num_images: int = 1) -> int:
    """Estimate the sequence length (image + text tokens) for VRAM sizing."""
    image_tokens = int(model.get("image_tokens") or DEFAULT_IMAGE_TOKENS)
    max_dynamic_patch = int(model.get("max_dynamic_patch") or 1)
    total = image_tokens * max_dynamic_patch * num_images + DEFAULT_TEXT_TOKENS
    max_ctx = int(model.get("max_position_embeddings") or DEFAULT_MAX_CONTEXT)
    return min(total, max_ctx)


def estimate_vram_gb(model: dict, max_tokens: int, batch_size: int = 1) -> float:
    """Estimate peak VRAM (GiB) for one inference shard of ``model``."""
    params = int(model.get("parameter_count") or 0)
    if params == 0:
        # Architecture unknown: fall back to the curated GPU count.
        required = model.get("required_gpu_count")
        return (float(required) if str(required).isdigit() else 1.0) * VRAM_FALLBACK_GB_PER_GPU

    weight_bytes = DTYPE_BYTES.get(model.get("torch_dtype") or "bfloat16", 2)
    weights_gb = params * weight_bytes / (1024 ** 3)

    # KV cache (GQA-aware), always fp16/bf16 regardless of weight quantization.
    num_layers = int(model.get("num_hidden_layers") or 32)
    hidden_size = int(model.get("hidden_size") or 4096)
    num_attn_heads = int(model.get("num_attention_heads") or 32)
    num_kv_heads = int(model.get("num_key_value_heads") or num_attn_heads)
    head_dim = -(-hidden_size // num_attn_heads) if num_attn_heads > 0 else hidden_size
    kv_cache_gb = 2 * batch_size * num_layers * (num_kv_heads * head_dim) * max_tokens * 2 / (1024 ** 3)

    return weights_gb + kv_cache_gb + VRAM_OVERHEAD_GB


def parse_gpu_memory(value: list[float] | None, visible_gpu_count: int) -> list[float]:
    """Resolve ``--gpu-memory`` into a per-visible-GPU VRAM list (GiB)."""
    if visible_gpu_count <= 0:
        return []
    if value is None:
        return [get_gpu_vram_gb()] * visible_gpu_count

    parsed = [float(v) for v in value]
    if any(v < 0 for v in parsed):
        raise ValueError("--gpu-memory values must be >= 0")
    if len(parsed) < visible_gpu_count:
        parsed.extend([0.0] * (visible_gpu_count - len(parsed)))
    elif len(parsed) > visible_gpu_count:
        print(f"Warning: --gpu-memory has {len(parsed)} values but only "
              f"{visible_gpu_count} GPU(s) visible; dropping extras: {parsed[visible_gpu_count:]}")
    return parsed[:visible_gpu_count]


def min_gpu_vram(gpu_memory: list[float]) -> float:
    """Minimum VRAM among enabled GPUs (those with VRAM > 0); 0.0 if none."""
    enabled = [v for v in gpu_memory if v > 0]
    return min(enabled) if enabled else 0.0


def enabled_gpu_indices(visible: list[str], gpu_memory: list[float]) -> list[str]:
    """Visible GPU tokens whose configured VRAM is > 0."""
    return [idx for idx, vram in zip(visible, gpu_memory) if vram > 0]


def gpu_allocation(available_gpus: int, estimated_vram_gb: float, per_gpu_vram_gb: float) -> dict:
    """Plan GPU usage for one model.

    Returns a dict with ``can_run`` plus, when runnable, ``gpu_per_parallel``
    (GPUs per shard) and ``parallel_per_task`` (concurrent shards). When not
    runnable it carries a human-readable ``reason``.
    """
    if available_gpus <= 0:
        return {"can_run": False, "reason": "no enabled GPU"}
    if per_gpu_vram_gb <= 0:
        return {"can_run": False, "reason": "all GPUs disabled (--gpu-memory)"}

    if estimated_vram_gb <= 0:
        return {"can_run": True, "gpu_per_parallel": 1, "parallel_per_task": available_gpus}

    gpus_needed = max(1, math.ceil(estimated_vram_gb / (per_gpu_vram_gb * VRAM_USABLE_FRACTION)))
    if gpus_needed > available_gpus:
        return {"can_run": False,
                "reason": f"need {gpus_needed} GPU(s) ({estimated_vram_gb:.1f}GiB), have {available_gpus}"}
    return {"can_run": True,
            "gpu_per_parallel": gpus_needed,
            "parallel_per_task": available_gpus // gpus_needed}
