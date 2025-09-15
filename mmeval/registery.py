import os

env_dir = os.getenv('ENV_DIR') or ""

series_mapping = {
    "qwenvl2": ["Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct", "Qwen2-VL-72B-Instruct",
                "Qwen2-VL-2B-Instruct-AWQ", "Qwen2-VL-7B-Instruct-AWQ", "Qwen2-VL-72B-Instruct-AWQ",
                "Qwen2-VL-2B-Instruct-GPTQ-Int4", "Qwen2-VL-7B-Instruct-GPTQ-Int4", "Qwen2-VL-72B-Instruct-GPTQ-Int4"],
    "qwenvl2d5": ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct",
                  "Qwen2.5-VL-3B-Instruct-AWQ", "Qwen2.5-VL-7B-Instruct-AWQ", "Qwen2.5-VL-32B-Instruct-AWQ", "Qwen2.5-VL-72B-Instruct-AWQ"],
    "qwenvl2d5_omni": ["Qwen2.5-Omni-3B", "Qwen2.5-Omni-7B", "Qwen2.5-Omni-7B-AWQ", "Qwen2.5-Omni-7B-GPTQ-Int4"],
    "gemma3": ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"],
    "llava": ["llava-1.5-7b-hf", "llava-1.5-13b-hf"],
    "llava_next": ["llava-v1.6-mistral-7b-hf", "llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf", "llava-v1.6-34b-hf", "llama3-llava-next-8b-hf", "llava-next-72b-hf", "llava-next-110b-hf"],
    "glm_4v": ["glm-4v-9b"],
    "ovis1d5": ["Ovis1.5-Llama3-8B", "Ovis1.5-Gemma2-9B"],
    "ovis1d6": ["Ovis1.6-Llama3.2-3B", "Ovis1.6-Gemma2-9B"],
    "ovis1d6_27b": ["Ovis1.6-Gemma2-27B"],
    "blip2_flan_t5": ["blip2-flan-t5-xl", "blip2-flan-t5-xxl"],
    "internvl_chat": ["InternVL-Chat-V1-1", "InternVL-Chat-V1-2", "InternVL-Chat-V1-2-Plus"],
    "internvl_chat1d5": ["Mini-InternVL-Chat-2B-V1-5", "Mini-InternVL-Chat-4B-V1-5", "InternVL-Chat-V1-5"],
    "internvl2": ["InternVL2-1B", "InternVL2-2B", "InternVL2-4B", "InternVL2-8B", "InternVL2-26B", "InternVL2-40B", "InternVL2-Llama3-76B"],
    "internvl2d5": ["InternVL2_5-1B", "InternVL2_5-2B", "InternVL2_5-4B", "InternVL2_5-8B", "InternVL2_5-26B", "InternVL2_5-38B", "InternVL2_5-78B",
                    "InternVL2_5-1B-MPO", "InternVL2_5-2B-MPO", "InternVL2_5-4B-MPO", "InternVL2_5-8B-MPO", "InternVL2_5-26B-MPO", "InternVL2_5-38B-MPO", "InternVL2_5-78B-MPO"],
    "internvl3": ["InternVL3-1B", "InternVL3-2B", "InternVL3-8B", "InternVL3-9B", "InternVL3-14B", "InternVL3-38B", "InternVL3-78B",
                  "InternVL3-1B-Instruct", "InternVL3-2B-Instruct", "InternVL3-8B-Instruct", "InternVL3-9B-Instruct", "InternVL3-14B-Instruct", "InternVL3-38B-Instruct", "InternVL3-78B-Instruct",
                  "InternVL3-1B-Pretrained", "InternVL3-2B-Pretrained", "InternVL3-8B-Pretrained", "InternVL3-9B-Pretrained", "InternVL3-14B-Pretrained", "InternVL3-38B-Pretrained", "InternVL3-78B-Pretrained"],
    "internvl3d5": ["InternVL3_5-1B", "InternVL3_5-2B", "InternVL3_5-4B", "InternVL3_5-8B", "InternVL3_5-14B", "InternVL3_5-GPT-OSS-20B-A4B-Preview", "InternVL3_5-30B-A3B", "InternVL3_5-38B", "InternVL3_5-241B-A28B",
                    "InternVL3_5-1B-MPO", "InternVL3_5-2B-MPO", "InternVL3_5-4B-MPO", "InternVL3_5-8B-MPO", "InternVL3_5-14B-MPO", "InternVL3_5-30B-A3B-MPO", "InternVL3_5-38B-MPO", "InternVL3_5-241B-A28B-MPO",
                    "InternVL3_5-1B-Pretrained", "InternVL3_5-2B-Pretrained", "InternVL3_5-4B-Pretrained", "InternVL3_5-8B-Pretrained", "InternVL3_5-14B-Pretrained", "InternVL3_5-30B-A3B-Pretrained", "InternVL3_5-38B-Pretrained", "InternVL3_5-241B-A28B-Pretrained",
                    "InternVL3_5-1B-Instruct", "InternVL3_5-2B-Instruct", "InternVL3_5-4B-Instruct", "InternVL3_5-8B-Instruct", "InternVL3_5-14B-Instruct", "InternVL3_5-30B-A3B-Instruct", "InternVL3_5-38B-Instruct", "InternVL3_5-241B-A28B-Instruct"],
    "moondream1": ["moondream1"],
    "moondream2": ["moondream2"],
    "internlm": ["internlm-xcomposer-7b" ]
}
series_infer_env_mapping = {
    "qwenvl2": {
        "env": os.path.join(env_dir, "qwenvl"),
        "infer_file": "qwenvl2.py",
    },
    "qwenvl2d5": {
        "env": os.path.join(env_dir, "qwenvl"),
        "infer_file": "qwenvl2d5.py",
    },
    "qwenvl2d5_omni": {
        "env": os.path.join(env_dir, "qwenvl"),
        "infer_file": "qwenvl2d5_omni.py",
    },
    "gemma3": {
        "env": os.path.join(env_dir, "gemma3"),
        "infer_file": "gemma3.py",
    },
    "llava": {
        "env": os.path.join(env_dir, "llava"),
        "infer_file": "llava.py",
    },
    "llava_next": {
        "env": os.path.join(env_dir, "llava_next"),
        "infer_file": "llava_next.py",
    },
    "glm_4v": {
        "env": os.path.join(env_dir, "glm_4v"),
        "infer_file": "glm_4v.py",
    },
    "ovis1d5": {
        "env": os.path.join(env_dir, "ovis1d5"),
        "infer_file": "ovis1d5.py",
    },
    "ovis1d6": {
        "env": os.path.join(env_dir, "ovis1d6"),
        "infer_file": "ovis1d6.py",
    },
    "ovis1d6_27b": {
        "env": os.path.join(env_dir, "ovis1d6_27b"),
        "infer_file": "ovis1d6.py",
    },
    "blip2_flan_t5": {
        "env": os.path.join(env_dir, "flan-t5"),
        "infer_file": "blip2_flan_t5.py",
    },
    "internvl_chat": {
        "env": os.path.join(env_dir, "internvl"),
        "infer_file": "internvl_chat.py",
    },
    "internvl_chat1d5": {
        "env": os.path.join(env_dir, "internvl"),
        "infer_file": "internvl_chat1d5.py",
    },
    "internvl2": {
        "env": os.path.join(env_dir, "internvl"),
        "infer_file": "internvl2.py",
    },
    "internvl2d5": {
        "env": os.path.join(env_dir, "internvl"),
        "infer_file": "internvl2d5.py",
    },
    "internvl3": {
        "env": os.path.join(env_dir, "internvl"),
        "infer_file": "internvl3.py",
    },
    "internvl3d5": {
        "env": os.path.join(env_dir, "internvl3d5"),
        "infer_file": "internvl3d5.py",
    },
    "moondream1": {
        "env": os.path.join(env_dir, "moondream1"),
        "infer_file": "moondream1.py",
    }, 
    "moondream2": {
        "env": os.path.join(env_dir, "moondream2"),
        "infer_file": "moondream2.py",
    },
    "internlm": {
        "env": os.path.join(env_dir, "internlm"),
        "infer_file": "internlm_xcomposer_7b.py",
    }
}
