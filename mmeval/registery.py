series_mapping = {
    "qwenvl2d5": ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct"],
    "aria": ["Aria"]
}

series_infer_env_mapping = {
    "qwenvl2d5": {
        "env": "/home/jovyan/shared/Yijiang-Li/envs/vllm",
        "infer_file": "qwenvl2d5.py",
    },
    "aria": {
        "env": "/u/boqiny2/miniconda3/envs/mmeval",
        "infer_file": "aria.py",
    }
}