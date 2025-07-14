series_mapping = {
    "qwenvl2d5": ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct"],
    "blip2opt": ["Salesforce/blip2-opt-2.7b"]
}

series_infer_env_mapping = {
    "qwenvl2d5": {
        "env": "/home/jovyan/shared/Yijiang-Li/envs/vllm",
        "infer_file": "qwenvl2d5.py",
    },
    "blip2opt": {
        "env": "/home/jovyan/shared/Yijiang-Li/envs/blip2",
        "infer_file": "blip2opt.py",
    }
}