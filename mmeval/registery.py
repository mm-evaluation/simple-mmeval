series_mapping = {
    "qwenvl2d5": ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct"],
    "idefics": [
        "idefics-80b-instruct",
        "idefics-9b-instruct",
        "Idefics2-8b",
        "Idefics3-8B-Llama3",
    ]
}

series_infer_env_mapping = {
    "qwenvl2d5": {
        "env": "vllm",
        "infer_file": "qwenvl2d5.py",
    },
    "idefics": {
        "env": "idefics",
        "infer_file": "idefics.py",
    }
}