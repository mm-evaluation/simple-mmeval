DATASET_MAP = {
    "default": {
        "modality": "image",
        "options": "options",
        "question": "question",
        "id": "id",
        "post-prompt": "None",
        "pre-prompt": "None",
        "options-prompt": "None",
    },
    "lmms-lab/MMBench_EN": {
        "options": ["A", "B", "C", "D"],
        "id": "index",
        "post-prompt": "\nAnswer with the option's letter from the given choices directly.",
        "pre-prompt": "hint",
        "options-prompt": " There are several options:",
    },
    "lmms-lab/MMBench_CN": {
        "options": ["A", "B", "C", "D"],
        "id": "index",
        "post-prompt": "\n请直接使用所提供的选项字母作为答案回答。",
        "pre-prompt": "hint",
        "options-prompt": " There are several options:",
    },
    "lmms-lab/MME": {
        "options": None,
        "id": "question_id",
        "post-prompt": "\nAnswer the question using a single word or phrase.",
    },
    "lmms-lab/ScienceQA-IMG": {
        "options": "choices",
        "id": None,
        "hint": "hint",
        "post-prompt": "\nAnswer with the option's letter from the given choices directly.",
    },
    "lmms-lab/textvqa": {
        "options": None,
        "id": "question_id",
        "post-prompt": "\nAnswer the question using a single word or phrase.",
    },
}