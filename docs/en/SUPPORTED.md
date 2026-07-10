# Supported Models and Datasets

For general usage, see the main [README](../../README.md).

---

- [Supported Models](#supported-models)
  - [Open Source Models](#open-source-models)
  - [API Models](#api-models)
- [Supported Datasets](#supported-datasets)
  - [Local JSON](#1-local-json-localjson)
  - [HuggingFace Datasets](#2-huggingface-datasets-mmeval_hf)
  - [VLMEvalKit Datasets](#3-vlmevalkit-datasets-evalkit)

---

## Supported Models

### Open Source Models

| Model Series | Supported Models | Inference Script |
|---|---|---|
| **BLIP2** | blip2-flan-t5-xl, blip2-flan-t5-xxl | `blip2.py` |
| **Bunny-Llama3** | Bunny-Llama-3-8B-V | `bunnyllama3.py` |
| **Cambrian** | cambrian-8b, cambrian-13b, cambrian-34b, cambrian-phi3-3b | `cambrian.py` |
| **Cosmos-Reason2** | Cosmos-Reason2-2B, Cosmos-Reason2-8B | `cosmos_reason2.py` |
| **Fuyu** | fuyu-8b | `fuyu.py` |
| **Gemma3** | gemma-3-4b-it, gemma-3-12b-it, gemma-3-27b-it | `gemma3.py` |
| **GLM-4V** | glm-4v-9b | `glm_4v.py` |
| **InstructBLIP** | instructblip-vicuna-7b, instructblip-vicuna-13b, instructblip-flan-t5-xl, instructblip-flan-t5-xxl | `instructblip.py` |
| **InternLM-XComposer** | internlm-xcomposer-7b | `internlm_xcomposer.py` |
| **InternVL-Chat** | InternVL-Chat-V1-1, InternVL-Chat-V1-2, InternVL-Chat-V1-2-Plus | `internvl_chat.py` |
| **InternVL-Chat-1.5** | Mini-InternVL-Chat-2B-V1-5, Mini-InternVL-Chat-4B-V1-5, InternVL-Chat-V1-5 | `internvl_chat1d5.py` |
| **InternVL2** | InternVL2-1B, InternVL2-2B, InternVL2-4B, InternVL2-8B, InternVL2-26B, InternVL2-40B, InternVL2-Llama3-76B | `internvl2.py` |
| **InternVL2.5** | InternVL2_5-1B, InternVL2_5-2B, InternVL2_5-4B, InternVL2_5-8B, InternVL2_5-26B, InternVL2_5-38B, InternVL2_5-78B, InternVL2_5-1B-MPO, InternVL2_5-2B-MPO, InternVL2_5-4B-MPO, InternVL2_5-8B-MPO, InternVL2_5-26B-MPO, InternVL2_5-38B-MPO, InternVL2_5-78B-MPO | `internvl2d5.py` |
| **InternVL3** | InternVL3-1B, InternVL3-2B, InternVL3-8B, InternVL3-9B, InternVL3-14B, InternVL3-38B, InternVL3-78B, InternVL3-1B-Instruct, InternVL3-2B-Instruct, InternVL3-8B-Instruct, InternVL3-9B-Instruct, InternVL3-14B-Instruct, InternVL3-38B-Instruct, InternVL3-78B-Instruct, InternVL3-1B-Pretrained, InternVL3-2B-Pretrained, InternVL3-8B-Pretrained, InternVL3-9B-Pretrained, InternVL3-14B-Pretrained, InternVL3-38B-Pretrained, InternVL3-78B-Pretrained | `internvl3.py` |
| **InternVL3.5** | InternVL3_5-1B, InternVL3_5-2B, InternVL3_5-4B, InternVL3_5-8B, InternVL3_5-14B, InternVL3_5-GPT-OSS-20B-A4B-Preview, InternVL3_5-30B-A3B, InternVL3_5-38B, InternVL3_5-241B-A28B, InternVL3_5-1B-MPO, InternVL3_5-2B-MPO, InternVL3_5-4B-MPO, InternVL3_5-8B-MPO, InternVL3_5-14B-MPO, InternVL3_5-30B-A3B-MPO, InternVL3_5-38B-MPO, InternVL3_5-241B-A28B-MPO, InternVL3_5-1B-Pretrained, InternVL3_5-2B-Pretrained, InternVL3_5-4B-Pretrained, InternVL3_5-8B-Pretrained, InternVL3_5-14B-Pretrained, InternVL3_5-30B-A3B-Pretrained, InternVL3_5-38B-Pretrained, InternVL3_5-241B-A28B-Pretrained, InternVL3_5-1B-Instruct, InternVL3_5-2B-Instruct, InternVL3_5-4B-Instruct, InternVL3_5-8B-Instruct, InternVL3_5-14B-Instruct, InternVL3_5-30B-A3B-Instruct, InternVL3_5-38B-Instruct, InternVL3_5-241B-A28B-Instruct | `internvl3d5.py` |
| **Janus** | Janus-1.3B | `janus.py` |
| **JanusFlow** | JanusFlow-1.3B | `janusflow.py` |
| **Janus-Pro** | Janus-Pro-1B, Janus-Pro-7B | `janus_pro.py` |
| **Llama-3.2-Vision** | Llama-3.2-11B-Vision-Instruct, Llama-3.2-90B-Vision-Instruct | `llama3d2_vision.py` |
| **Llama-4** | Llama-4-Scout-17B-16E-Instruct, Llama-4-Scout-17B-16E, Llama-4-Maverick-17B-128E-Instruct, Llama-4-Maverick-17B-128E | `llama4.py` |
| **LLaVA** | llava-1.5-7b-hf, llava-1.5-13b-hf | `llava.py` |
| **LLaVA-Next** | llava-v1.6-mistral-7b-hf, llava-v1.6-vicuna-7b-hf, llava-v1.6-vicuna-13b-hf, llava-v1.6-34b-hf, llama3-llava-next-8b-hf, llava-next-72b-hf, llava-next-110b-hf | `llava_next.py` |
| **LLaVA-Next-Interleave** | llava-next-interleave-qwen-0.5b, llava-next-interleave-qwen-7b, llava-next-interleave-qwen-7b-dpo | `llava_next_interleave.py` |
| **LLaVA-OneVision** | llava-onevision-qwen2-0.5b-si-hf, llava-onevision-qwen2-7b-si-hf, llava-onevision-qwen2-72b-si-hf, llava-onevision-qwen2-0.5b-ov-hf, llava-onevision-qwen2-7b-ov-hf, llava-onevision-qwen2-72b-ov-hf, llava-onevision-qwen2-7b-ov-chat-hf, llava-onevision-qwen2-72b-ov-chat-hf | `llava_ov.py` |
| **LLaVA-OneVision-1.5** | LLaVA-OneVision-1.5-8B-Instruct | `llava_ov_1d5.py` |
| **Mantis** | Mantis-8B-clip-llama3, Mantis-8B-siglip-llama3 | `mantis.py` |
| **Mantis-Fuyu** | Mantis-8B-Fuyu | `mantis_fuyu.py` |
| **Mantis-Idefics2** | Mantis-8B-Idefics2 | `mantis_idefics2.py` |
| **Mantis-LLaVA** | Mantis-llava-7b, Mantis-bakllava-7b | `mantis_llava.py` |
| **Moondream1** | moondream1 | `moondream1.py` |
| **Moondream2** | moondream2 | `moondream2.py` |
| **Ovis1.5** | Ovis1.5-Llama3-8B, Ovis1.5-Gemma2-9B | `ovis1d5.py` |
| **Ovis1.6** | Ovis1.6-Llama3.2-3B, Ovis1.6-Gemma2-9B, Ovis1.6-Gemma2-27B | `ovis1d6.py` |
| **Ovis2.5** | Ovis2.5-2B, Ovis2.5-9B | `ovis2d5.py` |
| **Phi-3V** | Phi-3.5-vision-instruct, Phi-3-vision-128k-instruct | `phi3v.py` |
| **Phi-4-Multimodal** | Phi-4-multimodal-instruct | `phi4mm.py` |
| **Qwen2-VL** | Qwen2-VL-2B-Instruct, Qwen2-VL-7B-Instruct, Qwen2-VL-72B-Instruct, Qwen2-VL-2B-Instruct-AWQ, Qwen2-VL-7B-Instruct-AWQ, Qwen2-VL-72B-Instruct-AWQ, Qwen2-VL-2B-Instruct-GPTQ-Int4, Qwen2-VL-7B-Instruct-GPTQ-Int4, Qwen2-VL-72B-Instruct-GPTQ-Int4 | `qwenvl2.py` |
| **Qwen2.5-VL** | Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-32B-Instruct, Qwen2.5-VL-72B-Instruct, Qwen2.5-VL-3B-Instruct-AWQ, Qwen2.5-VL-7B-Instruct-AWQ, Qwen2.5-VL-32B-Instruct-AWQ, Qwen2.5-VL-72B-Instruct-AWQ | `qwenvl2d5.py` |
| **Qwen2.5-Omni** | Qwen2.5-Omni-3B, Qwen2.5-Omni-7B, Qwen2.5-Omni-7B-GPTQ-Int4 | `qwenvl2d5_omni.py` |
| **Qwen3-VL** | Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct, Qwen3-VL-32B-Instruct, Qwen3-VL-30B-A3B-Instruct, Qwen3-VL-235B-A22B-Instruct, Qwen3-VL-2B-Thinking, Qwen3-VL-4B-Thinking, Qwen3-VL-8B-Thinking, Qwen3-VL-32B-Thinking, Qwen3-VL-30B-A3B-Thinking, Qwen3-VL-235B-A22B-Thinking | `qwen3_vl.py` |
| **Qwen3-Omni** | Qwen3-Omni-30B-A3B-Instruct, Qwen3-Omni-30B-A3B-Thinking, Qwen3-Omni-30B-A3B-Captioner | `qwen3_omni.py` |
| **R1-OneVision** | R1-Onevision-7B | `r1_onevision.py` |
| **SmolVLM** | SmolVLM-Instruct, SmolVLM-Instruct-DPO, SmolVLM-Instruct-Base, SmolVLM-Sythetic | `smolvlm.py` |
| **VideoLLaMA2** | VideoLLaMA2-7B | `videollama2.py` |
| **Vintern** | Vintern-1B-v2, Vintern-1B-v3_5, Vintern-3B-beta | `vintern.py` |
| **VLAA-Thinker** | VLAA-Thinker-Qwen2VL-2B, VLAA-Thinker-Qwen2VL-7B, VLAA-Thinker-Qwen2VL-7B-Zero, VLAA-Thinker-Qwen2.5VL-3B, VLAA-Thinker-Qwen2.5VL-7B | `vlaa_thinking.py` |
| **WeMM** | WeMM, WeMM-Chat-CN, WeMM-Chat-2k-CN | `wemm.py` |
| **XGen-MM** | xgen-mm-phi3-mini-instruct-interleave-r-v1.5 | `xgen.py` |
| **Xinyuan-VL** | Xinyuan-VL-2B | `xinyuanvl.py` |

### API Models

| Provider | Supported Models | Inference Script |
|---|---|---|
| **OpenAI** | gpt-4o-mini, gpt-4o, gpt-4.1-nano, gpt-4.1-mini, gpt-4.1, gpt-5-nano, gpt-5-mini, gpt-5, gpt-5.1, gpt-5.2 | `openai_gpt.py` |
| **Google Gemini** | gemini-2.5-flash-lite, gemini-2.5-flash, gemini-2.5-pro, gemini-3-flash-preview, gemini-3-pro-preview, gemini-3-pro-image-preview | `google_gemini.py` |
| **Anthropic Claude** | claude-haiku-4-5, claude-sonnet-4-0, claude-sonnet-4-5, claude-sonnet-4-6, claude-opus-4-0, claude-opus-4-1, claude-opus-4-5, claude-opus-4-6 | `anthropic_claude.py` |
| **xAI Grok** | grok-2-vision-1212, grok-4-0709, grok-4-fast-non-reasoning, grok-4-fast-reasoning, grok-4-1-fast-non-reasoning, grok-4-1-fast-reasoning | `xai_grok.py` |
| **Doubao (Ark)** | doubao-seed-1-6-vision-250815, doubao-seed-1-6-flash-250828, doubao-seed-1-6-lite-251015, doubao-seed-1-8-251228, doubao-seed-code-preview-251028, doubao-seed-2-0-mini-260215, doubao-seed-2-0-lite-260215, doubao-seed-2-0-code-preview-260215, doubao-seed-2-0-pro-260215 | `doubao_ark.py` |
| **Hunyuan** | hunyuan-vision, hunyuan-vision-1.5-instruct, hunyuan-t1-vision, hunyuan-turbos-vision, hunyuan-large-vision | `hunyuan_vision.py` |

---

## Supported Datasets

The framework provides three dataset backends, each specified through the `--dataset` argument.

### 1. Local JSON (`local@json`)

Load evaluation data from a local JSON file. Requires `--infile` and optionally `--img_dir`.

```bash
--dataset local@json --infile path/to/data.json --img_dir path/to/media/
```

**Dataset format** -- a JSON array of samples using the `messages` schema:

```json
[
  {
    "id": 0,
    "media": ["truck.png"],
    "messages": [
      {
        "role": "user",
        "question": "<image> What is shown in the image?",
        "answer": "A",
        "options": {"A": "A truck", "B": "A dog", "C": "A boat", "D": "A plane"},
        "choices": ["A", "B", "C", "D"],
        "hint": "Please choose the correct option."
      }
    ]
  }
]
```

Key fields:

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique sample identifier |
| `media` | No | List of media file paths (relative to `--img_dir`) |
| `messages` | Yes | List of message dicts with `role` and content fields |
| `messages[].role` | Yes | `"user"` or `"assistant"` |
| `messages[].question` | Conditional | The question text; use `<image>` and `<video>` as media placeholders. Required when `prompt` is not provided (used by the template engine). |
| `messages[].prompt` | No | If provided, used as the final prompt directly and template rendering is skipped |
| `messages[].options` | No | Dict of option labels to option text (for multiple-choice) |
| `messages[].choices` | No | List of choice keys (for multiple-choice questions) |
| `messages[].hint` | No | Optional hint text |

> **Note**: Each user message must have either `question` (rendered via template) or `prompt` (used directly). If both are provided, `prompt` takes precedence unless a user template is specified via `--template`.

Text-only samples simply use an empty `media` list:

```json
[
  {
    "id": 0,
    "media": [],
    "messages": [
      {
        "role": "user",
        "question": "What is the capital of France?",
        "prompt": "What is the capital of France?"
      }
    ]
  }
]
```

### 2. HuggingFace Datasets (`mmeval_hf@`)

Load datasets from HuggingFace Hub in the mm-eval format. These datasets include built-in Jinja2 prompt templates via their `metadata` config.

```bash
--dataset mmeval_hf@mm-eval/MMBench-en --split test
```

Currently available mm-eval HuggingFace datasets:

| Dataset | Splits | Usage |
|---------|--------|-------|
| [MMBench-en](https://huggingface.co/datasets/mm-eval/MMBench-en) | dev, test | `mmeval_hf@mm-eval/MMBench-en` |
| [MMBench-V11](https://huggingface.co/datasets/mm-eval/MMBench-V11) | dev, test | `mmeval_hf@mm-eval/MMBench-V11 --subset en` (subsets: en, cn) |

### 3. VLMEvalKit Datasets (`evalkit@`)

Direct access to 100+ VLMEvalKit benchmark datasets in TSV format. Datasets are automatically downloaded from HuggingFace on first use.

```bash
# By name
--dataset evalkit@MM-Math

# By URL
--dataset https://huggingface.co/datasets/mm-eval/VLMEvalKit/resolve/main/3DSRBench.tsv

# By local TSV path
--dataset datasets/MM-Math.tsv
```

Set the download directory (default: `./datasets`):

```bash
export DATASET_DIR=/path/to/datasets
```

<details>
<summary><b>Full list of supported VLMEvalKit datasets (click to expand)</b></summary>

| Category | Datasets |
|----------|----------|
| **General VQA** | MMBench_dev_ar, MMBench_dev_cn, MMBench_dev_en, MMBench_dev_en_test, MMBench_dev_pt, MMBench_dev_ru, MMBench_dev_tr, MMMB, MTL_MMBench_DEV, MMStar, MMVet, MMVet_Hard, POPE, LLaVABench, SEEDBench_IMG, SEEDBench2, SEEDBench2_Plus, RealWorldQA, WildVision, MUIRBench, NaturalBenchDataset, GQA_TestDev_Balanced, VizWiz |
| **Comprehensive** | MicroBench, MicroVQA, A-Bench_TEST, A-Bench_VAL, VL-RewardBench, GOBench, MM-IFEval, MIA-Bench, A-OKVQA |
| **OCR & Document** | OCRBench, InfoVQA_TEST, InfoVQA_VAL, ChartQA_TEST, CharXiv_descriptive_val, CharXiv_reasoning_val, TableVQABench |
| **Scientific & Math** | ScienceQA_TEST, ScienceQA_VAL, MathVista_MINI, MathVerse_MINI, MathVerse_MINI_Text_Dominant, MathVerse_MINI_Text_Lite, MathVerse_MINI_Vision_Dominant, MathVerse_MINI_Vision_Intensive, MathVerse_MINI_Vision_Only, MathVision, MathVision_MINI, MM-Math, OlympiadBench, WeMath, WeMath_COT |
| **Reasoning** | AI2D_TEST, AI2D_TEST_NO_MASK, BLINK, LogicVista, MMT-Bench_ALL, MMT-Bench_VAL, MMVP, VStarBench, TaskMeAnything_v1_imageqa_random, 3DSRBench, LEGO, A4Bench |
| **Quality & Perception** | Q-Bench1_TEST, Q-Bench1_VAL, AesBench_TEST, AesBench_VAL, HRBench4K, HRBench8K, R-Bench-Dis, R-Bench-Ref, VisOnlyQA-VLMEvalKit, CRPE_EXIST |
| **Medical** | PathVQA_TEST, PathVQA_VAL, PathMMU_TEST, PathMMU_VAL, OmniMedVQA, MedXpertQA_MM_test, WorldMedQA-V |
| **Multilingual** | MMMB_ar, MMMB_cn, MMMB_en, MMMB_pt, MMMB_ru, MMMB_tr, CMMU_MCQ, VCR_EN_EASY_ALL, VCR_EN_HARD_ALL, VCR_ZH_EASY_ALL, VCR_ZH_HARD_ALL |
| **Safety** | MLLMGuard_DS, AMBER |
| **Science Domains** | atomic_dataset, electro_dataset, mechanics_dataset, optics_dataset, quantum_dataset, statistics_dataset, hle, MMSci_DEV_Captioning_image_only, MMSci_DEV_MCQ, Creation_MMBench |

</details>
