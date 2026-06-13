# LLM-as-Judge API Reference (simple-mmeval scoring)

LLM judge used for `template,llm-match` scoring of MCQ / open-ended benchmarks.
Provider: **Azure OpenAI**, accessed via ByteDance internal modelhub proxy.

## Environment variables

```bash
# === Azure OpenAI judge ===
export AZURE_OPENAI_KEY="kEQGmoirhe9XZdDOCmE5MxAL3vDx2ViT_GPT_AK"
export AZURE_OPENAI_ENDPOINT="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl"
export AZURE_OPENAI_DEPLOYNAME="gpt-5.4-mini-2026-03-17"
export AZURE_OPENAI_API_VERSION="2024-02-01"

# === call-side concurrency / retry control ===
export JUDGE_MAX_CONCURRENCY=4     # cap concurrency to avoid quota saturation / degradation
export JUDGE_MAX_RETRY=3
export JUDGE_MAX_TOKENS=2048
```

## Scoring command

```bash
python mmeval/score.py --out_dir <DATASET_DIR> --score_result_glob 'result.json' \
  --matching_order template,llm-match \
  --judge_provider azure_openai \
  --judge_model gpt-5.4-mini-2026-03-17 \
  --score_output_name score_llm.json \
  --parallel_per_task 16 --no_score_resume
```

## Notes

- **judge model**: `gpt-5.4-mini-2026-03-17` (Azure deployment name is identical).
- **matching_order = `template,llm-match`**: try rule/template answer-matching first;
  fall back to the LLM judge only when the rule match fails. Fewer calls, more stable.
- **provider = `azure_openai`**, but the endpoint is the ByteDance internal modelhub
  proxy (`*.byteintl.net`), NOT public Azure — only reachable inside the corp network.
- `JUDGE_MAX_CONCURRENCY=4` keeps concurrency low to avoid saturating the judge quota
  (which otherwise degrades throughput/quality).

> Internal key/endpoint — usable only on the company network. For external reference,
> the reusable part is the *methodology*: rule-first + LLM fallback, the judge model,
> and the concurrency/retry settings.
