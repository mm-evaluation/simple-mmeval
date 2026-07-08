#!/usr/bin/env python
"""Single-shot verification of the Azure OpenAI judge configuration.

Reads the same env vars the scorer uses and sends one tiny chat request,
mirroring mmeval/scoring/match/llm.py (reasoning-model path: max_completion_tokens,
no temperature override).

Usage:
    AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/ \
    python scripts/examples/verify_judge.py
"""
import os
import sys

from openai import AzureOpenAI

key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deploy = os.getenv("AZURE_OPENAI_DEPLOYNAME", "gpt-5.4-mini-2026-03-17")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
max_tokens = int(os.getenv("JUDGE_MAX_TOKENS", "2048"))

missing = [n for n, v in [("AZURE_OPENAI_KEY", key), ("AZURE_OPENAI_ENDPOINT", endpoint)] if not v]
if missing:
    print(f"[verify] MISSING env: {', '.join(missing)}")
    sys.exit(2)

print(f"[verify] endpoint={endpoint}")
print(f"[verify] deployment={deploy}  api_version={api_version}  max_completion_tokens={max_tokens}")

client = AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=api_version)

try:
    resp = client.chat.completions.create(
        model=deploy,
        messages=[
            {"role": "system", "content": "You are a strict evaluator for VLM outputs."},
            {"role": "user", "content": 'Reply with JSON only: {"is_correct": 1, "reason": "ping"}'},
        ],
        max_completion_tokens=max_tokens,
    )
    print("[verify] OK -> response content:")
    print(repr(resp.choices[0].message.content))
    print(f"[verify] usage: {resp.usage}")
except Exception as exc:
    print(f"[verify] FAILED: {type(exc).__name__}: {exc}")
    sys.exit(1)
