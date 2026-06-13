import json
import os
import re
import threading
import time
from typing import Any, Dict

from mmeval.scoring.match.base import BaseMatcher, MatchResult
from mmeval.scoring.schema import (
    can_infer_option,
    can_infer_text,
    infer_mcq_option,
    normalize_for_exact,
    normalize_text,
)


def _build_judge_client(args):
    """Build the OpenAI / AzureOpenAI client. Shared by all llm-based matchers.

    Returns (client, model_or_deployment_name).
    """
    try:
        from openai import AzureOpenAI, OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is required for llm matchers.") from exc

    provider = (args.judge_provider or "openai").strip().lower()
    if provider == "azure_openai":
        key = os.getenv("AZURE_OPENAI_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not key or not endpoint:
            raise ValueError("AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT are required for azure_openai judge.")
        model = os.getenv("AZURE_OPENAI_DEPLOYNAME", args.judge_model)
        client = AzureOpenAI(
            api_key=key,
            azure_endpoint=endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        )
        return client, model

    key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not key:
        raise ValueError("OPENAI_API_KEY is required for openai judge.")
    kwargs = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs), args.judge_model


class _LLMMatcherBase(BaseMatcher):
    """Shared client/config + a `_chat` helper with retry and reasoning-model
    handling (gpt-5/o* use max_completion_tokens and reject temperature overrides).

    Concurrent LLM calls are capped by a process-wide semaphore (JUDGE_MAX_CONCURRENCY,
    default 4), DECOUPLED from the scorer's sample-worker count. Firing 16+ judge calls
    at once over a full dataset saturates the API's tokens/min quota; the gateway then
    returns degraded-but-valid responses (not errors), silently tanking accuracy.
    Capping concurrency keeps sustained throughput under quota."""

    _sem = None
    _sem_lock = threading.Lock()

    @classmethod
    def _get_sem(cls):
        if _LLMMatcherBase._sem is None:
            with _LLMMatcherBase._sem_lock:
                if _LLMMatcherBase._sem is None:
                    n = max(1, int(os.getenv("JUDGE_MAX_CONCURRENCY", "4")))
                    _LLMMatcherBase._sem = threading.Semaphore(n)
        return _LLMMatcherBase._sem

    def __init__(self, args):
        # Default to 3 retries (env-overridable); exponential backoff handles
        # transient rate-limit (HTTP 429) responses.
        self.max_retry = max(1, int(os.getenv("JUDGE_MAX_RETRY", "3")))
        self.temperature = float(args.judge_temperature)
        self.include_reason = bool(args.judge_include_reason)
        self.max_tokens = max(2048, int(os.getenv("JUDGE_MAX_TOKENS", "2048")))
        self.client, self.model = _build_judge_client(args)
        if not self.model:
            raise ValueError("`--judge_model` is required when an llm matcher is enabled.")

    def _chat(self, messages):
        is_reasoning = bool(re.match(r"\s*(gpt-5|o1|o3|o4)", str(self.model), re.I))
        kwargs = {"model": self.model, "messages": messages}
        if is_reasoning:
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
            kwargs["temperature"] = self.temperature
        sem = self._get_sem()
        for attempt in range(self.max_retry):
            try:
                with sem:  # cap concurrent API calls; don't hold it during backoff
                    resp = self.client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                if text.strip():
                    return text
            except Exception:
                pass
            time.sleep(1.5 * (2 ** attempt))  # 1.5s, 3s, 6s exponential backoff
        return None


class LLMJudgeMatcher(_LLMMatcherBase):
    """Our approach: the LLM directly judges whether the response is correct given
    QUESTION + MODEL_RESPONSE + GROUND_TRUTH, returning is_correct 0/1."""

    name = "llm-judge"

    def match(self, sample: Dict[str, Any], context: Dict[str, Any]) -> MatchResult:
        pred = normalize_text(context["pred"], extract_boxed=False, strip_latex_commands=False)
        gt = normalize_text(context["gt"], extract_boxed=False, strip_latex_commands=False)
        question = normalize_text(context["question"], extract_boxed=False, strip_latex_commands=False)
        question_type = context["question_type"]

        prompt = self._build_prompt(question_type, question, pred, gt, context)
        messages = [
            {"role": "system", "content": "You are a strict evaluator for VLM outputs."},
            {"role": "user", "content": prompt},
        ]

        text = self._chat(messages)
        parsed = self._parse_response(text) if text else None
        if parsed is None:
            return MatchResult(is_match=False, reason="llm_judge_failed")

        is_correct = int(parsed.get("is_correct", 0)) == 1
        reason = parsed.get("reason") if self.include_reason else None
        if is_correct:
            return MatchResult(is_match=True, matched=str(context["gt"]), reason=reason)
        return MatchResult(is_match=False, matched=None, reason=reason)

    def _build_prompt(self, question_type, question, pred, gt, context):
        options = context.get("option_text_map", {})
        options_text = ""
        if question_type == "mcq" and options:
            rendered = [f"{k}. {v}" for k, v in options.items()]
            options_text = "\nOptions:\n" + "\n".join(rendered)

        reason_instruction = "Include a concise reason." if self.include_reason else "Set reason to an empty string."
        return (
            "Judge whether MODEL_RESPONSE is correct given QUESTION and GROUND_TRUTH.\n"
            f"QuestionType: {question_type}\n"
            f"QUESTION: {question}\n"
            f"{options_text}\n"
            f"MODEL_RESPONSE: {pred}\n"
            f"GROUND_TRUTH: {gt}\n\n"
            "Return JSON only with schema:\n"
            '{"is_correct": 0_or_1, "reason": "string"}\n'
            f"{reason_instruction}\n"
            "No markdown, no extra fields."
        )

    def _parse_response(self, text):
        text = text.strip()
        if not text:
            return None
        try:
            return self._normalize_judge_obj(json.loads(text))
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return self._normalize_judge_obj(json.loads(match.group(0)))
            except Exception:
                return None
        return None

    @staticmethod
    def _normalize_judge_obj(obj):
        if not isinstance(obj, dict):
            return None
        raw = obj.get("is_correct", 0)
        if isinstance(raw, bool):
            value = 1 if raw else 0
        else:
            try:
                value = int(raw)
            except Exception:
                value = 0
        reason = obj.get("reason", "")
        return {"is_correct": 1 if value == 1 else 0, "reason": str(reason)}


class LLMMatchMatcher(_LLMMatcherBase):
    """VLMEvalKit's approach: the LLM only EXTRACTS which option the response picks
    (single letter A/B/C/D, or Z for none), then we rule-based exact-compare that
    letter to the ground-truth letter. Mirrors VLMEvalKit's extract_answer_from_item:
    rule-based prefetch (can_infer) first, GPT extraction only as a fallback."""

    name = "llm-match"

    # Verbatim from VLMEvalKit (vlmeval/dataset/utils/multiple_choice.py build_prompt).
    EXTRACT_TMPL = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Z. '
        'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 3: \n'
        'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: '
    )

    def match(self, sample: Dict[str, Any], context: Dict[str, Any]) -> MatchResult:
        if context["question_type"] != "mcq":
            return MatchResult(is_match=False, reason="llm-match supports mcq only")

        pred = normalize_text(context["pred"], extract_boxed=False, strip_latex_commands=False)
        question = normalize_text(context["question"], extract_boxed=False, strip_latex_commands=False)
        # Drop media placeholders from the text prompt.
        question = re.sub(r"<(?:image|video)>", " ", question).strip()
        choices = [str(c).strip().upper() for c in (context["options"] or [])]
        option_text_map = context.get("option_text_map", {})

        gt_letter = infer_mcq_option(context["gt"], choices, option_text_map) or \
            normalize_for_exact(context["gt"]).upper()

        opt = self._extract_option(question, pred, choices, option_text_map)
        if opt and gt_letter and opt == gt_letter:
            return MatchResult(is_match=True, matched=gt_letter)
        return MatchResult(is_match=False, matched=None)

    def _extract_option(self, question, pred, choices, option_text_map):
        # Prefetch with rules (VLMEvalKit can_infer = can_infer_option | can_infer_text).
        opt = can_infer_option(str(pred).upper(), choices) or can_infer_text(str(pred), option_text_map)
        if opt:
            return opt
        # GPT extraction fallback.
        rendered = [f"{k}. {v}" for k, v in option_text_map.items()] if option_text_map else []
        options_str = " ".join(rendered)
        prompt = self.EXTRACT_TMPL.format(question, options_str, pred)
        text = self._chat([{"role": "user", "content": prompt}])
        if not text:
            return None
        valid = choices + ["Z"]
        got = can_infer_option(text.upper(), valid)
        if got:
            return got
        m = re.search(r"\b([A-DZ])\b", text.upper())
        return m.group(1) if m else None


# Backwards-compatible alias.
LLMMatcher = LLMJudgeMatcher
