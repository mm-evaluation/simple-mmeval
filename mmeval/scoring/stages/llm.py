import json
import math
import os
import re
import threading
import time
from typing import Any, Dict

from mmeval.scoring.stages.base import BaseStage, StageResult
from mmeval.scoring.extraction import (
    can_infer_option,
    can_infer_text,
    extract_letter_set,
    extract_option_robust,
    normalize_for_exact,
    normalize_text,
    parse_gt_letters,
)


class _LocalJudgeClient:
    """In-process open-weight judge (judge_provider=local). The judge is a
    text-only task — both LLM stages build pure-text prompts and this client
    only ever tokenizes text — so the model is loaded as a causal LM
    (AutoModelForCausalLM + AutoTokenizer, dtype/device_map auto), which admits
    the strong instruction-following LLMs best suited to judging. It exposes the
    minimal chat-completions surface the judge stages consume, so prompts,
    parsing, retry/llm_error accounting and the semaphore are identical across
    providers. Generation is serialized by a lock: the bound is the GPU, not an
    API rate limit."""

    def __init__(self, model_name: str):
        import types as _types
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", device_map="auto")
        self._model.eval()
        self._lock = threading.Lock()
        self.chat = _types.SimpleNamespace(
            completions=_types.SimpleNamespace(create=self._create))

    def _create(self, model, messages, temperature=0.0, max_tokens=512,
                logprobs=False, **_ignored):
        import torch
        from types import SimpleNamespace as NS
        try:
            # Thinking-mode families (Qwen3/Qwen3.5) reason at length before
            # the terse judge/extraction answer; disable it — the verdict
            # contract is unchanged and per-call latency drops ~10x.
            prompt = self._tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=False)
        except TypeError:
            prompt = self._tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
        inputs = self._tok(prompt, return_tensors="pt").to(self._model.device)
        gen_kwargs = dict(max_new_tokens=int(max_tokens),
                          output_scores=bool(logprobs),
                          return_dict_in_generate=True,
                          pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id)
        if temperature and float(temperature) > 0:
            gen_kwargs.update(do_sample=True, temperature=float(temperature))
        else:
            gen_kwargs.update(do_sample=False)   # greedy: deterministic verdicts
        with self._lock, torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = out.sequences[0][inputs["input_ids"].shape[1]:]
        text = self._tok.decode(new_tokens, skip_special_tokens=True)
        content = None
        if logprobs and getattr(out, "scores", None):
            content = [NS(token=self._tok.decode([tid]),
                          logprob=torch.log_softmax(step[0].float(), dim=-1)[tid].item())
                       for tid, step in zip(new_tokens.tolist(), out.scores)]
        choice = NS(message=NS(content=text),
                    logprobs=NS(content=content) if content is not None else None)
        return NS(choices=[choice])


def local_judge_revision(model_name: str):
    """Resolved snapshot commit of a locally cached judge model. Enters the
    resume fingerprint for judge_provider=local: the model NAME alone does not
    pin the weights the way an API model string does."""
    base = os.path.join(
        os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface")), "hub",
        "models--" + str(model_name).replace("/", "--"), "refs", "main")
    try:
        with open(base) as f:
            return f.read().strip()
    except OSError:
        return None


_LOCAL_CLIENT_CACHE: Dict[str, "_LocalJudgeClient"] = {}


def _build_judge_client(args):
    """Build the judge client: local (in-process open-weight model via the
    framework's native transformers loading), openai, or azure_openai.

    Returns (client, model_or_deployment_name).
    """
    provider = (args.judge_provider or "openai").strip().lower()
    if provider == "local":
        if not args.judge_model:
            raise ValueError("`--judge_model` is required for the local judge.")
        # Process-wide cache: stages are rebuilt per result file, and a
        # multi-file scoring run must not reload the judge weights each time.
        if args.judge_model not in _LOCAL_CLIENT_CACHE:
            _LOCAL_CLIENT_CACHE[args.judge_model] = _LocalJudgeClient(args.judge_model)
        return _LOCAL_CLIENT_CACHE[args.judge_model], args.judge_model

    try:
        from openai import AzureOpenAI, OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is required for API judge providers.") from exc

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


def _resolve_setting(cli_value, env_name, default):
    """Single source of truth for judge knobs: CLI flag > env var > default."""
    if cli_value is not None:
        return int(cli_value)
    return int(os.getenv(env_name, str(default)))


# HTTP statuses worth retrying: timeout/conflict/rate-limit and server errors.
_RETRYABLE_STATUS = {408, 409, 429}
_RETRYABLE_EXC_NAMES = {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"}


def _is_retryable(exc) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS or status >= 500
    return type(exc).__name__ in _RETRYABLE_EXC_NAMES


def _verdict_probability(logprobs_content, verdict) -> Any:
    """P of the generated verdict digit, from API token logprobs. The judge reply
    is tiny JSON ({"is_correct": 1, ...}) whose FIRST digit token is the verdict;
    returns exp(logprob of that token), or None when logprobs are unavailable."""
    if not logprobs_content:
        return None
    target = str(int(verdict))
    for entry in logprobs_content:
        token = str(getattr(entry, "token", "")).strip().strip('"')
        if token == target:
            lp = getattr(entry, "logprob", None)
            return math.exp(lp) if lp is not None else None
        if token and token.strip("01") == "" and token != target:
            return None  # first digit token disagrees with the parsed verdict — don't guess
    return None


class _LLMStageBase(BaseStage):
    uses_llm = True
    """Shared client/config + a `_chat` helper with retry and reasoning-model
    handling (gpt-5/o* use max_completion_tokens and reject temperature overrides).

    Concurrent LLM calls are capped by a process-wide semaphore (--judge_concurrency /
    JUDGE_MAX_CONCURRENCY, default 4), DECOUPLED from the scorer's sample-worker count.
    Firing 16+ judge calls at once over a full dataset saturates the API's tokens/min
    quota; the gateway then returns degraded-but-valid responses (not errors), silently
    tanking accuracy. Capping concurrency keeps sustained throughput under quota.
    The semaphore is sized once per process by the first judge stage constructed."""

    _sem = None
    _sem_lock = threading.Lock()

    @classmethod
    def _get_sem(cls, n):
        if _LLMStageBase._sem is None:
            with _LLMStageBase._sem_lock:
                if _LLMStageBase._sem is None:
                    _LLMStageBase._sem = threading.Semaphore(max(1, n))
        return _LLMStageBase._sem

    def __init__(self, args):
        self.max_retry = max(1, _resolve_setting(getattr(args, "judge_max_retry", None), "JUDGE_MAX_RETRY", 3))
        self.concurrency = _resolve_setting(getattr(args, "judge_concurrency", None), "JUDGE_MAX_CONCURRENCY", 4)
        self.max_tokens = _resolve_setting(getattr(args, "judge_max_tokens", None), "JUDGE_MAX_TOKENS", 2048)
        self.temperature = float(args.judge_temperature)
        self.include_reason = bool(args.judge_include_reason)
        self.client, self.model = _build_judge_client(args)
        if not self.model:
            raise ValueError("`--judge_model` is required when a judge stage is enabled.")

    # Process-wide: set to False after a provider rejects the logprobs parameter,
    # so we don't pay a failed round-trip per sample on such deployments.
    _logprobs_supported = True

    def _chat(self, messages, want_logprobs=False):
        """Returns (text, error, logprobs): text is None on failure, error describes
        the last problem so callers can surface it instead of silently scoring
        'incorrect'. logprobs is the response token-logprob list (or None) — purely
        observational, requested only when want_logprobs and the provider allows it."""
        is_reasoning = bool(re.match(r"\s*(gpt-5|o1|o3|o4)", str(self.model), re.I))
        kwargs = {"model": self.model, "messages": messages}
        if is_reasoning:
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
            kwargs["temperature"] = self.temperature
        use_logprobs = want_logprobs and _LLMStageBase._logprobs_supported
        if use_logprobs:
            kwargs["logprobs"] = True
        sem = self._get_sem(self.concurrency)
        last_error = "no_attempt"
        for attempt in range(self.max_retry):
            try:
                with sem:  # cap concurrent API calls; don't hold it during backoff
                    resp = self.client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                if text.strip():
                    lp = getattr(resp.choices[0], "logprobs", None)
                    content = getattr(lp, "content", None) if lp is not None else None
                    return text, None, content
                last_error = "empty_response"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if not _is_retryable(exc):
                    if use_logprobs and getattr(exc, "status_code", None) == 400:
                        # 400 with logprobs requested: some gateways reject the
                        # parameter outright. Drop it (process-wide) and redo the
                        # call — the verdict matters more than the observability
                        # field. Auth/permission errors (401/403) fail fast below.
                        print(f"[{self.name}] provider rejected logprobs; disabling: {last_error}", flush=True)
                        _LLMStageBase._logprobs_supported = False
                        return self._chat(messages, want_logprobs=False)
                    # auth / bad request / not found — retrying cannot help
                    print(f"[{self.name}] non-retryable API error: {last_error}", flush=True)
                    return None, last_error, None
            print(f"[{self.name}] attempt {attempt + 1}/{self.max_retry} failed: {last_error}", flush=True)
            if attempt + 1 < self.max_retry:
                time.sleep(1.5 * (2 ** attempt))  # 1.5s, 3s, 6s exponential backoff
        return None, last_error, None


class LLMJudgeStage(_LLMStageBase):
    """The LLM directly judges whether the response is correct given
    QUESTION + MODEL_RESPONSE + GROUND_TRUTH, returning is_correct 0/1.
    Both verdicts DECIDE the sample (score 1.0/0.0); only transport/parse
    failures pass, carrying an llm_error marker."""

    name = "llm-judge"

    def run(self, sample: Dict[str, Any], context: Dict[str, Any]) -> StageResult:
        pred = normalize_text(context["pred"], extract_boxed=False, strip_latex_commands=False)
        gt = normalize_text(context["gt"], extract_boxed=False, strip_latex_commands=False)
        question = normalize_text(context["question"], extract_boxed=False, strip_latex_commands=False)
        question_type = context["question_type"]

        prompt = self._build_prompt(question_type, question, pred, gt, context)
        messages = [
            {"role": "system", "content": "You are a strict evaluator for VLM outputs."},
            {"role": "user", "content": prompt},
        ]

        text, error, logprobs = self._chat(messages, want_logprobs=True)
        parsed = self._parse_response(text) if text else None
        if parsed is None:
            # Transport/parse failure, NOT a "judged incorrect" verdict — the reason
            # marker keeps it distinguishable in score.json and in the summary.
            detail = error or "unparseable_judge_response"
            return StageResult.passed(reason=f"llm_error: {detail}")

        is_correct = int(parsed.get("is_correct", 0)) == 1
        # Observability only: P(verdict token) from API logprobs; never a
        # model-self-reported number, never an input to the verdict.
        prob = _verdict_probability(logprobs, parsed["is_correct"])
        meta = {"judge_prob": prob}
        reason = parsed.get("reason") if self.include_reason else None
        if is_correct:
            return StageResult.scored(1.0, matched=str(context["gt"]), reason=reason, meta=meta)
        return StageResult.scored(0.0, matched=None, reason=reason, meta=meta)

    def _build_prompt(self, question_type, question, pred, gt, context):
        options = context.get("option_texts", {})
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
                pass
        # Transport-level recovery from slightly malformed JSON that still carries
        # an UNAMBIGUOUS verdict (e.g. some instruct models drop the "reason" key:
        # {"is_correct": 1, ""}). This recovers the value the judge already
        # produced; it does NOT change how correctness is decided and never
        # guesses — only a single, consistent 0/1 (or true/false) is accepted, so
        # conflicting or absent verdicts remain llm_errors.
        verdicts = {v.lower() for v in re.findall(
            r'"is_correct"\s*:\s*(true|false|0|1)\b', text, flags=re.IGNORECASE)}
        if len(verdicts) == 1:
            v = verdicts.pop()
            reason = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
            return self._normalize_judge_obj(
                {"is_correct": 1 if v in ("1", "true") else 0,
                 "reason": reason.group(1) if reason else ""})
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


class LLMMatchStage(_LLMStageBase):
    """VLMEvalKit's approach: the LLM only EXTRACTS which option(s) the response
    picks, then we rule-based exact-compare against the ground truth. Mirrors
    extract_answer_from_item for mcq (rule-based can_infer prefetch, GPT extraction
    only as a fallback) and the LogicVista letters extractor for multi-letter gts
    (vlmeval/dataset/utils/logicvista.py:15-21, sorted-letters compare :50-68)."""

    name = "llm-match"

    # Modeled on VLMEvalKit's LogicVista extraction instruction (logicvista.py:15-21).
    EXTRACT_MULTI_TMPL = (
        'You are an AI assistant who will help me to match an answer with several '
        'options of a multiple-selection question. You are provided with a question, '
        'several options, and an answer, and you need to output which option letters '
        'the answer selects. Output ONLY the chosen uppercase letters separated by '
        'commas (e.g. "A, C"). If the answer selects none of the options, output Z.\n'
        'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: '
    )

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

    def run(self, sample: Dict[str, Any], context: Dict[str, Any]) -> StageResult:
        if context["question_type"] != "mcq":
            return StageResult.passed(reason="llm-match supports mcq only")

        pred = normalize_text(context["pred"], extract_boxed=False, strip_latex_commands=False)
        question = normalize_text(context["question"], extract_boxed=False, strip_latex_commands=False)
        # Drop media placeholders from the text prompt.
        question = re.sub(r"<(?:image|video)>", " ", question).strip()
        choices = [str(c).strip().upper() for c in (context["options"] or [])]
        option_texts = context.get("option_texts", {})

        # Compact gts only with real parsed options (see exact.py: the A-F
        # fallback would read word-like gts as letter sets).
        gt_letters = parse_gt_letters(context["gt"])
        if gt_letters is None and context.get("options_parsed"):
            gt_letters = extract_letter_set(context["gt"], choices)
        if gt_letters is not None and len(gt_letters) > 1:
            # Multi-letter gt (LogicVista): rule prefetch, then the letters-extraction
            # prompt; official sorted-set compare (utils/logicvista.py:50-68).
            got = extract_letter_set(pred, choices)
            error = None
            if got is None:
                rendered = " ".join(f"{k}. {v}" for k, v in option_texts.items())
                prompt = self.EXTRACT_MULTI_TMPL.format(question, rendered, pred)
                text, error, _ = self._chat([{"role": "user", "content": prompt}])
                if text:
                    got = extract_letter_set(text, choices)
            if got and got == gt_letters:
                return StageResult.scored(1.0, matched=", ".join(gt_letters))
            reason = f"llm_error: {error}" if error else None
            return StageResult.passed(reason=reason)

        # Single-letter gt: VLMEvalKit extract_answer_from_item protocol.
        gt_letter = (gt_letters[0] if gt_letters else None) or \
            extract_option_robust(context["gt"], choices, option_texts) or \
            normalize_for_exact(context["gt"]).upper()

        opt, error = self._extract_option(question, pred, choices, option_texts)
        if opt and gt_letter and opt == gt_letter:
            return StageResult.scored(1.0, matched=gt_letter)
        reason = f"llm_error: {error}" if error else None
        return StageResult.passed(reason=reason)

    def _extract_option(self, question, pred, choices, option_texts):
        """Returns (option_or_None, api_error_or_None)."""
        # Prefetch with rules (VLMEvalKit can_infer = can_infer_option | can_infer_text).
        # Original case, matching VLMEvalKit — uppercasing first would turn every
        # article "a" into a hit for option A.
        opt = can_infer_option(str(pred), choices) or can_infer_text(str(pred), option_texts)
        if opt:
            return opt, None
        # GPT extraction fallback.
        rendered = [f"{k}. {v}" for k, v in option_texts.items()] if option_texts else []
        options_str = " ".join(rendered)
        prompt = self.EXTRACT_TMPL.format(question, options_str, pred)
        text, error, _ = self._chat([{"role": "user", "content": prompt}])
        if not text:
            return None, error
        valid = choices + ["Z"]
        got = can_infer_option(text.upper(), valid)
        if got:
            return got, None
        # The extractor replies with a bare letter; accept any valid candidate (not
        # just A-D — benchmarks with options E/F+ must not fall off this cliff).
        m = re.search(rf"\b([{re.escape(''.join(valid))}])\b", text.upper())
        return (m.group(1) if m else None), None

