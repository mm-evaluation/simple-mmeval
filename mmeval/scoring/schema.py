import re
import unicodedata
from typing import Any, Dict, List, Optional


MCQ_OPTIONS_DEFAULT = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _to_text(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        value = value[0]
    if value is None:
        return ""
    return str(value)


def _extract_boxed_content(text: str) -> str:
    marker = "\\boxed{"
    start = text.find(marker)
    if start < 0:
        return text
    i = start + len(marker)
    depth = 1
    content = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(content).strip()
        content.append(ch)
        i += 1
    return text


def normalize_text(
    value: Any,
    lower: bool = False,
    extract_boxed: bool = True,
    strip_latex_commands: bool = True,
) -> str:
    text = _to_text(value)

    # Keep normalization mostly syntax-level and semantics-preserving.
    text = unicodedata.normalize("NFKC", text)
    if extract_boxed:
        text = _extract_boxed_content(text)

    # Remove CoT / answer wrappers if models output tagged sections.
    text = re.sub(r"</?(?:think|answer)\s*>", " ", text, flags=re.IGNORECASE)

    # Remove markdown code fences while preserving body text.
    text = text.replace("```", " ")

    # Common inference artifacts.
    text = text.replace("<|end_of_sentence|>", "")
    text = text.replace("<|end▁of▁sentence|>", "")
    text = text.replace("</s>", "")
    text = text.replace("<CONCLUSION>", "")
    text = text.replace("</CONCLUSION>", "")
    text = text.replace("Falcon: ", "")

    # Light LaTeX cleanup inspired by math-verify style normalization.
    if strip_latex_commands:
        text = re.sub(r"\\(?:left|right|displaystyle|mathrm|textbf|textit|text)\b", " ", text)
        text = re.sub(r"\\(?:,|;|!|\:)", " ", text)  # spacing commands
        text = text.replace("\\%", "%")

    # Normalize common unicode/operator variants.
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("×", "*").replace("·", "*")
    text = text.replace("÷", "/")

    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if lower:
        text = text.lower()
    return text


def normalize_for_exact(value: Any) -> str:
    return normalize_text(value, lower=True).strip()


def normalize_open_answer(value: Any) -> str:
    text = normalize_for_exact(value)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def get_field(sample: Dict[str, Any], field: str, default: Any = None) -> Any:
    if not field:
        return default
    if field == "messages[-1].response":
        messages = sample.get("messages") or []
        if not messages:
            return default
        last = messages[-1]
        if not isinstance(last, dict):
            return default
        return last.get("response", default)
    # Top-level (TSV/evalkit format keeps gt like `answer` at sample top level).
    if field in sample:
        return sample.get(field, default)
    # Fallback: mm-eval HF format stores gt/options/etc. inside the user message
    # (messages[0]), not at the top level. Look there before giving up.
    messages = sample.get("messages") or []
    if messages and isinstance(messages[0], dict) and field in messages[0]:
        return messages[0].get(field, default)
    return default


def get_question(sample: Dict[str, Any]) -> str:
    messages = sample.get("messages") or []
    if messages and isinstance(messages[0], dict):
        q = messages[0].get("question")
        if q is not None:
            return str(q)
    return str(sample.get("question", ""))


def _letters_for(n: int) -> List[str]:
    return MCQ_OPTIONS_DEFAULT[: max(0, n)]


def _option_letters_from_question(question: str) -> List[str]:
    """Parse option labels embedded in the question text. Handles both the
    "A: ... B: ..." (MMStar) and "(A) ... (B) ..." (BLINK) inline formats. Returns
    only the leading contiguous run from A (A,B,C,D,...) to avoid stray letters."""
    if not isinstance(question, str):
        return []
    found = set()
    # "(A)" / "(A) " style and "A." / "A:" / "A)" at a token boundary.
    for m in re.findall(r"\(([A-H])\)", question):
        found.add(m.upper())
    for m in re.findall(r"(?:^|[\n\s])([A-H])[\.\):]", question):
        found.add(m.upper())
    letters = []
    for ch in MCQ_OPTIONS_DEFAULT[:8]:
        if ch in found:
            letters.append(ch)
        else:
            break  # keep only the contiguous A,B,C,... prefix
    return letters


def _choices_to_letters(choices: List[Any]) -> List[str]:
    """A `choices` list is either option letters (['A','B',...]) or option values
    (['0','3',...]). Letters are used directly; values map positionally to A,B,C,..."""
    items = [str(c).strip() for c in choices]
    if items and all(len(c) == 1 and c.upper() in MCQ_OPTIONS_DEFAULT for c in items):
        return [c.upper() for c in items]
    return _letters_for(len(items))


def extract_options(sample: Dict[str, Any]) -> List[str]:
    msg = None
    messages = sample.get("messages") or []
    if messages and isinstance(messages[0], dict):
        msg = messages[0]

    for src in (msg, sample):
        if not isinstance(src, dict):
            continue
        options = src.get("options")
        if isinstance(options, dict) and options:
            return [str(k).strip().upper() for k in options.keys()]
        choices = src.get("choices")
        if isinstance(choices, list) and choices:
            return _choices_to_letters(choices)

    # Inline-options datasets (MMStar/BLINK keep options in the question text and
    # ship empty `options`/`choices`): derive the real letters from the question so
    # we don't fall back to A-F and mis-count stray letters (e.g. "F = ma").
    if isinstance(msg, dict):
        letters = _option_letters_from_question(msg.get("question") or msg.get("prompt") or "")
        if letters:
            return letters

    # TSV-style top-level option columns (A/B/C/D as sample keys).
    dynamic = [x for x in MCQ_OPTIONS_DEFAULT if x in sample]
    # Fall back to A-F as candidate letters (some MCQ datasets, e.g. LogicVista,
    # keep the choices in the image and don't list them in the question text).
    # Whether the item is actually MCQ vs open is decided by infer_question_type
    # from the ground-truth shape, not by this default.
    return dynamic or MCQ_OPTIONS_DEFAULT[:6]


def infer_question_type(
    sample: Dict[str, Any],
    score_force_question_type: str = "auto",
    score_type_field: Optional[str] = None,
    gt: Any = None,
) -> str:
    forced = (score_force_question_type or "auto").strip().lower()
    if forced in {"mcq", "open"}:
        return forced

    if score_type_field:
        raw = str(sample.get(score_type_field, "")).strip().lower()
        if raw in {"mcq", "mc", "multi-choice", "multiple_choice"}:
            return "mcq"
        if raw in {"open", "open-ended", "open_ended", "free"}:
            return "open"

    # The ground-truth shape is the most reliable signal: a bare option letter
    # ("A", "(D)") => MCQ; yes/no, a number, or free text => open-ended. This
    # correctly splits mixed benchmarks (e.g. RealWorldQA interleaves MCQ items
    # with yes/no and numeric items) where extract_options always yields A-F.
    if gt is not None:
        g = re.sub(r"[()\[\].:;\s\*]", "", str(gt)).upper()
        if len(g) == 1 and g in MCQ_OPTIONS_DEFAULT:
            return "mcq"
        if g != "":
            return "open"

    options = extract_options(sample)
    if options:
        return "mcq"
    return "open"


def extract_mcq_option(text: Any, candidates: List[str]) -> Optional[str]:
    # Prefer the content of the LAST <answer>...</answer> tag if the model emitted
    # the reasoning-style "<think>...</think><answer>X</answer>" format. Otherwise
    # the buried letter gets diluted by surrounding reasoning text and the regexes
    # below — which anchor on the start of the string — miss it.
    raw_text = _to_text(text)
    ans = re.findall(r"<answer\s*>\s*(.*?)\s*</answer\s*>", raw_text, flags=re.DOTALL | re.IGNORECASE)
    clean = normalize_text(ans[-1]) if ans else normalize_text(text)
    if not clean:
        return None

    uppercase = clean.strip().upper()
    if uppercase in candidates:
        return uppercase
    if len(uppercase) >= 2 and uppercase[0] in candidates and uppercase[1] in {".", ")", ":"}:
        return uppercase[0]

    patterns = [
        r"^(?:answer|option|choice)?\s*[:\-]?\s*([A-Z])(?:[\s\.\),:;]|$)",
        r"(?:final answer|the answer is|answer is|choice is|option is)\s*[:\-]?\s*([A-Z])(?:[\s\.\),:;]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, uppercase, re.IGNORECASE)
        if match:
            candidate = match.group(1).upper()
            if candidate in candidates:
                return candidate
    return None


# Punctuation/markdown that VLMEvalKit strips to whitespace so option letters
# become standalone tokens. PATCH: append full-width CJK punctuation, which the
# upstream char list misses — without it "最终答案：A" (full-width colon) never
# splits and the trailing letter is lost.
_CHOICE_PUNCT = ".()[],:;!*#{}" + "：；，。、！？「」『』（）【】《》"

# PATCH (kept from our matcher): explicit "final answer" style cues, extended with
# CJK markers ("答案"/"最终答案"/"选"). VLMEvalKit relies purely on token position;
# these cues let us resolve answers even when several option letters appear in the
# reasoning (count != 1, where VLMEvalKit would defer to its GPT extractor).
# NOTE: the cue phrases are case-insensitive, but the option letter is matched
# case-SENSITIVELY via (?-i:...) so a lowercase article ("the answer is a teddy
# bear") is not mistaken for option "A".
_ANSWER_CUE = re.compile(
    r"(?:final answer|the answer is|answer is|answer\s*:|option is|"
    r"(?:the\s+)?correct\s+(?:option|answer|choice)(?:\s+is)?|"  # "correct option: (C)"
    r"最终答案|答案|选)\s*[:\-：]?\s*"
    # The letter may be wrapped by any mix of '(' / '*' / full-width parens in any
    # order: "A", "(A)", "**A**", "**(A)**", "(A) text", "**(A) 0.0"...
    r"[\(（\*]{0,4}(?-i:([A-Z]))[\)）\*]{0,4}(?:[\s\.\),:;、，。]|$)",
    re.IGNORECASE,
)


def _strip_choice_punct(text: str) -> str:
    for ch in _CHOICE_PUNCT:
        text = text.replace(ch, " ")
    return text


def can_infer_option(answer: str, choices: List[str]) -> Optional[str]:
    """VLMEvalKit-style: strip punctuation -> tokens -> the unique option letter
    appearing near the end of the answer."""
    answer_mod = _strip_choice_punct(answer)
    splits = [x.strip() for x in answer_mod.split() if x.strip()]
    present = [c for c in choices if c in splits]
    if len(present) == 1:
        ch = present[0]
        # near the end (the model's final pick), mirroring VLMEvalKit's last-5 window
        if splits.index(ch) > len(splits) - 5 or splits.count(ch) == 1 and len(splits) <= 5:
            return ch
    return None


def can_infer_text(answer: str, option_text_map: Dict[str, str]) -> Optional[str]:
    """VLMEvalKit-style: the answer restates exactly one option's text."""
    if not option_text_map:
        return None
    low = answer.lower()
    cands = [k for k, v in option_text_map.items() if v and str(v).strip().lower() in low]
    if len(cands) == 1:
        return cands[0]
    return None


def infer_mcq_option(
    text: Any, candidates: List[str], option_text_map: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """Robust MCQ-letter extraction: VLMEvalKit's can_infer (option + text) plus our
    patches (<answer> tag, explicit answer cues, CJK punctuation)."""
    raw = _to_text(text)
    if not raw:
        return None
    cand_set = [str(c).strip().upper() for c in (candidates or MCQ_OPTIONS_DEFAULT)]

    # PATCH 1 (ours): trust the LAST <answer>...</answer> tag if present.
    tags = re.findall(r"<answer\s*>\s*(.*?)\s*</answer\s*>", raw, flags=re.DOTALL | re.IGNORECASE)
    if tags:
        inner = can_infer_option(tags[-1].upper(), cand_set) or (
            tags[-1].strip().upper() if tags[-1].strip().upper() in cand_set else None
        )
        if inner:
            return inner

    # PATCH 1b (parity with exact's normalize_text): the LAST \boxed{...}, e.g.
    # "$\boxed{C}$", "$\boxed{C: \theta=1.475}$", or nested "$\boxed{\text{C}}$".
    # Capture the option letter directly, tolerating an inner \text{}/\mathrm{} wrap
    # and a leading '(' / '*'.
    boxes = re.findall(r"\\boxed\{\s*(?:\\[a-zA-Z]+\s*\{\s*)?[\(\*]*\s*([A-Za-z])", raw)
    boxes = [b.upper() for b in boxes if b.upper() in cand_set]
    if boxes:
        return boxes[-1]

    # PATCH 2 (ours): explicit answer cue, taking the LAST occurrence (markdown/CJK safe).
    cues = _ANSWER_CUE.findall(raw)
    cues = [c.upper() for c in cues if c.upper() in cand_set]
    if cues:
        return cues[-1]

    # VLMEvalKit core: unique option letter near the end.
    opt = can_infer_option(raw.upper(), cand_set)
    if opt:
        return opt

    # PATCH 3 (ours): answer-label formats VLMEvalKit's position/count test drops.
    #  3a — answer-first: the response leads with "C:" / "C." / "**C)**" (count!=1
    #       cases like "C: A man surfing" where 'A' is just an article). Gated to
    #       SHORT responses only — in a long answer the leading letter is often
    #       reconsidered later ("A: ... wait, the answer is not listed"), so we
    #       defer those to the end-position / cue logic (parity with VLMEvalKit).
    if len(raw.strip()) <= 80:
        head = re.match(r"\s*\*{0,2}\(?([A-Z])\)?\s*[:\.\)]\s", raw)
        if head and head.group(1).upper() in cand_set:
            return head.group(1).upper()
    #  3b — a bolded option label "**D:**" / "**(D)**" anywhere; take the LAST
    #       (the model's final restated pick), e.g. "...is **D: the city on the coast**".
    bolds = re.findall(r"\*\*\s*\(?([A-Z])\)?\s*[:\.]", raw)
    bolds = [b.upper() for b in bolds if b.upper() in cand_set]
    if len(set(bolds)) == 1:
        return bolds[0]

    # VLMEvalKit fallback: unique option *text* match.
    return can_infer_text(raw, option_text_map or {})

