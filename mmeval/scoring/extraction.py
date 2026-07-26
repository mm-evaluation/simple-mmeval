import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


OPTION_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Question-type taxonomy. Every type is justified by at least one mm-eval org
# dataset (per-subset audit: docs/en/SCORING_COVERAGE.md): mcq (ScienceQA-IMG, MMStar, ...;
# multi-letter answers like LogicVista's "A, C" are mcq with |letters|>1),
# yes_no (VSR, GQA-Spatial verify rows), numeric (CharXiv, VLMsAreBlind, SeePhys),
# open (MM-Vet-v2, CharXiv descriptive).
QUESTION_TYPES = ("mcq", "yes_no", "numeric", "open")



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


# normalize_text machinery, compiled once (hot path: called several times per sample).
_COT_TAG_RE = re.compile(r"</?(?:think|answer)\s*>", re.IGNORECASE)
_LATEX_CMD_RE = re.compile(r"\\(?:left|right|displaystyle|mathrm|textbf|textit|text)\b")
_LATEX_SPACING_RE = re.compile(r"\\(?:,|;|!|\:)")
_WHITESPACE_RE = re.compile(r"\s+")
# Inference artifacts and unicode operator variants folded via one translate/replace pass.
_ARTIFACT_REPLACEMENTS = (
    ("```", " "), ("<|end_of_sentence|>", ""), ("<|end▁of▁sentence|>", ""),
    ("</s>", ""), ("<CONCLUSION>", ""), ("</CONCLUSION>", ""), ("Falcon: ", ""),
)
_OPERATOR_TRANSLATION = str.maketrans({"−": "-", "–": "-", "—": "-", "×": "*", "·": "*", "÷": "/"})


def normalize_text(
    value: Any,
    lower: bool = False,
    extract_boxed: bool = True,
    strip_latex_commands: bool = True,
) -> str:
    """Syntax-level, semantics-preserving normalization of a model answer:
    NFKC folding, \boxed{} content extraction, <think>/<answer> tag removal,
    inference-artifact stripping, light LaTeX cleanup, unicode operator folding,
    whitespace collapse (and optional lowercasing)."""
    text = _to_text(value)

    text = unicodedata.normalize("NFKC", text)
    if extract_boxed:
        text = _extract_boxed_content(text)

    text = _COT_TAG_RE.sub(" ", text)
    for old_str, new_str in _ARTIFACT_REPLACEMENTS:
        text = text.replace(old_str, new_str)

    if strip_latex_commands:
        text = _LATEX_CMD_RE.sub(" ", text)
        text = _LATEX_SPACING_RE.sub(" ", text)
        text = text.replace("\\%", "%")

    text = text.translate(_OPERATOR_TRANSLATION)

    text = _WHITESPACE_RE.sub(" ", text.strip())
    if lower:
        text = text.lower()
    return text


def normalize_for_exact(value: Any) -> str:
    return normalize_text(value, lower=True).strip()


_WRAPPING_QUOTES_RE = re.compile(r"^[\"'`]+|[\"'`]+$")
_TRAILING_SENTENCE_PUNCT_RE = re.compile(r"[\.。!！?？]+$")


def normalize_open_answer(value: Any) -> str:
    """normalize_for_exact plus open-answer specifics: strip wrapping quotes and
    trailing sentence punctuation ("... Bridge." vs "bridge")."""
    text = normalize_for_exact(value)
    text = _WRAPPING_QUOTES_RE.sub("", text)
    text = _TRAILING_SENTENCE_PUNCT_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


# Render inputs — the fields a dataset's prompt template can reference; they are
# the ONLY fields that live inside messages[0] (dataset-standard boundary rule).
_RENDER_INPUT_FIELDS = frozenset({"question", "options", "choices", "hint", "prompt"})


def get_field(sample: Dict[str, Any], field: str, default: Any = None) -> Any:
    """Read a field from the ONE canonical result.json layout: the prediction
    at messages[-1].response; render inputs (question/options/choices/hint/
    prompt) inside messages[0]; every grading field (answer, question_type,
    reference_response, and any --score_gt_field) at the sample TOP level.
    All backends emit this layout; files with grading fields embedded in
    messages[0] are rejected by the scorer."""
    if not field:
        return default
    if field == "messages[-1].response":
        messages = sample.get("messages") or []
        last = messages[-1] if messages else None
        return last.get("response", default) if isinstance(last, dict) else default
    if field in _RENDER_INPUT_FIELDS:
        messages = sample.get("messages") or []
        first = messages[0] if messages else None
        return first.get(field, default) if isinstance(first, dict) else default
    return sample.get(field, default)


def get_question(sample: Dict[str, Any]) -> str:
    q = get_field(sample, "question")
    return "" if q is None else str(q)


_PAREN_OPTION_RE = re.compile(r"\(([A-H])\)")
_LABELLED_OPTION_RE = re.compile(r"(?:^|[\n\s])([A-H])[\.\):]")


def _option_letters_from_question(question: str) -> List[str]:
    """Parse option labels embedded in the question text. Handles both the
    "A: ... B: ..." (MMStar) and "(A) ... (B) ..." (BLINK) inline formats. Returns
    only the leading contiguous run from A (A,B,C,D,...) to avoid stray letters."""
    if not isinstance(question, str):
        return []
    found = set()
    # "(A)" / "(A) " style and "A." / "A:" / "A)" at a token boundary.
    for m in _PAREN_OPTION_RE.findall(question):
        found.add(m.upper())
    for m in _LABELLED_OPTION_RE.findall(question):
        found.add(m.upper())
    letters = []
    for ch in OPTION_LETTERS[:8]:
        if ch in found:
            letters.append(ch)
        else:
            break  # keep only the contiguous A,B,C,... prefix
    return letters


def _choices_to_letters(choices: List[Any]) -> List[str]:
    """A `choices` list is either option letters (['A','B',...]) or option values
    (['0','3',...]). Letters are used directly; values map positionally to A,B,C,..."""
    items = [str(c).strip() for c in choices]
    if items and all(len(c) == 1 and c.upper() in OPTION_LETTERS for c in items):
        return [c.upper() for c in items]
    return OPTION_LETTERS[: len(items)]


_LETTER_PREFIX_RE = re.compile(r"\s*([A-Z])\s*[\.\):]\s*(.+)$")


def _strip_letter_prefix(text: str, letter: str) -> str:
    """VSI-Bench stores choices as letter-prefixed strings ('A. back-right');
    strip the positional letter prefix so option-text matching sees the bare text."""
    m = _LETTER_PREFIX_RE.match(text)
    return m.group(2).strip() if m and m.group(1) == letter else text


def _flat_letter_options(src: Dict[str, Any]) -> Dict[str, str]:
    """Options stored as per-letter keys with non-empty values (GMAI-MMBench,
    WorldMedQA-V, HRBench store msg['A']='...' with empty options/choices; an
    empty value, e.g. GMAI's E='', means the option does not exist)."""
    out = {}
    for letter in OPTION_LETTERS[:8]:
        value = src.get(letter)
        if value is None or str(value).strip() == "":
            break  # contiguous prefix only, like the inline-question parser
        out[letter] = str(value)
    return out


def parsed_options(sample: Dict[str, Any]) -> Optional[List[str]]:
    """Candidate option letters from an explicit option source, or None when the
    sample carries none (options-in-image datasets). All option sources — the
    `options` dict, the `choices` list, inline-question text and flat per-letter
    keys — are render inputs and therefore live inside messages[0]
    (dataset-standard boundary rule); only grading fields moved to the sample
    top level."""
    messages = sample.get("messages") or []
    msg = messages[0] if messages and isinstance(messages[0], dict) else None
    if not isinstance(msg, dict):
        return None

    options = msg.get("options")
    if isinstance(options, dict) and options:
        return [str(k).strip().upper() for k in options.keys()]
    choices = msg.get("choices")
    if isinstance(choices, list) and choices:
        return _choices_to_letters(choices)

    # Flat per-letter keys inside the user message (GMAI-MMBench msg["A"].."E",
    # WorldMedQA-V, HRBench) — empty values excluded.
    flat = _flat_letter_options(msg)
    if flat:
        return list(flat.keys())

    # Inline-options datasets (MMStar/BLINK keep options in the question text and
    # ship empty `options`/`choices`): derive the real letters from the question so
    # we don't fall back to A-F and mis-count stray letters (e.g. "F = ma").
    letters = _option_letters_from_question(msg.get("question") or msg.get("prompt") or "")
    if letters:
        return letters
    return None


def extract_options(sample: Dict[str, Any]) -> List[str]:
    """Candidate option letters: the sample's parsed option source, else A-F
    (some MCQ datasets, e.g. LogicVista, keep the choices in the image and don't
    list them in the question text). Whether the item is actually MCQ vs open is
    decided by infer_question_type from the ground-truth shape, not by this
    default."""
    return parsed_options(sample) or OPTION_LETTERS[:6]


def extract_option_texts(sample: Dict[str, Any], letters: List[str]) -> Dict[str, str]:
    """Map option letter -> option text, from whichever source the sample uses
    (all render inputs inside messages[0] — see extract_options)."""
    messages = sample.get("messages") or []
    msg = messages[0] if messages and isinstance(messages[0], dict) else None
    if isinstance(msg, dict):
        msg_options = msg.get("options")
        if isinstance(msg_options, dict) and msg_options:
            return {str(k).strip().upper(): str(v) for k, v in msg_options.items()}
        # BLINK-style: `choices` holds the option *values* positionally (A,B,C,...).
        msg_choices = msg.get("choices")
        if isinstance(msg_choices, list) and msg_choices:
            vals = [str(c) for c in msg_choices]
            if not all(len(v.strip()) == 1 and v.strip().upper() in OPTION_LETTERS for v in vals):
                # VSI-Bench stores letter-prefixed values ("A. back-right"); strip
                # the positional letter so text matching sees the bare option.
                return {
                    OPTION_LETTERS[i]: _strip_letter_prefix(vals[i], OPTION_LETTERS[i])
                    for i in range(min(len(vals), len(OPTION_LETTERS)))
                }

    # Flat per-letter option keys inside the user message (HRBench msg["A"]="27B",
    # GMAI-MMBench, WorldMedQA-V) with empty options/choices.
    option_texts = {}
    for letter in letters:
        if isinstance(msg, dict) and letter in msg:
            option_texts[letter] = str(msg[letter])
    return option_texts


_GT_LABEL_STRIP_RE = re.compile(r"[()\[\].:;\s\*]")
_LETTER_SEPARATOR_RE = re.compile(r"[,;/&\s]+|\band\b", re.IGNORECASE)


def parse_gt_letters(gt: Any) -> Optional[Tuple[str, ...]]:
    """MCQ ground truth -> sorted letter tuple. A single bare letter ("B", "(D)")
    gives a 1-tuple; multi-letter answers must be SEPARATED letters like "A, C" /
    "A and C" (LogicVista format), >=2 unique letters within A-H. Compact "AC" is
    deliberately NOT accepted — as a *type signal* it would misread open answers
    that happen to be A-H-only words ("BED")."""
    g = _GT_LABEL_STRIP_RE.sub("", str(gt)).upper()
    if len(g) == 1 and g in OPTION_LETTERS:
        return (g,)
    tokens = [t for t in _LETTER_SEPARATOR_RE.split(str(gt).strip()) if t]
    if len(tokens) < 2:
        return None
    letters = []
    for t in tokens:
        if len(t) == 1 and t.upper() in OPTION_LETTERS[:8]:
            letters.append(t.upper())
        else:
            return None
    if len(set(letters)) != len(letters):
        return None
    return tuple(sorted(letters))


def parse_number(value: Any) -> Optional[float]:
    """The whole (normalized) answer is a number, else None. A trailing
    percent sign reads as /100 — the official numeric semantics (ChartQA
    relaxed_correctness `_to_float`, lmms-eval chartqa/utils.py, taken from
    Qwen-VL evaluate_vqa.py:L113): "24%" is 0.24, NOT 24, so a model
    answering "24%" against gt "24" does not match."""
    text = normalize_open_answer(value).replace(",", "")
    percent = text.endswith("%")
    if percent:
        text = text.rstrip("%").strip()
    try:
        number = float(text)
    except ValueError:
        return None
    return number / 100.0 if percent else number


_WORD_RE = re.compile(r"[a-z]+")


def extract_yes_no(value: Any) -> Optional[str]:
    """VLMEvalKit YOrN_Extraction parity (vlmeval/dataset/utils/yorn.py:254-261):
    word-level; "yes" without "no" -> yes, "no" without "yes" -> no, both or
    neither -> None (upstream 'Unknown')."""
    words = set(_WORD_RE.findall(normalize_text(value, lower=True)))
    if "yes" in words and "no" not in words:
        return "yes"
    if "no" in words and "yes" not in words:
        return "no"
    return None


def infer_question_type(sample: Dict[str, Any], question_type: str = "auto", gt: Any = None) -> str:
    """Resolve the sample's question type. Order: forced CLI value -> the sample's
    standardized `question_type` field (recognized vocabulary only — mm-eval org
    datasets like VSI-Bench reuse that field for subtask names, which fall
    through) -> gt shape -> options presence."""
    forced = (question_type or "auto").strip().lower()
    if forced in QUESTION_TYPES:
        return forced

    # Per-sample field contract: exactly the four grading values. A full-org
    # sweep of every mm-eval dataset/split (577k+ rows, datasets-server
    # statistics API) found ONLY these; dataset-level task_type stays in
    # metadata, source-benchmark labels stay in `source_question_type`.
    field_value = str(get_field(sample, "question_type", "") or "").strip().lower()
    if field_value in QUESTION_TYPES:
        return field_value

    # The ground-truth shape is the most reliable signal and correctly splits
    # mixed benchmarks (RealWorldQA interleaves letter, yes/no and numeric gts;
    # GQA-Spatial mixes verify and query rows; LogicVista mixes single- and
    # multi-letter answers — both are mcq).
    if gt is not None:
        norm = normalize_open_answer(gt)
        if norm in {"yes", "no"}:
            return "yes_no"
        if parse_gt_letters(gt) is not None:
            return "mcq"
        if parse_number(gt) is not None:
            return "numeric"
        if _GT_LABEL_STRIP_RE.sub("", str(gt)) != "":
            return "open"

    # No usable gt signal (missing/blank gt — the sample is marked invalid before
    # any stage runs): default to mcq. NOTE extract_options never returns empty
    # (A-F fallback), so an "options present?" check here would be dead logic.
    return "mcq"


_STRICT_CUE_RES = (
    re.compile(r"^(?:answer|option|choice)\s*[:\-]?\s*(?-i:([A-Z]))(?:[\s\.\),:;]|$)", re.IGNORECASE),
    re.compile(r"(?:final answer|the answer is|answer is|choice is|option is)\s*[:\-]?\s*(?-i:([A-Z]))(?:[\s\.\),:;]|$)", re.IGNORECASE),
)


def extract_option_strict(text: Any, candidates: List[str]) -> Optional[str]:
    # Prefer the content of the LAST <answer>...</answer> tag if the model emitted
    # the reasoning-style "<think>...</think><answer>X</answer>" format. Otherwise
    # the buried letter gets diluted by surrounding reasoning text and the regexes
    # below — which anchor on the start of the string — miss it.
    raw_text = _to_text(text)
    ans = _ANSWER_TAG_RE.findall(raw_text)
    clean = normalize_text(ans[-1]) if ans else normalize_text(text)
    if not clean:
        return None

    stripped = clean.strip()
    uppercase = stripped.upper()
    # The whole answer is a single option letter (any case) — unambiguous.
    if uppercase in candidates:
        return uppercase
    # Labelled answer-first form "B. ..." / "B) ..." / "B: ...". Punctuation (not a
    # space) and an uppercase letter are required, so "A teddy bear" is NOT option A.
    if len(stripped) >= 2 and uppercase[0] in candidates and stripped[0].isupper() and stripped[1] in {".", ")", ":"}:
        return uppercase[0]

    # Cue phrases are case-insensitive but the letter is matched case-SENSITIVELY
    # ((?-i:...)), so "the answer is a teddy bear" never reads as option A.
    for pattern in _STRICT_CUE_RES:
        match = pattern.search(stripped)
        if match:
            candidate = match.group(1)
            if candidate in candidates:
                return candidate
    return None


# Punctuation/markdown replaced by whitespace so option letters become standalone
# tokens — VLMEvalKit's character set plus full-width CJK punctuation (without it
# "最终答案：A" never splits and the trailing letter is lost).
_CHOICE_PUNCT_TRANSLATION = str.maketrans({
    ch: " " for ch in ".()[],:;!*#{}" + "：；，。、！？「」『』（）【】《》"
})

# Explicit "final answer" style cues (English + CJK markers 答案/最终答案/选).
# Cues resolve answers even when several option letters appear in the reasoning
# (where pure token-position logic gives up and upstream defers to its GPT
# extractor). Cue phrases are case-insensitive but the option letter is matched
# case-SENSITIVELY via (?-i:...) so a lowercase article ("the answer is a teddy
# bear") is never mistaken for option "A".
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
    return text.translate(_CHOICE_PUNCT_TRANSLATION)


def can_infer_option(answer: str, choices: List[str]) -> Optional[str]:
    """VLMEvalKit-style: strip punctuation -> tokens -> the unique option letter
    appearing near the end of the answer. Case-sensitive like upstream: pass the
    ORIGINAL-case answer, or lowercase articles ("a cute cat") count as option A."""
    answer_mod = _strip_choice_punct(answer)
    splits = [x.strip() for x in answer_mod.split() if x.strip()]
    present = [c for c in choices if c in splits]
    if len(present) == 1:
        ch = present[0]
        # Near the end (the model's final pick), mirroring VLMEvalKit's last-5
        # window; additionally a leading letter in a short (<=5 token) answer
        # such as "B is the correct answer" is accepted — upstream defers those
        # to its GPT extractor, which resolves them the same way.
        if splits.index(ch) > len(splits) - 5 or (splits.count(ch) == 1 and len(splits) <= 5):
            return ch
    return None


def can_infer_text(answer: str, option_texts: Dict[str, str]) -> Optional[str]:
    """VLMEvalKit-style: the answer restates exactly one option's text. Upstream's
    length guard applies: a long CoT that merely mentions one option somewhere is
    not an answer restatement."""
    if not option_texts:
        return None
    low = answer.lower()
    if len(low) > 2 * sum(len(str(v)) for v in option_texts.values()):
        return None
    cands = [k for k, v in option_texts.items() if v and str(v).strip().lower() in low]
    if len(cands) == 1:
        return cands[0]
    return None


# extract_option_robust machinery, compiled once. Rule order matters and is
# documented on each step below.
_ANSWER_TAG_RE = re.compile(r"<answer\s*>\s*(.*?)\s*</answer\s*>", re.DOTALL | re.IGNORECASE)
_WRAPPING_BRACKETS_RE = re.compile(r"^[\(\[]|[\)\]\.]$")
_BOXED_LETTER_RE = re.compile(r"\\boxed\{\s*(?:\\[a-zA-Z]+\s*\{\s*)?[\(\*]*\s*([A-Za-z])")
_CUE_TAIL_RE = re.compile(
    r"(?:final answer|the answer is|answer is|answer\s*[:\-：]|option is|choice is)"
    r"\s*[\(（\*]{0,4}([A-Za-z])[\)）\*]{0,4}\s*[\.。!！]?\s*$",
    re.IGNORECASE,
)
_HEAD_LABEL_RE = re.compile(r"\s*\*{0,2}\(?([A-Z])\)?\s*[:\.\)]\s")
_BOLD_LABEL_RE = re.compile(r"\*\*\s*\(?([A-Z])\)?\s*[:\.]")


def extract_option_robust(
    text: Any, candidates: List[str], option_texts: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """Robust (tier-2) MCQ letter extraction for chain-of-thought output:
    VLMEvalKit's can_infer core (token position + option-text restatement)
    extended with <answer>-tag, \boxed{}, answer-cue (incl. CJK), and
    answer-label rules. Returns the extracted candidate letter or None.

    Args: the raw prediction text (str or 1-element list), the candidate
    letters, and the letter->option-text map for restatement matching.
    """
    raw = _to_text(text)
    if not raw:
        return None
    cand_set = [str(c).strip().upper() for c in (candidates or OPTION_LETTERS)]

    # 1. The whole (normalized) answer is one bare option letter, any case
    #    ("b", "(C)", "\boxed{D}") — unambiguous, accept directly.
    whole = _WRAPPING_BRACKETS_RE.sub("", normalize_text(raw).strip()).strip()
    if whole.upper() in cand_set:
        return whole.upper()

    # 2. The LAST <answer>...</answer> tag, when present, is the model's final
    #    pick — resolve inside it (bare letter, token match, text restatement).
    tags = _ANSWER_TAG_RE.findall(raw)
    if tags:
        inner_raw = tags[-1].strip()
        if inner_raw.upper() in cand_set:
            return inner_raw.upper()
        inner = can_infer_option(inner_raw, cand_set) or can_infer_text(inner_raw, option_texts or {})
        if inner:
            return inner

    # 3. The LAST \boxed{...} letter, tolerating an inner \text{}/\mathrm{}
    #    wrap and a leading '(' / '*': "$\boxed{C}$", "$\boxed{\text{C}}$".
    boxes = [b.upper() for b in _BOXED_LETTER_RE.findall(raw) if b.upper() in cand_set]
    if boxes:
        return boxes[-1]

    # 4. The LAST explicit answer cue (markdown/CJK tolerant, uppercase letter).
    cues = [c.upper() for c in _ANSWER_CUE.findall(raw) if c.upper() in cand_set]
    if cues:
        return cues[-1]

    # 5. A lowercase letter after a cue counts only when it TERMINATES the
    #    response ("the answer is b") — there it cannot be an article. Mid-text
    #    lowercase letters stay ignored ("the answer is a cute cat" is not A).
    tail = _CUE_TAIL_RE.search(raw)
    if tail and tail.group(1).upper() in cand_set:
        return tail.group(1).upper()

    # 6. VLMEvalKit core: the unique candidate letter near the end of the
    #    answer, ORIGINAL case (uppercasing first would make every article "a"
    #    a hit for option A).
    opt = can_infer_option(raw, cand_set)
    if opt:
        return opt

    # 7. Answer-label formats the position/count test drops:
    #    7a — answer-first label ("C: A man surfing") in SHORT responses only;
    #         long answers often reconsider the leading letter later, so they
    #         defer to the end-position/cue rules above.
    if len(raw.strip()) <= 80:
        head = _HEAD_LABEL_RE.match(raw)
        if head and head.group(1).upper() in cand_set:
            return head.group(1).upper()
    #    7b — a uniquely bolded option label "**D:**" anywhere (the model's
    #         final restated pick).
    bolds = {b.upper() for b in _BOLD_LABEL_RE.findall(raw) if b.upper() in cand_set}
    if len(bolds) == 1:
        return next(iter(bolds))

    # 8. Fallback: the answer restates exactly one option's text.
    return can_infer_text(raw, option_texts or {})


def _letters_from_segment(segment: str, candidates: List[str]) -> Optional[Tuple[str, ...]]:
    """A text segment that consists ONLY of candidate letters and separators
    ("A, C" / "A and C" / "AC") -> sorted unique letters; anything else None."""
    text = normalize_text(segment).strip().rstrip(".。")
    if not text:
        return None
    tokens = [t for t in _LETTER_SEPARATOR_RE.split(text) if t]
    letters: List[str] = []
    for t in tokens:
        up = t.upper()
        if len(up) == 1 and up in candidates:
            letters.append(up)
        elif 2 <= len(up) <= len(candidates) and all(c in candidates for c in up) and len(set(up)) == len(up):
            letters.extend(up)  # compact form "AC"
        else:
            return None
    if not letters or len(set(letters)) != len(letters):
        return None
    return tuple(sorted(letters))


_MULTI_ANSWER_CUE_RE = re.compile(
    r"(?:final answers?|the answers? (?:are|is)|answers? (?:are|is)|answers?\s*[:\-：]|最终答案|答案)",
    re.IGNORECASE,
)


def extract_letter_set(text: Any, candidates: List[str]) -> Optional[Tuple[str, ...]]:
    """Multi-select prediction -> sorted letter tuple. Tries, in order: the LAST
    <answer> tag, the segment after the LAST answer cue, then the whole answer.
    Only letters-and-separators segments are accepted — free text defers to the
    LLM extractor (official LogicVista protocol: GPT extracts the letters, then
    sorted set equality; vlmeval/dataset/utils/logicvista.py:50-68)."""
    raw = _to_text(text)
    if not raw:
        return None
    cand = [str(c).strip().upper() for c in (candidates or OPTION_LETTERS[:8])]

    tags = _ANSWER_TAG_RE.findall(raw)
    if tags:
        got = _letters_from_segment(tags[-1], cand)
        if got:
            return got

    cues = list(_MULTI_ANSWER_CUE_RE.finditer(raw))
    if cues:
        got = _letters_from_segment(raw[cues[-1].end():], cand)
        if got:
            return got

    return _letters_from_segment(raw, cand)
