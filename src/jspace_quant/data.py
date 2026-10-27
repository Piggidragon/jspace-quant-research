"""Input data: the paper's eval sets, readout positions, candidate tokens, held-out text."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from jspace_quant import config


@dataclass
class EvalItem:
    set: str
    name: str
    prompt: str
    intermediates: list[str]
    target: str | None = None


def load_eval_set(name: str) -> list[EvalItem]:
    data = json.loads((config.EVAL_DATA_DIR / f"lens-eval-{name}.json").read_text(encoding="utf-8"))
    items = [EvalItem(name, it["name"], it["prompt"], list(it["intermediates"]), it.get("target"))
             for it in data["items"]]
    return items[: config.SMOKE_ITEMS_PER_SET] if config.SMOKE else items


def model_prompt(prompt: str) -> str:
    """The prompt as it goes to the model: without trailing spaces.

    Many eval prompts end in a space ("… is the "). Spark tokenizes that as a lone space token,
    whereas in running text the space is attached to the next word (" Atlantic"). The readout
    position would be a token that practically never occurs; the model demonstrably produces
    nonsense there. Without the space the readout position is the last real token, still directly
    before `target`.
    """
    return prompt.rstrip(" ")


def readout_position(eval_set: str, pieces: list[str]) -> int:
    """Readout position per data/evaluations/README.md: for `poetry` the last newline token (end
    of line 1 of the couplet), otherwise the last prompt token (directly before `target`)."""
    if eval_set == "poetry":
        newline = [i for i, p in enumerate(pieces) if "\n" in p]
        if not newline:
            raise ValueError("poetry prompt without a newline token")
        return newline[-1]
    return len(pieces) - 1


# --- candidate tokens for an intermediate concept -------------------------------------------
#
# The paper scores the lens rank of a concept word. A word corresponds to different tokens
# depending on spelling (" Brazil" vs. "Brazil"); per concept we take the minimum rank over all
# single-token spellings. For order-ops the paper expands each concept into a synonym set (numbers:
# digit + number word; operations: symbol + word forms). The exact list is not published;
# ORDER_OPS_SYNONYMS is our reconstruction.

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen"]
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty"}


def number_word(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens * 10] + (f"-{_ONES[ones]}" if ones else "")


ORDER_OPS_SYNONYMS: dict[str, list[str]] = {
    "addition": ["+", "plus", "add", "adding", "addition", "sum"],
    "subtraction": ["-", "minus", "subtract", "subtracting", "subtraction", "difference"],
    "multiplication": ["*", "×", "times", "multiply", "multiplied", "multiplication", "product"],
    "division": ["/", "÷", "divide", "divided", "division", "quotient"],
    "mod": ["%", "mod", "modulo", "remainder"],
    "squared": ["²", "squared", "square"],
}


def surface_forms(word: str, eval_set: str) -> list[str]:
    """All spellings whose tokens count as a hit for `word`."""
    if eval_set == "order-ops":
        base = [word, number_word(int(word))] if word.isdigit() else ORDER_OPS_SYNONYMS[word]
    else:
        base = [word]
    forms: list[str] = []
    for b in base:
        for v in (b, b.lower(), b.capitalize()):
            for s in (v, " " + v):
                if s not in forms:
                    forms.append(s)
    return forms


@dataclass
class Candidates:
    word: str
    token_ids: list[int]
    single_token: bool  # False: no single-token form, fall back to the first token of " word"
    forms: list[str] = field(default_factory=list)


def candidate_tokens(word: str, eval_set: str, tokenize: Callable[[str], list[int]]) -> Candidates:
    forms = surface_forms(word, eval_set)
    ids: list[int] = []
    for f in forms:
        toks = tokenize(f)
        if len(toks) == 1 and toks[0] not in ids:
            ids.append(toks[0])
    if ids:
        return Candidates(word, ids, True, forms)
    return Candidates(word, [tokenize(" " + word)[0]], False, forms)


# --- held-out text ---------------------------------------------------------------------------

def heldout_paragraphs() -> list[str]:
    """WikiText-103 paragraphs that occur in no fitting corpus (see config.HELDOUT_*)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    shard = hf_hub_download("Salesforce/wikitext", config.WIKITEXT_SHARD, repo_type="dataset",
                            revision=config.WIKITEXT_REVISION)
    texts = pq.read_table(shard, columns=["text"]).column("text").to_pylist()
    out = []
    for t in texts[config.HELDOUT_FIRST_ROW:]:
        t = " ".join(t.split())
        if len(t) >= config.HELDOUT_MIN_CHARS:
            out.append(t)
            if len(out) == config.HELDOUT_N_PARAGRAPHS:
                break
    return out


def heldout_positions(n_tokens: int) -> list[int]:
    """Positions as in fitting (sinks and the last position excluded), every STRIDE-th."""
    return list(range(config.HELDOUT_SKIP_FIRST, n_tokens - 1, config.HELDOUT_STRIDE))
