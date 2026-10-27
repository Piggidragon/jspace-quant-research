"""Causal test (GPU): do lens directions still *act* on quantized models, not just read?

Uses the paper's `probe-swap` set (anthropics/jacobian-lens, data/experiments): 90 two-hop prompts,
each with a bridge concept (`intermediate`, e.g. "Brazil" in "the language spoken in the country
where the Amazon River ends is"), its `answer` ("Portuguese"), a replacement concept (`swap_to`,
"Mexico") and the answer it should lead to (`swap_answer`, "Spanish").

Per item and arm (model × lens) the model's next token at the last prompt position is recorded under
these conditions, applied at every prompt position:

    baseline        no intervention
    steer           at STEER_LAYERS add α·h_rms·(v̂_swap_to − v̂_intermediate)
                                                     -> does the answer follow the new concept?
    steer_control   the same towards an unrelated concept from another item
    swap            the paper's concept patch in the workspace band: swap the lens coordinates of
                    intermediate and swap_to (rank 2: h' = h + V(σ(c) − c), c = V⁺h)
    ablate          at STEER_LAYERS remove the intermediate's lens direction (rank-1 projection)
                                                     -> does the correct answer break?
    ablate_control  remove an unrelated concept's direction

v̂_t is the lens direction of token t at a layer: the pull-back Jᵀu_t of its (norm-scaled)
unembedding row, as in jlens-gguf (LensReadout.lens_vector/steer_vector/swap_factors/
ablate_factors). It uses the unembedding of the model being intervened on, as when deploying the lens
on that model. Steering layers and strength were chosen in a pilot on BF16 only, before any quantized
model was run (docs/en/04-decisions.md §7). Results: results/tables/interventions/<model>.csv
"""

from __future__ import annotations

import json
import logging
import time
from functools import cache

import numpy as np
import pandas as pd
from jlens_gguf.client import NativeClient
from jlens_gguf.readout import LensReadout

from jspace_quant import capture, config, data, readout

logger = logging.getLogger(__name__)

CONDITIONS = ("baseline", "steer", "steer_control", "swap", "ablate", "ablate_control")


def load_items() -> list[dict]:
    path = config.EVAL_DATA_DIR.parent / "experiments" / "probe-swap.json"
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    return items[: config.SMOKE_ITEMS_PER_SET] if config.SMOKE else items


def concept_token(word: str, tokenize) -> int | None:
    """The single token that represents a concept word (the form with a leading space first, as
    the word would appear mid-sentence); None if no spelling is a single token."""
    for form in (" " + word, word, " " + word.lower(), word.lower(), " " + word.capitalize()):
        toks = tokenize(form)
        if len(toks) == 1:
            return toks[0]
    return None


def control_concept(items: list[dict], i: int) -> str:
    """An unrelated concept for item i: the swap_to of the next item whose concepts differ."""
    own = {items[i]["intermediate"].lower(), items[i]["swap_to"].lower()}
    for k in range(1, len(items)):
        cand = items[(i + k) % len(items)]["swap_to"]
        if cand.lower() not in own:
            return cand
    raise ValueError("no control concept found")


def band_layers() -> list[int]:
    """Workspace band from the main analysis (results/tables/summary.json)."""
    summary = json.loads((config.TABLES_DIR / "summary.json").read_text())
    lo, hi = summary["workspace_band"]
    return list(range(lo, hi + 1))


@cache
def _lens_readout(model: str, lens: str) -> LensReadout:
    return LensReadout(readout.readout_weights(model), readout.lens(lens))


def lowrank(ro: LensReadout, kind: str, layers: list[int], a: int, b: int | None = None) -> list[dict]:
    """Native low-rank edits (swap of two lens coordinates, or ablation of one direction)."""
    if kind == "swap" and b is None:
        raise ValueError("swap needs two concept tokens")
    out = []
    for layer in layers:
        A, B = ro.swap_factors(layer, a, b) if kind == "swap" and b is not None else ro.ablate_factors(layer, [a])
        out.append({"layer": layer, "pos_start": 0, "pos_end": -1, "mode": "lowrank", "a": A, "b": B})
    return out


def steer(ro: LensReadout, layers: list[int], away: int, towards: int, alpha: float) -> list[dict]:
    """Additive edits α·h_rms·(v̂_towards − v̂_away); h_rms is the lens's typical residual norm."""
    return [{"layer": l, "pos_start": 0, "pos_end": -1, "mode": "add",
             "vector": ro.steer_vector(l, towards, alpha) - ro.steer_vector(l, away, alpha)} for l in layers]


def _next_token(client: NativeClient, tokens: list[int], ivs: list[dict] | None) -> np.ndarray:
    fr = client.forward(tokens, capture=False, logits_positions=[len(tokens) - 1], interventions=ivs)
    return readout.log_softmax(fr.logits[len(tokens) - 1][None])[0]


def run_model(client: NativeClient, model: str) -> pd.DataFrame:
    items = load_items()
    tok = capture._tokenize_plain(client)
    prefix = capture.chat_prefix(client)
    band = band_layers()
    layers = list(config.STEER_LAYERS)
    rows = []
    t0 = time.time()
    for i, it in enumerate(items):
        prompt = prefix + data.model_prompt(it["prompt"])
        tokens = client.tokenize(prompt, add_special=True, parse_special=True)
        ans = data.candidate_tokens(it["answer"], "probe-swap", tok).token_ids
        swp = data.candidate_tokens(it["swap_answer"], "probe-swap", tok).token_ids
        ctrl = control_concept(items, i)
        c_int, c_swap, c_ctrl = (concept_token(w, tok) for w in (it["intermediate"], it["swap_to"], ctrl))
        base = {"model": model, "item": it["name"], "category": it["category"], "control_concept": ctrl,
                "concept_tokens_ok": None not in (c_int, c_swap, c_ctrl)}

        def record(lens: str, condition: str, logp: np.ndarray) -> None:
            top = int(logp.argmax())
            rows.append(base | {"lens": lens, "condition": condition, "top1": top,
                                "answer": top in ans, "swap_answer": top in swp,
                                "says_concept": top in (c_swap, c_ctrl),
                                "logp_answer": float(logp[ans].max()), "logp_swap_answer": float(logp[swp].max())})

        record("-", "baseline", _next_token(client, tokens, None))
        if c_int is None or c_swap is None or c_ctrl is None:
            continue
        for lens in config.INTERVENTION_LENSES[model]:
            ro = _lens_readout(model, lens)
            plan = {
                "steer": steer(ro, layers, c_int, c_swap, config.STEER_ALPHA),
                "steer_control": steer(ro, layers, c_int, c_ctrl, config.STEER_ALPHA),
                "swap": lowrank(ro, "swap", band, c_int, c_swap),
                "ablate": lowrank(ro, "ablate", layers, c_int),
                "ablate_control": lowrank(ro, "ablate", layers, c_ctrl),
            }
            for condition, ivs in plan.items():
                record(lens, condition, _next_token(client, tokens, ivs))
    logger.info("%s: %d items, %d rows  %.0fs", model, len(items), len(rows), time.time() - t0)
    # each model's unembedding is ~1.3 GB in fp32 (twice with the lens readout): release it before
    # the next model, otherwise memory grows with every model of a sweep
    _lens_readout.cache_clear()
    readout.readout_weights.cache_clear()
    return pd.DataFrame(rows)


def run(model: str, *, gpu: str = config.SERVER_GPU, port: int = config.SERVER_PORT) -> None:
    from jspace_quant.server import native_server

    log_dir = config.RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with native_server(config.MODELS[model], gpu=gpu, port=port,
                       log_path=log_dir / f"server-intervene-{model}.log") as client:
        df = run_model(client, model)
    out = config.TABLES_DIR / "interventions"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{model}.csv", index=False)


def is_done(model: str) -> bool:
    return (config.TABLES_DIR / "interventions" / f"{model}.csv").exists()
