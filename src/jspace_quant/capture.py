"""Step 1 (GPU): capture each model's residual stream at all readout positions.

For each model, two file pairs are written to results/activations/<model>/:

    evals.npy   + evals.json    — [n_items, 36, 2560] float32 at the readout position of each eval
                                  item; JSON with tokens, position, candidate tokens of the
                                  intermediate concepts and the greedy continuation (behavior check)
    heldout.npy + heldout.json  — [n_pos, 36, 2560] float32 at held-out WikiText positions, plus
                                  the respective next token (for perplexity)

All lens and metric computations then run offline on these files (analysis.py): every model is run
exactly once, and all experimental arms see identical activations.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
from jlens_gguf.client import NativeClient

from jspace_quant import config, data

logger = logging.getLogger(__name__)

N_GENERATE = 8  # tokens of greedy continuation for the behavior check


def _tokenize_plain(client: NativeClient):
    cache: dict[str, list[int]] = {}

    def tok(s: str) -> list[int]:
        if s not in cache:
            cache[s] = client.tokenize(s, add_special=False, parse_special=False)
        return cache[s]

    return tok


def chat_prefix(client: NativeClient) -> str:
    """Chat template up to the start of the assistant's answer (see config.PROMPT_FORMAT)."""
    if config.PROMPT_FORMAT == "raw":
        return ""
    return client.apply_template([{"role": "user", "content": config.CHAT_INSTRUCTION}], add_assistant=True)


def capture_evals(client: NativeClient) -> tuple[np.ndarray, list[dict]]:
    tok = _tokenize_plain(client)
    prefix = chat_prefix(client)
    acts, meta = [], []
    for eval_set in config.EVAL_SETS:
        items = data.load_eval_set(eval_set)
        t0 = time.time()
        for it in items:
            # same tokenization as in fitting (BOS, special tokens parsed)
            prompt = prefix + data.model_prompt(it.prompt)
            tokens, pieces = client.tokenize_with_pieces(prompt, add_special=True, parse_special=True)
            pos = data.readout_position(eval_set, pieces)
            fr = client.forward(tokens[: pos + 1], capture_layers=config.LAYERS, dtype="f32")
            acts.append(np.stack([fr.activations[l][pos] for l in config.LAYERS]))
            generated = None
            if it.target is not None:
                g = client.forward(tokens, capture=False, n_predict=N_GENERATE)
                generated = g.generated_text
            meta.append({
                "set": eval_set, "name": it.name, "prompt": prompt, "target": it.target,
                "tokens": tokens, "pos": pos, "readout_piece": pieces[pos],
                "intermediates": [vars(data.candidate_tokens(w, eval_set, tok)) for w in it.intermediates],
                "generated": generated,
            })
        logger.info("%-12s %3d items  %.1fs", eval_set, len(items), time.time() - t0)
    return np.stack(acts).astype(np.float32), meta


def capture_heldout(client: NativeClient) -> tuple[np.ndarray, list[dict]]:
    acts, meta = [], []
    paragraphs = data.heldout_paragraphs()
    for i, text in enumerate(paragraphs):
        tokens = client.tokenize(text)[: config.HELDOUT_MAX_TOKENS]
        positions = data.heldout_positions(len(tokens))
        fr = client.forward(tokens, capture_layers=config.LAYERS, dtype="f32")
        for p in positions:
            acts.append(np.stack([fr.activations[l][p] for l in config.LAYERS]))
            meta.append({"paragraph": i, "pos": p, "token": tokens[p], "next_token": tokens[p + 1]})
    logger.info("held-out: %d positions from %d paragraphs", len(acts), len(paragraphs))
    return np.stack(acts).astype(np.float32), meta


def token_counts(client: NativeClient) -> None:
    """Token frequencies in fitting corpus A (proxy for how common a concept token is, used in the
    concept analysis). Model-independent, since all models share the tokenizer."""
    path = config.TABLES_DIR / "token_counts_corpusA.npz"
    if path.exists():
        return
    text = (config.ROOT / "data" / "corpus" / "wikitext103-A-n1000.txt").read_text(encoding="utf-8")
    counts: dict[int, int] = {}
    for para in text.split("\n\n"):
        for t in client.tokenize(para, add_special=False, parse_special=False):
            counts[t] = counts.get(t, 0) + 1
    ids = np.array(sorted(counts), dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=ids, counts=np.array([counts[i] for i in ids], dtype=np.int64),
                        n_vocab=client.props()["n_vocab"])
    logger.info("token frequencies corpus A: %d types, %d tokens", len(ids), sum(counts.values()))


def save(model: str, name: str, acts: np.ndarray, meta: list[dict]) -> None:
    out = config.ACTIVATIONS_DIR / model
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{name}.npy", acts)
    (out / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("saved: %s/%s.npy %s", out, name, acts.shape)


def load(model: str, name: str) -> tuple[np.ndarray, list[dict]]:
    d = config.ACTIVATIONS_DIR / model
    acts = np.load(d / f"{name}.npy", mmap_mode="r")
    meta = json.loads((d / f"{name}.json").read_text(encoding="utf-8"))
    return acts, meta


def run(model: str, *, gpu: str = config.SERVER_GPU, port: int = config.SERVER_PORT) -> None:
    from jspace_quant.server import native_server

    log_dir = config.RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with native_server(config.MODELS[model], gpu=gpu, port=port,
                       log_path=log_dir / f"server-capture-{model}.log") as client:
        props = client.props()
        if props["n_layer"] != config.N_LAYERS or props["n_embd"] != config.D_MODEL:
            raise RuntimeError(f"unexpected model shape: {props}")
        save(model, "evals", *capture_evals(client))
        save(model, "heldout", *capture_heldout(client))
        token_counts(client)


def is_captured(model: str) -> bool:
    d = config.ACTIVATIONS_DIR / model
    return all((d / f"{s}.{ext}").exists() for s in ("evals", "heldout") for ext in ("npy", "json"))


def run_parallel(models: list[str], gpus: list[str]) -> None:
    """One model per GPU at a time; each GPU takes the next pending model."""
    import queue
    from concurrent.futures import ThreadPoolExecutor

    slots: queue.Queue = queue.Queue()
    for i, g in enumerate(gpus):
        slots.put((g, config.SERVER_PORT + i))

    def job(model: str) -> None:
        gpu, port = slots.get()
        try:
            run(model, gpu=gpu, port=port)
        finally:
            slots.put((gpu, port))

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for f in [pool.submit(job, m) for m in models]:
            f.result()  # propagate errors
