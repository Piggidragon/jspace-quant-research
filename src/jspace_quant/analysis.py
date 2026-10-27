"""Step 2 (CPU): compute the readouts of all experimental arms and condense them into metrics.

compute()   — walks layer by layer and position chunk by position chunk and computes the logits
              of every arm; from them, per (arm, layer, position):
                * top-1 agreement, top-10 overlap and KL(P_ref || P_arm) against the reference
                * top-1 agreement with the model's own output (fidelity)
                * for eval items: the best rank of every intermediate concept
              Results are stored per arm (results/tables/arms/<split>/) and only computed for arms
              without a result; per model, output tokens and next-token NLL are stored
              (results/tables/models/).
summarize() — pass@k, hit rate per layer, workspace band, agreement curves, behavior check,
              representation similarity, concept analysis, model metadata, each with bootstrap
              CIs -> results/tables/*.csv + summary.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from jspace_quant import capture, config, metrics, readout

logger = logging.getLogger(__name__)

CHUNK = 256
SPLITS = ("evals", "heldout")
STRENGTH_BINS = ["rank 1", "rank 2-10", "rank 11-100", "rank > 100"]
FREQUENCY_BINS = ["rare", "less common", "more common", "common"]


# ---------------------------------------------------------------------------------------------
# loading, consistency checks, storage
# ---------------------------------------------------------------------------------------------

def _load_split(split: str, models: list[str]) -> tuple[dict[str, np.ndarray], list[dict]]:
    acts, metas = {}, {}
    for m in models:
        acts[m], metas[m] = capture.load(m, split)
    ref = metas[config.REFERENCE_MODEL]
    for m, meta in metas.items():
        if len(meta) != len(ref):
            raise ValueError(f"{split}: {m} has {len(meta)} positions, reference has {len(ref)}")
        key = "tokens" if split == "evals" else "token"
        if any(a[key] != b[key] for a, b in zip(meta, ref)):
            raise ValueError(f"{split}: tokenization of {m} differs from the reference")
    return acts, ref


def _concepts(meta: list[dict]) -> pd.DataFrame:
    rows = []
    for i, it in enumerate(meta):
        for c in it["intermediates"]:
            rows.append({"item": i, "set": it["set"], "name": it["name"], "word": c["word"],
                         "token_ids": c["token_ids"], "single_token": c["single_token"]})
    return pd.DataFrame(rows)


def _arm_file(split: str, arm: str) -> Path:
    safe = arm.replace("|", "__").replace("=", "-")
    return config.TABLES_DIR / "arms" / split / f"{safe}.npz"


def _model_file(split: str, model: str) -> Path:
    return config.TABLES_DIR / "models" / f"{split}_{model}.npz"


def load_arm(split: str, arm: str) -> dict[str, np.ndarray]:
    return dict(np.load(_arm_file(split, arm)))


def load_model_outputs(split: str, model: str) -> dict[str, np.ndarray]:
    return dict(np.load(_model_file(split, model)))


def available_models() -> list[str]:
    return [m for m in config.MODELS if (config.ACTIVATIONS_DIR / m / "evals.npy").exists()]


def available_arms() -> list[config.Arm]:
    """Arms whose models have been captured (in configuration order)."""
    models = set(available_models())
    return [a for a in config.ARMS if a.model in models and a.unembed in models]


# ---------------------------------------------------------------------------------------------
# readouts
# ---------------------------------------------------------------------------------------------

def _model_outputs(split: str, model: str, acts: np.ndarray, meta: list[dict]) -> None:
    """Each model's output (last block, J = I, own unembedding) and next-token NLL."""
    path = _model_file(split, model)
    if path.exists():
        return
    n = len(meta)
    top, nll = np.empty(n, dtype=np.int64), np.full(n, np.nan)
    for s in range(0, n, CHUNK):
        sl = slice(s, s + CHUNK)
        logits = readout.readout_weights(model).unembed(np.asarray(acts[sl, config.FINAL_LAYER]))
        top[sl] = logits.argmax(-1)
        if split == "heldout":
            nxt = np.array([p["next_token"] for p in meta[sl]])
            nll[sl] = -readout.log_softmax(logits)[np.arange(len(nxt)), nxt]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, out_top1=top, nll=nll)


def compute(split: str, *, force: bool = False) -> None:
    arms = available_arms()
    todo = [a for a in arms if force or not _arm_file(split, a.name).exists()]
    if not todo:
        logger.info("%s: all %d arms already computed", split, len(arms))
        return
    ref_arm = config.ARMS_BY_NAME[config.REFERENCE_ARM]
    models = sorted({ref_arm.model, *(a.model for a in todo), *(a.unembed for a in todo)},
                    key=list(config.MODELS).index)
    acts, meta = _load_split(split, models)
    n_pos = len(meta)
    for m in models:
        _model_outputs(split, m, acts[m], meta)
    out_top1 = {m: load_model_outputs(split, m)["out_top1"] for m in models}
    logger.info("%s: computing %d of %d arms over %d positions", split, len(todo), len(arms), n_pos)

    concepts = _concepts(meta) if split == "evals" else None
    concept_items = concepts["item"].to_numpy() if concepts is not None else np.empty(0, dtype=np.int64)
    concept_tokens = concepts["token_ids"].tolist() if concepts is not None else []

    shape = (config.N_LAYERS, n_pos)
    res = {a.name: {"top1_ref": np.zeros(shape, bool), "overlap_ref": np.zeros(shape, np.float32),
                    "kl_ref": np.zeros(shape, np.float32), "top1_own": np.zeros(shape, bool),
                    "top1_ids": np.zeros(shape, np.int64),
                    "ranks": np.zeros((config.N_LAYERS, len(concept_items)), np.int64)} for a in todo}

    t0 = time.time()
    for layer in config.LAYERS:
        for s in range(0, n_pos, CHUNK):
            sl = slice(s, s + CHUNK)
            h = {m: np.asarray(acts[m][sl, layer]) for m in models}
            ref_logits = readout.arm_logits(ref_arm, layer, h[ref_arm.model])
            logp_ref = readout.log_softmax(ref_logits)
            topk_ref = readout.topk_ids(ref_logits, config.TOPK)
            sel = np.flatnonzero((concept_items >= s) & (concept_items < s + CHUNK))
            for arm in todo:
                r = res[arm.name]
                logits = ref_logits if arm is ref_arm else readout.arm_logits(arm, layer, h[arm.model])
                topk = readout.topk_ids(logits, config.TOPK)
                r["top1_ids"][layer, sl] = topk[:, 0]
                r["top1_ref"][layer, sl] = topk[:, 0] == topk_ref[:, 0]
                r["overlap_ref"][layer, sl] = metrics.topk_overlap(topk, topk_ref)
                r["kl_ref"][layer, sl] = metrics.kl_divergence(logp_ref, readout.log_softmax(logits))
                r["top1_own"][layer, sl] = topk[:, 0] == out_top1[arm.model][sl]
                for ci in sel:
                    r["ranks"][layer, ci] = readout.min_rank(logits[[concept_items[ci] - s]], concept_tokens[ci])[0]
        logger.info("%s: layer %2d/%d  (%.0fs)", split, layer, config.FINAL_LAYER, time.time() - t0)

    for arm in todo:
        path = _arm_file(split, arm.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **res[arm.name])
    logger.info("%s done in %.0fs", split, time.time() - t0)


# ---------------------------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------------------------

def _boot(values: np.ndarray, seed_offset: int = 0) -> tuple[float, float]:
    return metrics.bootstrap_ci(values, n=config.BOOTSTRAP_SAMPLES, seed=config.SEED + seed_offset)


def concept_ranks(arms: list[config.Arm]) -> pd.DataFrame:
    """Long table: rank of every intermediate concept × arm × layer."""
    _, meta = capture.load(config.REFERENCE_MODEL, "evals")
    concepts = _concepts(meta)[["item", "set", "name", "word", "single_token"]]
    rows = []
    for arm in arms:
        ranks = load_arm("evals", arm.name)["ranks"]
        for layer in config.LAYERS:
            df = concepts.copy()
            df["arm"], df["layer"], df["rank"] = arm.name, layer, ranks[layer]
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def floor_sets(cr: pd.DataFrame) -> list[str]:
    """Eval sets on which the reference itself is at floor (pass@10 across all layers
    < config.FLOOR_PASS_AT_10). Quantization cannot make them worse; they are reported but excluded
    from the core aggregate and from hypothesis testing."""
    ref = cr[cr.arm == config.REFERENCE_ARM]
    best = ref.groupby(["set", "item", "word"])["rank"].min().reset_index()
    per_set = (best.assign(hit=best["rank"] <= 10).groupby(["set", "item"])["hit"].mean()
               .groupby("set").mean())
    return sorted(per_set[per_set < config.FLOOR_PASS_AT_10].index)


def _pass_at_k_tables(cr: pd.DataFrame, arms: list[config.Arm], band: tuple[int, int],
                      core: list[str]) -> pd.DataFrame:
    """pass@k per arm × set × k × layer scope, with CI and paired difference to the reference.
    Aggregates: "all" (all sets) and "core" (without floor sets)."""
    rows = []
    scopes = {config.SCOPE_ALL: (0, config.FINAL_LAYER), config.SCOPE_BAND: band}
    for scope, (lo, hi) in scopes.items():
        best = (cr[(cr.layer >= lo) & (cr.layer <= hi)]
                .groupby(["arm", "set", "item", "word"], sort=False)["rank"].min().reset_index())
        for k in config.PASS_AT_K:
            best["hit"] = (best["rank"] <= k).astype(float)
            per_item = best.groupby(["arm", "set", "item"])["hit"].mean().unstack("arm")
            for eval_set in config.EVAL_SETS + [config.AGG_ALL, config.AGG_CORE]:
                if eval_set == config.AGG_ALL:
                    sub = per_item
                elif eval_set == config.AGG_CORE:
                    sub = per_item.loc[per_item.index.get_level_values("set").isin(core)]
                else:
                    sub = per_item.loc[eval_set]
                if sub.empty:
                    continue
                ref = sub[config.REFERENCE_ARM].to_numpy()
                for arm in arms:
                    v = sub[arm.name].to_numpy()
                    lo_ci, hi_ci = _boot(v)
                    d_lo, d_hi = _boot(v - ref, 1)
                    rows.append({"scope": scope, "set": eval_set, "k": k, "arm": arm.name, "model": arm.model,
                                 "role": arm.role, "pass_at_k": v.mean(), "ci_lo": lo_ci, "ci_hi": hi_ci,
                                 "delta_ref": (v - ref).mean(), "delta_ci_lo": d_lo, "delta_ci_hi": d_hi,
                                 "n_items": len(v)})
    return pd.DataFrame(rows)


def _clusters(split: str) -> np.ndarray:
    _, meta = capture.load(config.REFERENCE_MODEL, split)
    return np.array([p["paragraph"] for p in meta]) if split == "heldout" else np.arange(len(meta))


def _layer_curves(split: str, arms: list[config.Arm]) -> pd.DataFrame:
    clusters = _clusters(split)
    rows = []
    for arm in arms:
        d = load_arm(split, arm.name)
        for layer in config.LAYERS:
            for metric in ("top1_ref", "overlap_ref", "kl_ref", "top1_own"):
                v = d[metric][layer].astype(np.float64)
                lo, hi = metrics.cluster_bootstrap_ci(v, clusters, n=config.BOOTSTRAP_SAMPLES, seed=config.SEED)
                rows.append({"split": split, "arm": arm.name, "model": arm.model, "role": arm.role,
                             "layer": layer, "metric": metric, "mean": v.mean(), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def _band_agreement(split: str, arms: list[config.Arm], band: tuple[int, int]) -> pd.DataFrame:
    """Mean over the workspace band per arm; CI by bootstrapping over paragraphs or items (first
    average each position over the band layers, then bootstrap)."""
    clusters = _clusters(split)
    rows = []
    for arm in arms:
        d = load_arm(split, arm.name)
        row = {"split": split, "arm": arm.name, "model": arm.model, "role": arm.role}
        for metric in ("top1_ref", "overlap_ref", "kl_ref", "top1_own"):
            v = d[metric][band[0]: band[1] + 1].astype(np.float64).mean(axis=0)
            lo, hi = metrics.cluster_bootstrap_ci(v, clusters, n=config.BOOTSTRAP_SAMPLES, seed=config.SEED)
            row |= {metric: v.mean(), f"{metric}_ci_lo": lo, f"{metric}_ci_hi": hi}
        rows.append(row)
    return pd.DataFrame(rows)


def summarize() -> dict:
    T = config.TABLES_DIR
    arms = available_arms()
    cr = concept_ranks(arms)
    # fixed gzip mtime, so an unchanged table stays byte-identical
    cr.to_csv(T / "concept_ranks.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})

    # hit rate per layer (rank ≤ 10 in exactly this layer), pooled over all concepts
    hit = (cr.assign(hit=cr["rank"] <= 10).groupby(["arm", "set", "layer"])["hit"].mean().reset_index())
    pooled = cr.assign(hit=cr["rank"] <= 10).groupby(["arm", "layer"])["hit"].mean().reset_index()
    pooled["set"] = config.AGG_ALL
    hit = pd.concat([hit, pooled])
    hit.to_csv(T / "hit_rate_per_layer.csv", index=False)

    ref_curve = pooled[pooled.arm == config.REFERENCE_ARM].sort_values("layer")["hit"].to_numpy()
    band = metrics.workspace_band(ref_curve)
    logger.info("workspace band (reference hit rate ≥ 50 %% of peak): L%d-L%d", *band)

    floor = floor_sets(cr)
    core = [s for s in config.EVAL_SETS if s not in floor]
    logger.info("floor sets (reference pass@10 < %.2f): %s", config.FLOOR_PASS_AT_10, floor or "none")
    _pass_at_k_tables(cr, arms, band, core).to_csv(T / "pass_at_k.csv", index=False)

    pd.concat([_layer_curves(s, arms) for s in SPLITS]).to_csv(T / "agreement_per_layer.csv", index=False)
    pd.concat([_band_agreement(s, arms, band) for s in SPLITS]).to_csv(T / "agreement_band.csv", index=False)

    behavior().to_csv(T / "behavior.csv", index=False)
    model_table().to_csv(T / "models.csv", index=False)
    representation_similarity().to_csv(T / "representation_similarity.csv", index=False)
    concept_effects(cr, band).to_csv(T / "concept_effects.csv", index=False)
    iv = intervention_summary()
    if iv is not None:
        iv.to_csv(T / "interventions_summary.csv", index=False)

    summary = {"workspace_band": list(band), "floor_sets": floor, "core_sets": core,
               "n_concepts": int(cr.groupby(["item", "word"]).ngroups),
               "single_token_share": float(cr.drop_duplicates(["item", "word"])["single_token"].mean()),
               "models": available_models(), "n_arms": len(arms)}
    (T / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------------------------
# behavior check, model metadata
# ---------------------------------------------------------------------------------------------

def behavior() -> pd.DataFrame:
    """Behavior check: answer accuracy (greedy) per model/set and held-out perplexity.

    Besides each model's value, the paired difference to the reference (same items or positions):
    accuracy as a difference, perplexity as a ratio exp(mean NLL difference); CIs by bootstrapping
    over items or paragraphs.
    """
    models = available_models()
    paragraphs = _clusters("heldout")
    ref = config.REFERENCE_MODEL

    correct: dict[tuple[str, str], np.ndarray] = {}
    for m in models:
        _, meta = capture.load(m, "evals")
        for eval_set in config.EVAL_SETS:
            items = [it for it in meta if it["set"] == eval_set and it["target"] is not None]
            if items:
                correct[m, eval_set] = np.array(
                    [it["generated"].strip().lower().startswith(it["target"].strip().lower()) for it in items],
                    dtype=float)

    def cboot(v: np.ndarray) -> tuple[float, float]:
        return metrics.cluster_bootstrap_ci(v, paragraphs, n=config.BOOTSTRAP_SAMPLES, seed=config.SEED)

    nll_ref = load_model_outputs("heldout", ref)["nll"]
    rows = []
    for m in models:
        for (mm, eval_set), c in correct.items():
            if mm != m:
                continue
            lo, hi = _boot(c)
            d = c - correct[ref, eval_set]
            d_lo, d_hi = _boot(d, 1)
            rows.append({"model": m, "metric": f"accuracy_{eval_set}", "value": c.mean(), "ci_lo": lo,
                         "ci_hi": hi, "vs_ref": d.mean(), "vs_ref_ci_lo": d_lo, "vs_ref_ci_hi": d_hi,
                         "n": len(c)})
        nll = load_model_outputs("heldout", m)["nll"]
        lo, hi = cboot(nll)
        diff = nll - nll_ref
        d_lo, d_hi = cboot(diff)
        rows.append({"model": m, "metric": "heldout_ppl", "value": float(np.exp(nll.mean())),
                     "ci_lo": float(np.exp(lo)), "ci_hi": float(np.exp(hi)),
                     "vs_ref": float(np.exp(diff.mean())), "vs_ref_ci_lo": float(np.exp(d_lo)),
                     "vs_ref_ci_hi": float(np.exp(d_hi)), "n": len(nll)})
    return pd.DataFrame(rows)


def model_table() -> pd.DataFrame:
    """Per model: family, type distribution of the weight tensors, bits per weight (from the GGUF)."""
    from gguf import GGUFReader

    rows = []
    for m in available_models():
        r = GGUFReader(str(config.MODELS[m]))
        weights = [t for t in r.tensors if len(t.shape) > 1]  # without 1-D norm weights
        n = sum(int(t.n_elements) for t in r.tensors)
        nbytes = sum(int(t.n_bytes) for t in r.tensors)
        types = pd.Series([t.tensor_type.name for t in weights]).value_counts().to_dict()
        info = config.MODEL_INFO[m]
        rows.append({"model": m, "label": info.label, "family": info.family, "params": n,
                     "bits_per_weight": 8 * nbytes / n, "tensor_types": json.dumps(types)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# extension A: representations without a lens
# ---------------------------------------------------------------------------------------------

def representation_similarity() -> pd.DataFrame:
    """Per model and layer, against BF16 on identical positions: relative error
    ‖h_q − h_ref‖/‖h_ref‖ and cosine (mean over positions, CI by cluster bootstrap) and linear CKA
    over all positions. Independent of any lens."""
    rows = []
    for split in SPLITS:
        clusters = _clusters(split)
        ref, _ = capture.load(config.REFERENCE_MODEL, split)
        for m in available_models():
            if m == config.REFERENCE_MODEL:
                continue
            acts, _ = capture.load(m, split)
            for layer in config.LAYERS:
                a = np.asarray(ref[:, layer], dtype=np.float64)
                b = np.asarray(acts[:, layer], dtype=np.float64)
                na = np.linalg.norm(a, axis=1)
                rel = np.linalg.norm(b - a, axis=1) / na
                cos = (a * b).sum(axis=1) / (na * np.linalg.norm(b, axis=1))
                rlo, rhi = metrics.cluster_bootstrap_ci(rel, clusters, n=config.BOOTSTRAP_SAMPLES, seed=config.SEED)
                rows.append({"split": split, "model": m, "layer": layer, "rel_error": rel.mean(),
                             "rel_error_ci_lo": rlo, "rel_error_ci_hi": rhi, "cosine": cos.mean(),
                             "cka": metrics.linear_cka(a, b)})
        logger.info("representation similarity %s done", split)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# extension B: which concepts suffer?
# ---------------------------------------------------------------------------------------------

def concept_effects(cr: pd.DataFrame, band: tuple[int, int]) -> pd.DataFrame:
    """Per intermediate concept and arm (fixed lens, refit, noise): best rank in the workspace band
    compared with the reference, plus concept properties for the breakdown:

    * ref_strength — how well the reference reads it (rank 1 / 2-10 / 11-100 / > 100)
    * frequency    — frequency of the candidate token in fitting corpus A (quartile; proxy for
                     "rare vs. common"), from results/tables/token_counts_corpusA.npz
    """
    inband = cr[(cr.layer >= band[0]) & (cr.layer <= band[1])]
    best = inband.groupby(["arm", "item", "set", "name", "word"], sort=False)["rank"].min().unstack("arm")
    ref = best[config.REFERENCE_ARM]
    counts = _token_counts()
    _, meta = capture.load(config.REFERENCE_MODEL, "evals")
    freq = {}
    for i, it in enumerate(meta):
        for c in it["intermediates"]:
            freq[(i, c["word"])] = max(int(counts[t]) for t in c["token_ids"]) if counts is not None else np.nan
    rows = []
    for arm in config.ARMS:
        if arm.role not in ("fixed", "refit", "noise") or arm.name not in best:
            continue
        df = pd.DataFrame({"ref_rank": ref, "arm_rank": best[arm.name]}).reset_index()
        df["arm"], df["model"], df["role"] = arm.name, arm.model, arm.role
        df["delta_log10_rank"] = np.log10(df["arm_rank"]) - np.log10(df["ref_rank"])
        df["lost"] = (df["ref_rank"] <= 10) & (df["arm_rank"] > 10)
        df["gained"] = (df["ref_rank"] > 10) & (df["arm_rank"] <= 10)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["ref_strength"] = pd.cut(out["ref_rank"], [0, 1, 10, 100, np.inf], labels=STRENGTH_BINS)
    out["frequency"] = [freq.get((i, w), np.nan) for i, w in zip(out["item"], out["word"])]
    if out["frequency"].notna().any():
        out["frequency_q"] = pd.qcut(out["frequency"].rank(method="first"), 4, labels=FREQUENCY_BINS)
    return out


def _token_counts() -> np.ndarray | None:
    path = config.TABLES_DIR / "token_counts_corpusA.npz"
    if not path.exists():
        logger.warning("no token frequencies (%s) — frequency analysis skipped", path)
        return None
    d = np.load(path)
    counts = np.zeros(int(d["n_vocab"]), dtype=np.int64)
    counts[d["ids"]] = d["counts"]
    return counts


# ---------------------------------------------------------------------------------------------
# causal test: interventions with lens directions (intervene.py)
# ---------------------------------------------------------------------------------------------

def intervention_summary() -> pd.DataFrame | None:
    """Per model × lens: rates under each condition with bootstrap CIs, plus the paired difference
    to the reference arm (BF16, lens A) on the same items.

    steer_success   share of items whose next token switches to swap_answer under steering
    steer_control   the same share when steering towards an unrelated concept
    swap_success    share switching to swap_answer under the coordinate swap
    ablate_drop     share of correct answers lost by ablating the intermediate concept
    ablate_control  share lost by ablating an unrelated concept
    """
    d = config.TABLES_DIR / "interventions"
    files = sorted(d.glob("*.csv")) if d.exists() else []
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.concept_tokens_ok]
    base = df[df.condition == "baseline"].set_index(["model", "item"])["answer"].astype(float)
    measures = {
        "steer_success": ("steer", "swap_answer", False), "steer_control": ("steer_control", "swap_answer", False),
        "swap_success": ("swap", "swap_answer", False), "steer_says_concept": ("steer", "says_concept", False),
        "ablate_drop": ("ablate", "answer", True), "ablate_control": ("ablate_control", "answer", True),
    }
    per_item = {}
    for (model, lens), g in df[df.condition != "baseline"].groupby(["model", "lens"]):
        cols = {}
        for name, (cond, col, drop) in measures.items():
            v = g[g.condition == cond].set_index("item")[col].astype(float)
            cols[name] = base.loc[model].reindex(v.index) - v if drop else v
        per_item[model, lens] = pd.DataFrame(cols)
    ref = per_item[config.REFERENCE_MODEL, config.REFERENCE_LENS]
    rows = []
    for (model, lens), t in per_item.items():
        row = {"model": model, "lens": lens, "n_items": len(t),
               "baseline_accuracy": base.loc[model].reindex(t.index).mean()}
        for name in measures:
            v = t[name].to_numpy()
            lo, hi = _boot(v)
            diff = v - ref[name].reindex(t.index).to_numpy()
            d_lo, d_hi = _boot(diff, 1)
            row |= {name: v.mean(), f"{name}_ci_lo": lo, f"{name}_ci_hi": hi,
                    f"{name}_vs_ref": diff.mean(), f"{name}_vs_ref_ci_lo": d_lo, f"{name}_vs_ref_ci_hi": d_hi}
        rows.append(row)
    return pd.DataFrame(rows)
