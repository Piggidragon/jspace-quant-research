"""Central configuration: paths, models, lenses, experimental arms, constants."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = ROOT / "models"
LENSES_DIR = ROOT / "lenses"

# Smoke mode (JSPACE_SMOKE=1): a few items, separate output directory — for testing the pipeline
# without touching the actual results.
SMOKE = os.environ.get("JSPACE_SMOKE") == "1"
RESULTS_DIR = ROOT / "results" / ("smoke" if SMOKE else "")
ACTIVATIONS_DIR = RESULTS_DIR / "activations"  # large, not versioned
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
SMOKE_ITEMS_PER_SET = 4
EVAL_DATA_DIR = ROOT / "repos" / "jacobian-lens" / "data" / "evaluations"
JLENS_SERVER_BIN = ROOT / "repos" / "jlens-gguf" / "native" / "jlens-server"

# Spark-X2.6-4B (spark2_5): 36 blocks, residual width 2560. Layer 35 is the last block; reading it
# with the identity is exactly the model's output (final norm + unembedding).
N_LAYERS = 36
D_MODEL = 2560
FINAL_LAYER = N_LAYERS - 1
LAYERS = list(range(N_LAYERS))


@dataclass(frozen=True)
class ModelInfo:
    file: str
    family: str  # ref | legacy | kquant | ablation
    label: str


# Flat quants only (every weight tensor has the same type), see docs/en/04-decisions.md.
# Main study: bf16, q8_0, q4_0. Extension (docs/en/02-methodology.md §6): bit-width ladder in two
# quant families and a component ablation (only one part of the model in Q4_0), created with
# scripts/make_quant_variants.sh.
MODEL_INFO: dict[str, ModelInfo] = {
    "bf16": ModelInfo("Spark-X2.6-4B.gguf", "ref", "BF16"),
    "q8_0": ModelInfo("Spark-X2.6-4B-Q8_0.gguf", "legacy", "Q8_0"),
    "q5_0": ModelInfo("Spark-X2.6-4B-Q5_0-pure.gguf", "legacy", "Q5_0"),
    "q4_0": ModelInfo("Spark-X2.6-4B-Q4_0-pure.gguf", "legacy", "Q4_0"),
    "q6_k": ModelInfo("Spark-X2.6-4B-Q6_K-pure.gguf", "kquant", "Q6_K"),
    "q5_k": ModelInfo("Spark-X2.6-4B-Q5_K-pure.gguf", "kquant", "Q5_K"),
    "q4_k": ModelInfo("Spark-X2.6-4B-Q4_K-pure.gguf", "kquant", "Q4_K"),
    "q3_k": ModelInfo("Spark-X2.6-4B-Q3_K-pure.gguf", "kquant", "Q3_K"),
    "q2_k": ModelInfo("Spark-X2.6-4B-Q2_K-pure.gguf", "kquant", "Q2_K"),
    "q4_0-attn": ModelInfo("Spark-X2.6-4B-Q4_0-attn.gguf", "ablation", "attention only Q4_0"),
    "q4_0-ffn": ModelInfo("Spark-X2.6-4B-Q4_0-ffn.gguf", "ablation", "MLP only Q4_0"),
    "q4_0-embd": ModelInfo("Spark-X2.6-4B-Q4_0-embd.gguf", "ablation", "embedding only Q4_0"),
}
MODELS: dict[str, Path] = {k: MODELS_DIR / v.file for k, v in MODEL_INFO.items()}
MAIN_MODELS = ["bf16", "q8_0", "q4_0"]
EXTENSION_MODELS = [m for m in MODEL_INFO if m not in MAIN_MODELS]
REFERENCE_MODEL = "bf16"

# All lenses: ridge regression, 1000 WikiText paragraphs, jlens-gguf defaults (scripts/fit_lenses.sh).
LENSES: dict[str, Path] = {
    "bf16-A": LENSES_DIR / "bf16-A.gguf",  # reference lens
    "bf16-B": LENSES_DIR / "bf16-B.gguf",  # same model, disjoint corpus -> noise floor
    "q8_0-A": LENSES_DIR / "q8_0-A.gguf",
    "q4_0-A": LENSES_DIR / "q4_0-A.gguf",
}
REFERENCE_LENS = "bf16-A"


@dataclass(frozen=True)
class Arm:
    """An experimental arm: activations of `model`, transported with `lens` (None = identity,
    i.e. the classic logit lens), decoded with final norm + unembedding of `unembed`."""

    name: str
    model: str
    lens: str | None
    unembed: str
    role: str  # reference | noise | fixed | fixed_ref_unembed | refit | identity


def _arms() -> list[Arm]:
    arms = [
        Arm("bf16|bf16-A", "bf16", "bf16-A", "bf16", "reference"),
        Arm("bf16|bf16-B", "bf16", "bf16-B", "bf16", "noise"),
        Arm("bf16|identity", "bf16", None, "bf16", "identity"),
    ]
    for q in ("q8_0", "q4_0"):
        arms += [
            Arm(f"{q}|bf16-A", q, "bf16-A", q, "fixed"),
            Arm(f"{q}|bf16-A|U=bf16", q, "bf16-A", "bf16", "fixed_ref_unembed"),
            Arm(f"{q}|{q}-A", q, f"{q}-A", q, "refit"),
            Arm(f"{q}|identity", q, None, q, "identity"),
        ]
    # extension: only the fixed reference lens, with the model's own and with the BF16 unembedding
    for m in EXTENSION_MODELS:
        arms += [
            Arm(f"{m}|bf16-A", m, "bf16-A", m, "fixed"),
            Arm(f"{m}|bf16-A|U=bf16", m, "bf16-A", "bf16", "fixed_ref_unembed"),
        ]
    return arms


ARMS: list[Arm] = _arms()
ARMS_BY_NAME: dict[str, Arm] = {a.name: a for a in ARMS}
REFERENCE_ARM = "bf16|bf16-A"
NOISE_ARM = "bf16|bf16-B"

# Causal test (intervene.py). Steering layers and strength (α in units of the layer's typical
# residual norm) were chosen in a pilot on BF16 only, before any quantized model was run
# (docs/en/04-decisions.md §7); the coordinate swap uses the workspace band.
STEER_LAYERS = range(18, 26)
STEER_ALPHA = 0.05
# which lenses provide the intervention directions per model
INTERVENTION_LENSES: dict[str, list[str]] = {
    "bf16": ["bf16-A", "bf16-B"],
    "q8_0": ["bf16-A", "q8_0-A"],
    "q4_0": ["bf16-A", "q4_0-A"],
    **{m: ["bf16-A"] for m in EXTENSION_MODELS},
}

# Eval sets from anthropics/jacobian-lens (data/evaluations, Apache 2.0).
EVAL_SETS = ["multihop", "multilingual", "order-ops", "association", "typo", "poetry"]
PASS_AT_K = (1, 5, 10)
# Sets on which the reference already scores below this pass@10 are "at floor" and excluded from
# hypothesis testing (fixed before the main run after the BF16 pilot, docs/en/04-decisions.md §5).
FLOOR_PASS_AT_10 = 0.10
TOPK = 10  # size of stored top-k lists and the overlap metric
SCOPE_ALL, SCOPE_BAND = "all layers", "workspace band"
AGG_ALL, AGG_CORE = "all", "core"

# Held-out text for distribution comparisons: WikiText-103 from row 20000 on (corpus A ends at
# row 4596, B at 9199), same revision as the fitting corpora.
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_SHARD = "wikitext-103-raw-v1/train-00000-of-00002.parquet"
HELDOUT_FIRST_ROW = 20000
HELDOUT_N_PARAGRAPHS = 3 if SMOKE else 40
HELDOUT_MIN_CHARS = 600
HELDOUT_MAX_TOKENS = 128
HELDOUT_SKIP_FIRST = 16  # as in fitting: exclude attention sinks
HELDOUT_STRIDE = 4  # every 4th position -> ~28 positions per paragraph

# Prompt format of the eval items. Spark is a chat/reasoning model and completes raw text poorly
# ("… is the" continues with " the the the"). In chat format the prompt is the beginning of the
# assistant's answer after a short instruction, and the model continues it. Chosen on BF16 alone,
# before any quant result (docs/en/04-decisions.md §5).
PROMPT_FORMAT = "chat"  # "chat" | "raw"
CHAT_INSTRUCTION = "Continue the text."

# jlens-server: one model per GPU; with several GPUs, captures run in parallel.
SERVER_PORT = int(os.environ.get("JSPACE_PORT", "8091"))
SERVER_GPU = os.environ.get("JSPACE_GPU", "0")  # CUDA_VISIBLE_DEVICES; one model fits one card
SERVER_GPUS = os.environ.get("JSPACE_GPUS", "0").split(",")
SERVER_CTX = 4096

BOOTSTRAP_SAMPLES = 2000
SEED = 0
