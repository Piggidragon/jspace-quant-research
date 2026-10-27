"""Lens readouts per experimental arm: logits = Unembed(FinalNorm(A_l · h_l + b_l)).

Lens transport and readout come from jlens-gguf (JacobianLensGGUF.transport,
ReadoutWeights.unembed); this module only adds cached loading and the arm logic.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from jlens_gguf.lens import JacobianLensGGUF
from jlens_gguf.model_reader import ReadoutWeights

from jspace_quant import config
from jspace_quant.config import Arm


@cache
def readout_weights(model: str) -> ReadoutWeights:
    """Final norm + unembedding, dequantized to fp32 from the model's GGUF."""
    return ReadoutWeights.from_gguf(str(config.MODELS[model]))


@cache
def lens(name: str) -> JacobianLensGGUF:
    return JacobianLensGGUF.load(str(config.LENSES[name]))


def arm_logits(arm: Arm, layer: int, h: np.ndarray) -> np.ndarray:
    """[n, d] residuals from `layer` -> [n, n_vocab] logits of the arm.

    Layers without a fitted matrix (the last block, or all layers for lens=None) are read with
    J = I; on the last block that is exactly the model's output.
    """
    h = np.asarray(h, dtype=np.float32)
    if arm.lens is not None:
        L = lens(arm.lens)
        if layer in L.jacobians:
            h = L.transport(h, layer)
    return readout_weights(arm.unembed).unembed(h)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=-1, keepdims=True)
    z = logits - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def topk_ids(logits: np.ndarray, k: int) -> np.ndarray:
    """[n, V] -> [n, k] token ids, sorted by descending logit."""
    part = np.argpartition(-logits, k, axis=-1)[:, :k]
    order = np.take_along_axis(logits, part, axis=-1).argsort(axis=-1)[:, ::-1]
    return np.take_along_axis(part, order, axis=-1)


def min_rank(logits: np.ndarray, token_ids: list[int]) -> np.ndarray:
    """[n, V] -> [n] best (smallest) 1-based rank among `token_ids`.

    Rank = 1 + number of tokens with a strictly larger logit (ties go to the candidate).
    """
    cand = logits[:, token_ids]  # [n, c]
    best = cand.max(axis=-1, keepdims=True)
    return 1 + (logits > best).sum(axis=-1)
