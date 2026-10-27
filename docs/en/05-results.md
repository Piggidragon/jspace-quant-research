# 05 · Results and reflection

All numbers from `results/tables/`, figures in `results/figures/en/`. CIs: 95 % bootstrap, paired
against the reference. "Core" = the four eval sets that are not already at floor on BF16
(`multihop`, `multilingual`, `order-ops`, `typo`; 351 items).

## Summary

1. **8 bit changes nothing measurable.** Q8_0 deviates from BF16 less than a second BF16 lens that was
   only fitted on different text.
2. **The hidden concepts hold down to 4 bit,** crumble at 3 bit and collapse at 2 bit.
3. **The full readout distribution reacts much earlier:** Q6_K is still at the noise floor; from 5 bit
   on it drifts clearly — monotonically with decreasing bit width and growing with layer depth.
4. **The drift arises in the residual stream,** not in the quantized unembedding, and no single
   component explains it — attention, MLP and embedding each contribute.
5. **At 4 bit the space is shifted, not destroyed:** a refitted lens recovers lost concepts and
   predicts the quantized model's behavior as well as for the original.

## 1. Workspace band and positive control

The hidden concepts become readable in layers **L22–L34** (reference hit rate ≥ 50 % of its peak, peak
~19 % at L26–L30). Before that they are practically never in the top 10 (`hit_rate_per_layer.png`). This
matches the paper's picture.

The **positive control is mixed**: on `multihop` (+0.075) and `order-ops` (+0.109) our lens beats the
logit lens, on `typo` it clearly loses (0.44 vs. 0.94), and in the core aggregate the logit lens is ahead
(0.54 vs. 0.43). Within the workspace band alone both are on par (0.31).

## 2. Main study: BF16 → Q8_0 → Q4_0

**Concepts** (pass@10 over all layers, difference to the reference of 0.43):

| Arm | Core | Notable |
|---|---|---|
| noise floor | ±0.000 | `order-ops` −0.082 (significant!) |
| Q8_0, fixed lens | +0.005 [−0.003; +0.014] | — |
| Q4_0, fixed lens | −0.019 [−0.047; +0.009] | `typo` −0.115 [−0.198; −0.042] |
| Q4_0, BF16 unembedding | −0.019 | `typo` −0.094 |
| Q4_0, refitted | +0.007 [−0.023; +0.038] | `typo` −0.031 (n.s.) |

**Distribution** (mean over the workspace band, held-out text):

| Arm | Top-1 = reference | KL [nats] | Fidelity |
|---|---|---|---|
| noise floor | 0.915 | 0.043 | 0.668 |
| Q8_0, fixed lens | 0.949 | 0.036 | 0.668 |
| Q4_0, fixed lens | 0.649 | 0.904 | 0.668 |
| Q4_0, BF16 unembedding | 0.655 | 0.897 | 0.666 |
| Q4_0, refitted | 0.648 | 0.914 | 0.676 |

Per layer (`agreement_per_layer_heldout.png`), Q4_0 starts close to the reference (0.90 at L0), drops
below the noise floor from about L10 on and ends at ~0.65 in the band. Q8_0 stays above the noise floor
everywhere. The noise floor itself is lowest in early layers (0.78): that is where the lens is least
well determined.

**Behavior:** Q8_0 unchanged (perplexity ×1.02). Q4_0: perplexity ×1.46 [1.08; 1.94], `multilingual`
−9 points [−17; −3].

## 3. Bit-width ladder

| Level | Bits/weight | Top-1 = ref. (band) | pass@10 core Δ | Perplexity | rel. error h (band) |
|---|---|---|---|---|---|
| noise floor | — | 0.915 | — | — | — |
| Q8_0 | 8.5 | 0.949 | +0.005 | ×1.02 | 0.08 |
| Q6_K | 6.6 | 0.909 | +0.000 | ×0.97 | 0.15 |
| Q5_K | 5.5 | 0.825 | −0.022* | ×0.98 | 0.25 |
| Q5_0 | 5.5 | 0.802 | −0.009 | ×1.31* | 0.28 |
| Q4_K | 4.5 | 0.711 | +0.005 | ×1.33* | 0.37 |
| Q4_0 | 4.5 | 0.649 | −0.019 | ×1.46* | 0.50 |
| Q3_K | 3.4 | 0.501 | **−0.055*** | ×2.81* | 0.69 |
| Q2_K | 2.6 | 0.102 | **−0.258*** | ×393* | 1.15 |

\* CI excludes 0 or 1.

- **Distribution:** monotonic in bit width; Q6_K at the noise floor, clearly worse below
  (`dose_response.png`).
- **Concepts:** within the noise down to 4 bit, significantly fewer from 3 bit, collapse at 2 bit
  (`multilingual` 0.41 → 0.01).
- **K vs. legacy quants:** at the same bit width the K scheme deviates less (Q4_K 0.71 vs. Q4_0 0.65;
  Q5_K 0.83 vs. Q5_0 0.80) — the finer scaling also pays off internally.
- **Without a lens:** the relative error of the residual stream grows from layer 0 to ~L12 and then
  stays roughly constant (`ladder_per_layer.png`); lens agreement follows the same course.

## 4. Component ablation

Only one part of the model in Q4_0, everything else BF16 (`ablation.png`):

| Variant | Top-1 = ref. (band) | rel. error h (band) | Perplexity | pass@10 core Δ |
|---|---|---|---|---|
| embedding only | 0.834 | 0.18 | ×1.10 | −0.008 |
| attention only | 0.766 | 0.32 | ×1.15 | −0.019 |
| MLP only | 0.774 | 0.31 | ×1.22* | −0.009 |
| everything (Q4_0) | 0.649 | 0.50 | ×1.46* | −0.019 |

Attention and MLP contribute about equally, the embedding less; no component alone reaches the drift of
full Q4_0 — the errors add up. The quantized embedding alone already pushes agreement below the noise
floor, although it is a single tensor: its error enters the input of every layer.

## 5. Which concepts suffer?

Among the concepts the reference reads in the band (rank ≤ 10, n = 249), the share that is lost:

| | rare | less common | more common | common |
|---|---|---|---|---|
| noise floor | 0.14 | 0.02 | 0.02 | 0.02 |
| Q4_0 | 0.21 | 0.10 | 0.09 | 0.07 |
| Q3_K | 0.35 | 0.40 | 0.14 | 0.17 |
| Q2_K | 0.90 | 0.98 | 0.98 | 0.73 |

Rare concepts are lost more often — but they already vary most between two BF16 lenses. A clear
frequency effect beyond the noise only shows at 3 bit.

## 6. Hypotheses

| | Result |
|---|---|
| H1 | Q8_0: yes. 4 bit: concepts yes, distribution no. ≤ 3 bit: no. |
| H2 | Yes: distribution monotonic across the whole ladder; concepts measurably affected only from 3 bit. |
| H3 | Partly: refitting recovers concepts and keeps fidelity; the deviation from the BF16 readout remains, because the model itself computes differently. |
| H0 | Kept for Q8_0 (and Q6_K), rejected from 5 bit on. |

## Reflection

**Flat instead of commonly used quants.** In practice almost everyone runs mixed quants such as
`Q4_K_M`, often with an importance matrix. We deliberately chose flat quants so that bit width is the
only variable and layer curves are not distorted by a bit budget that changes from layer to layer. The
price: our 4-bit models are harsher than what is actually deployed. Q4_0 even quantizes the embedding to
4 bit, which `Q4_K_M` leaves unquantized (unembedding as Q8_0), and the ablation shows that the embedding
alone contributes measurably. Our 4-bit results are therefore rather an upper bound on the damage;
whether mixed and imatrix quants protect concepts better, we did not test.

**Surrogate instead of the true Jacobian lens.** Ridge regression learns which direction of `h_l`
*correlates* with `h_35`, not which one *causes* it. The consequence shows in the mixed positive
control: on `typo` our lens is worse than the logit lens. For comparing quants this is consistent,
because all lenses are fitted the same way; whether a true Jacobian lens reacts differently to
quantization remains open.

**The concept metric is lenient and noisy.** pass@k counts the *best* rank over 36 layers; a shift by a
few layers stays invisible. At the same time it already varies by up to 0.08 per set between two BF16
lenses. Effects below ~0.05 are not reliable with 55–107 items per set. That is why we added the
distribution metrics — they show the drift much earlier.

**Distribution vs. concepts.** That the distribution drifts noticeably at 4 bit while the targeted
concepts remain is plausible if quantization noise mainly shifts weak directions and frequent filler
words (early top-1 tokens are often " the"), while the strong, task-relevant directions in the band are
preserved. We have not demonstrated this mechanism.

**Breakdown by reference strength.** Grouping concepts by how well the reference reads them, well-read
concepts can only get worse and poorly read ones only better (regression to the mean). Such breakdowns
are only interpretable against the noise floor, which is why it appears in every table.

**CKA with caution.** Q8_0 has a cosine of 0.995 to the reference but a CKA of only 0.91 — presumably
because linear CKA is dominated by a few directions with very large variance. We report it but base
conclusions on the relative error and the readouts.

**Prompt format.** The eval sets were written for text completion; Spark is a chat model. We chose the
chat format on BF16 alone, before any quant results existed. Still, the numbers hold for this format;
`association` and `poetry` are not measurable with it on this model.

**One model.** All statements concern Spark-X2.6-4B. According to the literature small models are more
sensitive; larger ones might be more robust.
