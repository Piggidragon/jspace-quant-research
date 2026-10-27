# 02 · Methodology

Reasons behind the individual decisions: [`04-decisions.md`](04-decisions.md). How to run it:
[`03-setup.md`](03-setup.md). Results: [`05-results.md`](05-results.md).

## 1. Question and hypotheses

How robust are Jacobian-lens readouts to weight quantization?

| | Hypothesis | Tested via |
|---|---|---|
| H1 | The lens fitted on BF16 reads the same intermediate concepts on Q8_0/Q4_0. | pass@k of the fixed lens vs. reference |
| H2 | The deviation grows monotonically with stronger quantization. | agreement metrics across levels |
| H3 | A lens refitted on the quant restores the signal (space shifted, not destroyed). | refitted vs. fixed lens |
| H4 | The lens directions fitted on BF16 still act causally on the quant: intervening along them changes the answer as on BF16. | steering success and ablation effect vs. reference (§6 E) |
| H0 | Quantization changes nothing measurable. | all differences within the noise floor |

Expectation from the literature ([`01-background.md`](01-background.md)): no effect for Q8_0, a clear
monotonic drift for Q4_0, weak concepts first.

## 2. Materials

**Models** — flat quants only, i.e. every weight tensor has the same type:

| Key | Type | Bits/weight | Origin |
|---|---|---|---|
| `bf16` | BF16 | 16.0 | official, reference |
| `q8_0` | Q8_0 | 8.5 | official |
| `q4_0` | Q4_0 | 4.5 | our own (`scripts/make_flat_q4.sh`) |

Extension (§6): Q5_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K and three ablation models
(`scripts/make_quant_variants.sh`).

**Fitting corpus:** WikiText-103, frozen in `data/corpus/`. Corpus A = the first 1000 paragraphs with
≥ 600 characters (all lenses), corpus B = the next 1000 (noise lens only).

**Test data:**
- the paper's six eval sets (`anthropics/jacobian-lens/data/evaluations`, 551 items): `multihop`,
  `multilingual`, `order-ops`, `association`, `typo`, `poetry`. Every item has a prompt and one or
  more **intermediate concepts** that are not in the text.
- 40 WikiText paragraphs that appear in no corpus (1120 positions), for distribution comparisons and
  perplexity.

## 3. Lens fitting

Per layer `l`, a ridge regression of the last layer's residual stream on that of layer `l`:

```
A_l, b_l = argmin Σ_p ‖A·h_l[p] + b − h_35[p]‖² + λ‖A‖²,   λ = 1e-4 · trace(Cov)/d
```

over all positions of the 1000 paragraphs (truncated to 128 tokens, first 16 and last position
excluded ⇒ 110,906 positions). Four lenses, all with identical parameters:

| Lens | Model | Corpus | Role |
|---|---|---|---|
| `bf16-A` | BF16 | A | reference lens |
| `q8_0-A` | Q8_0 | A | refitted |
| `q4_0-A` | Q4_0 | A | refitted |
| `bf16-B` | BF16 | B | noise floor |

## 4. Experimental arms

An arm combines the activations of a model M, a lens and the unembedding of a model U:

```
logits(l, p) = Unembed_U( RMSNorm( A_l · h_l^M[p] + b_l ) )
```

| Arm | M | Lens | U | Purpose |
|---|---|---|---|---|
| reference | BF16 | bf16-A | BF16 | point of comparison |
| noise floor | BF16 | bf16-B | BF16 | deviation from fitting alone |
| logit lens | M | — | M | baseline, positive control |
| fixed lens | Q | bf16-A | Q | did the space stay the same? (H1, H2) |
| fixed lens, BF16 unembedding | Q | bf16-A | BF16 | drift in the residual stream or in the 4-bit unembedding? |
| refitted | Q | q-A | Q | is the space still linearly readable? (H3) |

The last layer (35) has no lens matrix and is always read with the identity — that is exactly the
model's output.

**Prompt format:** Spark is a chat model. Every eval prompt is placed as the beginning of the
assistant's answer in the chat template (instruction "Continue the text."), trailing spaces removed.
Held-out text stays raw, as in fitting.

## 5. Metrics and statistics

**Concepts (the paper's metric):** the readout position is the last prompt token (for `poetry` the end
of the first line). A concept's rank = best rank over its single-token spellings (for `order-ops` also
synonyms such as "5"/"five", "*"/"times").
- **pass@k** = mean over items of the share of concepts whose best rank across layers is ≤ k (k = 1,
  5, 10); once over all layers, once only in the workspace band.
- **Hit rate per layer** = share of concepts with rank ≤ 10 in that layer.
- **Workspace band** = contiguous range around the peak of the reference hit rate in which it is
  ≥ 50 % of the peak (last layer excluded).

**Distribution**, per arm, layer and position against the reference readout: top-1 equal, top-10
overlap, KL(reference ‖ arm).

**Fidelity:** top-1 of the readout = the model's own output.

**Behavior:** share of correct greedy answers (`multihop`, `multilingual`, `order-ops`) and perplexity
on the held-out text.

**Statistics:**
- 95 % bootstrap CIs (2000 resamples); for held-out metrics clustered by paragraph.
- Comparisons to the reference are paired (same items/positions).
- An effect counts if the CI excludes 0 *and* it is clearly larger than for the noise lens.
- **Floor sets:** sets on which the reference already has pass@10 < 0.10 are excluded from H1–H3 (fixed
  before the main run). The **"core"** aggregate averages over the rest.

## 6. Extension

Added after the main run; A–D evaluated with the fixed BF16 lens only:

- **A — representations without a lens:** per layer, relative error `‖h_q − h_BF16‖/‖h_BF16‖`, cosine
  and linear CKA against BF16.
- **B — which concepts suffer?** Rank change in the band by strength in the reference and by token
  frequency in corpus A.
- **C — component ablation:** only attention, only MLP or only embedding in Q4_0, rest BF16.
- **D — bit-width ladder:** legacy family (Q8_0, Q5_0, Q4_0) and K family (Q6_K to Q2_K), all flat.
- **E — causal test (interventions):** the paper's `probe-swap` set (90 two-hop questions with bridge
  concept, answer, swap concept and swap answer, e.g. Brazil → Portuguese, Mexico → Spanish). At every
  prompt position the residual stream is changed along lens directions and the next token is measured:
  - **steering** at L18–L25: `h += α·h_rms·(v̂_swap − v̂_bridge)`, α = 0.05 — success if the next
    token is the swap answer. Control: steering towards an unrelated concept.
  - the paper's **coordinate swap** in the workspace band (`h' = h + V(σ(c) − c)`, `c = V⁺h`).
  - **ablation** of the bridge direction at L18–L25 — effect = correct answers lost. Control: ablating
    an unrelated concept.

  `v̂_t` is the lens direction of token `t` at the given layer (`Jᵀu_t`, normalized), computed with the
  unembedding of the model being intervened on. Arms: BF16 (lens A and noise lens B), Q8_0 and Q4_0
  each with the fixed and the refitted lens, all other models with the fixed lens. Layers and strength
  come from a BF16 pilot ([`04-decisions.md`](04-decisions.md) §7).

## 7. Limitations

- **Surrogate lens:** regression instead of the causal Jacobian; consistent across all quants, but not
  the paper's lens.
- **One model, one prompt format, flat quants only.**
- **Noise:** the concept metric already varies by up to 0.08 per set between two BF16 lenses; in early
  layers their readouts disagree at ~20 % of positions.
- **Concept tokenization:** 6 % of concepts have no single-token form (fallback: first token).
