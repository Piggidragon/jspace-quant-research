# 04 · Methodological decisions

What we found along the way and how it changed the design, in the order it happened. Decisions §1–§5
were made **before** the main run, without looking at results of the quantized models; §6 is a
correction of the statistics after the first main run.

## 1. Spark only runs with patches

The llama.cpp pinned by jlens-gguf does not know the `spark2_5` architecture. With current llama.cpp and
patch 1 ([`03-setup.md`](03-setup.md)) the model loads; the Python readout matches llama.cpp's logits
(correlation 0.99998, same top token).

The first readout from the BF16 model returned token ids beyond the vocabulary size (143,784 > 131,072).
The cause was a bug in jlens-gguf when reading BF16 weights ⇒ patch 2.

## 2. The reference is BF16, not F16

The official file labeled "F16" is BF16 according to its header — the model's native format, hence a
lossless reference.

## 3. Flat quants only

The official `Q4_K_M` mixes types per layer (Q6_K/Q4_K, single tensors Q8_0/BF16). A deviation in layer
4 could then simply come from layer 4 having fewer bits than layer 3. ⇒ We use only **flat** quants
(every tensor the same type) and create them ourselves with `llama-quantize --pure`. What this choice
means for the conclusions is discussed in the reflection ([`05-results.md`](05-results.md) §Reflection).

"Flat" is independent of the **encoding scheme**: legacy quants (Q8_0, Q5_0, Q4_0) store blocks of 32
weights with one scale; K-quants (Q6_K … Q2_K) store super-blocks of 256 weights with their own scales
and minima per sub-block. Only the recipes with `_M`/`_S` in their name are mixed.

## 4. Matrix distances are misleading

Comparing the lens *matrices*, two BF16 lenses fitted on different corpora differ more (relative
distance 0.73) than the BF16 and the Q4_0 lens (0.49). When *reading out* real activations, however, the
two BF16 lenses largely agree (top-1 equal at 78–95 % of positions, depending on the layer). The
matrices are practically arbitrary in directions where the activations barely vary. ⇒ We only measure
**readouts**, and the second BF16 lens serves as the **noise floor**: a quant effect has to be larger
than the ~20 % disagreement that fitting alone causes in early layers.

## 5. Prompt format

**Trailing spaces.** Many eval prompts end in a space ("… is the "). Spark turns that into a lone space
token that hardly occurs in normal text (there the space is attached to the next word). ⇒ Remove
trailing spaces; read at the last real token.

**Spark does not complete raw text.** For "Fact: The ocean on the coast of the country where Carnival is
most famously celebrated is the" the model answers " the the the …" — it does not work on the task at
all, so there is no intermediate concept to read. Pilot test on BF16 (pass@10 lens / logit lens /
answer correct):

| Set | raw | chat format |
|---|---|---|
| multihop | 0.13 / 0.06 / 0.01 | **0.46** / 0.39 / **0.34** |
| multilingual | 0.22 / 0.19 / 0.04 | **0.41** / 0.42 / **0.49** |
| order-ops | 0.59 / 0.55 / 0.13 | 0.43 / 0.32 / **0.31** |
| association | 0.00 / 0.03 / — | 0.01 / 0.03 / — |
| typo | 0.31 / 0.76 / — | 0.44 / 0.94 / — |
| poetry | 0.00 / 0.02 / — | 0.01 / 0.02 / — |

⇒ **Chat format:** the prompt is the beginning of the assistant's answer after "Continue the text.".

Two further consequences of the pilot test:
- `association` and `poetry` are already at floor on BF16. Quantization cannot make them worse ⇒ they
  are reported but excluded from the hypotheses.
- On `typo` the logit lens clearly beats our lens. Probably because the corrected word is almost the
  next output token there, which the logit lens reads directly. That is a weakness of the regression
  surrogate, not an effect of quantization.

## 6. Paired statistics

In the first main run the confidence interval of the perplexity was uselessly wide (60–230), because the
models were compared unpaired and the paragraphs differ a lot in difficulty. Since all models see the
same positions, we compute differences per paragraph or item ⇒ much narrower, meaningful intervals.
