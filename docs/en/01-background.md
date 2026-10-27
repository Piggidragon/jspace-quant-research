# 01 · Background

## Global workspace and the Jacobian lens

Gurnee, Sofroniew et al. (2026), *Verbalizable Representations Form a Global Workspace in Language
Models* ([transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace/index.html)),
show that language models hold concepts in their middle layers that can be put into words — including
concepts that never appear in the text (e.g. "Brazil" as the intermediate step of a two-hop question).
They read them out with the **Jacobian lens**:

```
lens_l(h) = Unembed( J_l · h ),   J_l = E[ ∂h_final / ∂h_l ]
```

`J_l` maps a residual-stream vector from layer `l` into the space of the last layer; the unembedding
turns it into a ranked list of tokens. With `J_l = I` this is the classic **logit lens**. The original
lenses are fitted by backpropagation over 1000 sequences of 128 tokens; according to the paper ~100
already give usable quality. Limitations the authors name: single-token concepts only, early layers
hardly interpretable, and the lens is "an imperfect tool".

Reference code and eval sets: [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
(Apache 2.0). Pre-fitted lenses exist only for other models
([neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens)), not for Spark.

## Tool: jlens-gguf

[igorbarshteyn/jlens-gguf](https://github.com/igorbarshteyn/jlens-gguf) applies the lens to any GGUF
file llama.cpp can load: a C++ server (`jlens-server`) returns the residual stream of every layer,
Python/numpy computes lens and readout. Since llama.cpp computes no gradients, jlens-gguf fits a
**ridge regression** `h_final ≈ A_l·h_l + b` instead of the true Jacobian (forward passes only). That
is a correlational surrogate in the tradition of the **tuned lens** (Belrose et al. 2023,
[arXiv:2303.08112](https://arxiv.org/abs/2303.08112)).

## Model: Spark-X2.6-4B

[XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B), GGUFs at
[XHToken/Spark-X2.6-4B-GGUF](https://huggingface.co/XHToken/Spark-X2.6-4B-GGUF).

| | |
|---|---|
| Architecture | `spark2_5`, 36 layers, residual width 2560 |
| Attention | hybrid: sliding window (512) with full attention in every fourth layer |
| Vocabulary | 131,072, tied embedding and unembedding |
| Native dtype | bfloat16 |

**Why Spark:** small enough for one GPU, yet practically usable — rather than the older test models
(GPT-2, Pythia) much interpretability work runs on. It is trained with a modern recipe and is not a
classic dense full-attention architecture, so the study also tests whether the method carries beyond
the standard case.

## Literature: quantization and internal representations

We found no work on exactly our question (lens readouts under GGUF quantization, with a refitted lens
per level). Closest neighbors and the expectations they suggest:

| Work | Finding | Expectation for us |
|---|---|---|
| Hu & Ramachandran 2026, [arXiv:2609.01587](https://arxiv.org/abs/2609.01587) | 8 bit is near-lossless for RTN, GPTQ and AWQ; damage is spread over many layers | Q8_0: no measurable effect |
| Duan 2026, [arXiv:2606.03002](https://arxiv.org/abs/2606.03002) | SAE features degrade gradually under quantization, mostly "blurred"; perplexity misses the damage; strongly active features survive more often | perplexity only as a control; strong concepts last longer |
| Wu, Dhiman & Koshiyama 2026, [arXiv:2608.06564](https://arxiv.org/abs/2608.06564) | low-bit quantization shrinks decision margins roughly proportionally | systematic shift without structural break |
| Wang et al. 2025, [arXiv:2505.13963](https://arxiv.org/abs/2505.13963) (TrustNLP @ ACL 2026) | smaller models lose more factual knowledge under quantization | 4B is a rather sensitive regime |
| menelly/vspace-jlens (community, not peer-reviewed) | NF4 largely preserves workspace geometry (cosine 0.968); on two very small models (0.5B, 1.7B) the J-lens fails its own positive control | 4 bit: geometry largely preserved; positive control needed |
| willkn/jlens_re (analysis on GPT-2-medium) | the raw J-lens is worse than the logit lens on next-token faithfulness at every layer | the logit lens is a serious comparison, not just a baseline |

Caveat: these findings come from other methods (RTN, GPTQ, AWQ, NF4); GGUF quants are structurally
different. Quantization foundations: Frantar et al. 2022 (GPTQ,
[arXiv:2210.17323](https://arxiv.org/abs/2210.17323)), Lin et al. 2023 (AWQ,
[arXiv:2306.00978](https://arxiv.org/abs/2306.00978)). Representation similarity: Kornblith et al.
2019 (CKA, [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)).
