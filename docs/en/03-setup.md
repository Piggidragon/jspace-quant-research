# 03 · Setup and reproduction

Tested on Linux with Python 3.12 (venv via `uv`), CUDA 13 and two GPUs with 12 GB each. One GPU is
enough; every model fits on one card.

## Versions

| Component | Version |
|---|---|
| llama.cpp | upstream `a37cc4610` (CUDA build), supports `spark2_5` |
| [jlens-gguf](https://github.com/igorbarshteyn/jlens-gguf) | `16e5526` + [`patches/jlens-gguf-local.patch`](../../patches/jlens-gguf-local.patch) |
| [jacobian-lens](https://github.com/anthropics/jacobian-lens) | `581d398` (only `data/evaluations/`) |
| `llama-quantize` | from a llama.cpp build with `spark2_5` support |

## Our patches to jlens-gguf

1. **`native/jlens_server.cpp`:** the llama.cpp pinned by jlens-gguf does not know `spark2_5`; upstream
   does, but replaced `use_mmap`/`use_mlock` with `load_mode`. The patch maps the old flags onto it.
2. **`jlens_gguf/model_reader.py`:** BF16 tensors arrived as raw bytes and were converted to numbers
   incorrectly (twice the rows, byte values as weights). This affected readouts from the BF16 model,
   not fitting. Now decoded via `gguf.quants.dequantize`.

Details: [`patches/README.md`](../../patches/README.md).

## Steps

```bash
# third-party repos (not versioned)
git clone https://github.com/anthropics/jacobian-lens repos/jacobian-lens
git clone https://github.com/igorbarshteyn/jlens-gguf repos/jlens-gguf
git -C repos/jlens-gguf checkout 16e5526
git -C repos/jlens-gguf apply ../../patches/jlens-gguf-local.patch
LLAMA_DIR=/path/to/llama.cpp repos/jlens-gguf/native/build.sh   # build libllama with CUDA first

# Python
uv venv .venv && uv pip install -p .venv/bin/python -e repos/jlens-gguf -e ".[dev]"
.venv/bin/python -m pytest

# models
hf download XHToken/Spark-X2.6-4B-GGUF Spark-X2.6-4B.gguf Spark-X2.6-4B-Q8_0.gguf --local-dir models
export LLAMA_QUANTIZE=/path/to/llama-quantize
scripts/make_flat_q4.sh              # Q4_0
scripts/make_quant_variants.sh       # extension: ladder + ablation
.venv/bin/python scripts/quant_layout.py models/<file>.gguf   # check the type per tensor

# lenses (~40 min each) and analysis
scripts/fit_lenses.sh
.venv/bin/python -m jspace_quant capture --gpus 0,1   # only models not captured yet
.venv/bin/python -m jspace_quant analyze              # only arms not computed yet
.venv/bin/python -m jspace_quant plot                 # figures in English and German

# notebooks
.venv/bin/python scripts/build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.en.ipynb
```

Without `LLAMA_DIR`, `build.sh` builds against the pinned submodule, which does not know `spark2_5`; if
`libllama` does not exist yet, the script builds it without CUDA. `JSPACE_SMOKE=1` runs the pipeline on
a few items into `results/smoke/`. `JSPACE_GPU`/`JSPACE_GPUS` select the GPUs, which must be free.

## Runtimes

| Step | Duration |
|---|---|
| Quantizing | seconds to 4.5 min per model |
| Lens fit | ~40 min per lens (CPU-bound: Gram matrices in float64) |
| Capture | ~1 min per model |
| Analysis | ~35 min for the 11 arms of the main study, ~50 min for the 18 arms of the extension (CPU) |

## Directories

| Path | Content | In the repo |
|---|---|---|
| `src/jspace_quant/`, `tests/`, `scripts/` | code | yes |
| `data/corpus/`, `patches/` | corpora, patches | yes |
| `results/tables/`, `results/figures/{en,de}/`, `results/logs/` | results | yes |
| `notebooks/walkthrough.{en,de}.ipynb` | walkthrough with results | yes |
| `results/activations/` (~7 GB), `lenses/` (1.8 GB), `models/` (~50 GB) | intermediate data | no, reproducible |
