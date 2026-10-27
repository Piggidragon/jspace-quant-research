# Local patches

`jlens-gguf-local.patch` applies to [igorbarshteyn/jlens-gguf](https://github.com/igorbarshteyn/jlens-gguf)
at commit `16e5526` (Apache 2.0):

1. **`native/jlens_server.cpp` — API update.** The llama.cpp pinned by jlens-gguf (`e3546c7`) does
   not know the `spark2_5` architecture. Upstream llama.cpp does, but replaced `use_mmap`/`use_mlock`
   in `llama_model_params` with the `load_mode` enum, so the server no longer compiles. The patch maps
   the old flags onto the enum; behavior is unchanged.
2. **`jlens_gguf/model_reader.py` — BF16 bug fix.** gguf-py returns BF16 tensors as raw bytes (numpy
   has no bfloat16); jlens-gguf converted them with `astype(float32)`, producing an unembedding
   matrix with twice the rows, filled with byte values instead of weights. This affected readouts
   from BF16 models (here: the reference model, whose embedding/unembedding is BF16), not fitting.
   Fix: decode BF16 via `gguf.quants.dequantize`.

```bash
git clone https://github.com/igorbarshteyn/jlens-gguf repos/jlens-gguf
git -C repos/jlens-gguf checkout 16e5526
git -C repos/jlens-gguf apply ../../patches/jlens-gguf-local.patch
```
