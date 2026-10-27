#!/usr/bin/env bash
# Create the flat Q4_0 model from the official BF16 GGUF.
#
# --pure disables llama.cpp's k-quant mixing rules, and the two type overrides put the
# (tied) token embedding and the output tensor into Q4_0 as well, so every weight tensor
# ends up Q4_0 (only norms stay F32) — the same flat layout as the official Q8_0. No imatrix.
#
#   LLAMA_QUANTIZE=/path/to/llama-quantize scripts/make_flat_q4.sh
#
# Needs a llama-quantize from a llama.cpp build that knows the spark2_5 architecture (upstream
# a37cc4610 or later); found on PATH or via LLAMA_QUANTIZE.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q="${LLAMA_QUANTIZE:-llama-quantize}"

mkdir -p "$ROOT/results/logs"
"$Q" --pure --output-tensor-type q4_0 --token-embedding-type q4_0 \
    "$ROOT/models/Spark-X2.6-4B.gguf" "$ROOT/models/Spark-X2.6-4B-Q4_0-pure.gguf" Q4_0 \
    2>&1 | tee "$ROOT/results/logs/quantize-q4_0-pure.log"
