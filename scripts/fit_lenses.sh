#!/usr/bin/env bash
# Fit all four regression lenses of the project, one after another, on one GPU.
#
#   bf16-A  BF16 model,  corpus A   reference lens
#   q8_0-A  Q8_0 model,  corpus A   refit on Q8_0
#   q4_0-A  Q4_0 model,  corpus A   refit on flat Q4_0
#   bf16-B  BF16 model,  corpus B   noise floor (disjoint corpus, same model)
#
# Every fit uses jlens-gguf's defaults apart from the corpus: 1000 prompts, max 128 tokens,
# first 16 positions and the last one excluded, ridge 1e-4 (relative to trace(Cxx)/d), affine,
# Gram matrices in float64. Each fit takes ~40 min (CPU-bound Gram accumulation).
#
#   scripts/fit_lenses.sh                 # all four
#   scripts/fit_lenses.sh q4_0-A bf16-B   # selected ones
#   JSPACE_GPU=1 scripts/fit_lenses.sh    # other GPU
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
GPU="${JSPACE_GPU:-0}"

declare -A MODEL=(
    [bf16-A]=Spark-X2.6-4B.gguf [bf16-B]=Spark-X2.6-4B.gguf
    [q8_0-A]=Spark-X2.6-4B-Q8_0.gguf [q4_0-A]=Spark-X2.6-4B-Q4_0-pure.gguf
)
declare -A CORPUS=(
    [bf16-A]=wikitext103-A-n1000.txt [q8_0-A]=wikitext103-A-n1000.txt
    [q4_0-A]=wikitext103-A-n1000.txt [bf16-B]=wikitext103-B-n1000.txt
)

lenses=("$@")
[ ${#lenses[@]} -eq 0 ] && lenses=(bf16-A q8_0-A q4_0-A bf16-B)

mkdir -p "$ROOT/lenses" "$ROOT/results/logs"
for name in "${lenses[@]}"; do
    [ -n "${MODEL[$name]:-}" ] || { echo "unknown lens: $name" >&2; exit 1; }
    log="$ROOT/results/logs/fit-$name.log"
    echo "=== $name  ($(date +%T))  log: $log"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m jlens_gguf fit \
        --model "$ROOT/models/${MODEL[$name]}" \
        --corpus "$ROOT/data/corpus/${CORPUS[$name]}" --n-prompts 1000 \
        --n-gpu-layers 99 -o "$ROOT/lenses/$name.gguf" >"$log" 2>&1
    echo "=== $name done ($(date +%T))"
done
