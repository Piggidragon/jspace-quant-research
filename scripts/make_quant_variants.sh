#!/usr/bin/env bash
# Create the extension models from the official BF16 GGUF (no imatrix anywhere).
#
# Bit-width ladder — every weight tensor (incl. the tied embedding) gets the same type:
#   q5_0                      legacy family, between the official Q8_0 and our Q4_0
#   q6_k q5_k q4_k q3_k q2_k  k-quant family, 6 → 2 bit
#
# Component ablation — only one component in Q4_0, everything else stays BF16:
#   q4_0-attn   attn_qkv, attn_output, attn_gate
#   q4_0-ffn    ffn_up, ffn_gate, ffn_down
#   q4_0-embd   token embedding (tied, i.e. also the unembedding)
# Implemented as a flat Q4_0 target whose other tensors are overridden back to BF16.
#
#   scripts/make_quant_variants.sh            # all
#   scripts/make_quant_variants.sh q3_k q4_0-ffn
#   LLAMA_QUANTIZE=/path/to/llama-quantize scripts/make_quant_variants.sh
#
# Verify every result with scripts/quant_layout.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q="${LLAMA_QUANTIZE:-llama-quantize}"
SRC="$ROOT/models/Spark-X2.6-4B.gguf"

flat() {  # flat <file-suffix> <ftype> <ggml type for embedding/output>
    "$Q" --pure --token-embedding-type "$3" --output-tensor-type "$3" "$SRC" \
        "$ROOT/models/Spark-X2.6-4B-$1.gguf" "$2"
}

only_q4() {  # only_q4 <file-suffix> <tensor-type overrides back to bf16...>
    local suffix=$1; shift
    local args=()
    for t in "$@"; do args+=(--tensor-type "$t=bf16"); done
    "$Q" --pure "${args[@]}" "$SRC" "$ROOT/models/Spark-X2.6-4B-$suffix.gguf" Q4_0
}

build() {
    case $1 in
        q5_0) flat Q5_0-pure Q5_0 q5_0 ;;
        q6_k) flat Q6_K-pure Q6_K q6_K ;;
        q5_k) flat Q5_K-pure Q5_K_M q5_K ;;
        q4_k) flat Q4_K-pure Q4_K_M q4_K ;;
        q3_k) flat Q3_K-pure Q3_K_M q3_K ;;
        q2_k) flat Q2_K-pure Q2_K q2_K ;;
        q4_0-attn) only_q4 Q4_0-attn ffn_up ffn_gate ffn_down token_embd ;;
        q4_0-ffn)  only_q4 Q4_0-ffn attn_qkv attn_output attn_gate token_embd ;;
        q4_0-embd) only_q4 Q4_0-embd attn_qkv attn_output attn_gate ffn_up ffn_gate ffn_down ;;
        *) echo "unknown variant: $1" >&2; exit 1 ;;
    esac
}

variants=("$@")
[ ${#variants[@]} -eq 0 ] && variants=(q5_0 q6_k q5_k q4_k q3_k q2_k q4_0-attn q4_0-ffn q4_0-embd)

mkdir -p "$ROOT/results/logs"
for v in "${variants[@]}"; do
    echo "=== $v ($(date +%T))"
    build "$v" >"$ROOT/results/logs/quantize-$v.log" 2>&1
    tail -n 3 "$ROOT/results/logs/quantize-$v.log" | grep -E "quant size|total time" || true
done
