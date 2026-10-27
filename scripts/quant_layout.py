"""Print the per-tensor quantization layout of a GGUF: which ggml type each weight got.

Usage: python -I scripts/quant_layout.py MODEL.gguf
Needs gguf-py (pip install gguf, or llama.cpp's gguf-py on PYTHONPATH).
"""

import re
import sys
from collections import Counter, defaultdict

from gguf import GGUFReader

r = GGUFReader(sys.argv[1])

for k, f in r.fields.items():
    if k.startswith("quantize.") or k == "general.file_type":
        v = f.parts[f.data[0]]
        print(f"{k} = {bytes(v).decode(errors='replace') if v.dtype.kind == 'u' and v.dtype.itemsize == 1 else v.tolist()}")

per_kind = defaultdict(dict)  # kind -> {layer: type}
other = {}
for t in r.tensors:
    m = re.match(r"blk\.(\d+)\.(.+)\.weight$", t.name)
    if m:
        per_kind[m.group(2)][int(m.group(1))] = t.tensor_type.name
    else:
        other[t.name] = t.tensor_type.name

print("\nnon-layer tensors:")
for n, ty in other.items():
    print(f"  {n:30s} {ty}")

print("\nper-layer tensors (type: layers):")
for kind, by_layer in sorted(per_kind.items()):
    groups = defaultdict(list)
    for l, ty in sorted(by_layer.items()):
        groups[ty].append(l)
    desc = "; ".join(f"{ty}: {ls if len(ls) < len(by_layer) else 'all'}" for ty, ls in groups.items())
    print(f"  {kind:22s} {desc}")

print("\ntype totals:", dict(Counter(t.tensor_type.name for t in r.tensors)))
