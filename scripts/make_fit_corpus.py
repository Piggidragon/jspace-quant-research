"""Freeze the WikiText fitting corpora as local text files.

Writes two disjoint corpora of N paragraphs each from WikiText-103 (train split):

  A  = the first N paragraphs with >= MIN_CHARS characters  (main corpus, all quants)
  B  = the next N such paragraphs                           (noise-floor control, F16 only)

Selection matches the reference implementation (anthropics/jacobian-lens,
jlens/examples.py: first records of >= 600 chars, in dataset order). Rows are read
from the first train parquet shard at a pinned dataset revision (the HF rows API
rate-limits after a few dozen requests). Each paragraph is one block, blocks are
separated by a blank line, which is the format
`python -m jlens_gguf fit --corpus FILE --n-prompts N` reads.

Usage: python -I scripts/make_fit_corpus.py [N] [OUT_DIR]
"""

import json
import sys
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

DATASET = "Salesforce/wikitext"
SHARD = "wikitext-103-raw-v1/train-00000-of-00002.parquet"
MIN_CHARS = 600


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "data" / "corpus"
    out.mkdir(parents=True, exist_ok=True)

    revision = HfApi().dataset_info(DATASET).sha
    shard = hf_hub_download(DATASET, SHARD, repo_type="dataset", revision=revision)

    paras = []  # (row_idx, text)
    for row_idx, text in enumerate(pq.read_table(shard, columns=["text"]).column("text").to_pylist()):
        text = " ".join(text.split())  # one line, no blank lines inside a block
        if len(text) >= MIN_CHARS:
            paras.append((row_idx, text))
            if len(paras) == 2 * n:
                break
    if len(paras) < 2 * n:
        sys.exit(f"shard has only {len(paras)} paragraphs >= {MIN_CHARS} chars")

    manifest = {"dataset": DATASET, "revision": revision, "shard": SHARD, "min_chars": MIN_CHARS,
                "fetched": date.today().isoformat(), "license": "CC BY-SA 4.0", "corpora": {}}
    for name, part in (("A", paras[:n]), ("B", paras[n:])):
        path = out / f"wikitext103-{name}-n{n}.txt"
        path.write_text("\n\n".join(t for _, t in part) + "\n", encoding="utf-8")
        manifest["corpora"][name] = {"file": path.name, "n": len(part),
                                     "row_idx_first": part[0][0], "row_idx_last": part[-1][0]}
        print(f"wrote {path} ({len(part)} paragraphs, rows {part[0][0]}..{part[-1][0]})")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
