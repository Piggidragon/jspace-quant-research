# Fitting corpora

Excerpts from **WikiText-103** (`Salesforce/wikitext`, configuration `wikitext-103-raw-v1`, split
`train`), Merity et al. 2016, licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The texts come from Wikipedia
articles and are unchanged except for collapsed whitespace (one paragraph per block, blocks
separated by a blank line).

| File | Content | Purpose |
|---|---|---|
| `wikitext103-A-n1000.txt` | the first 1000 paragraphs with ≥ 600 characters | fitting all lenses |
| `wikitext103-B-n1000.txt` | the next 1000 such paragraphs (disjoint from A) | noise-floor lens on BF16 |
| `manifest.json` | dataset revision, shard, row ranges, retrieval date | reproducibility |

Regenerate (identical files for the same revision):

```bash
.venv/bin/python -I scripts/make_fit_corpus.py 1000
```

Use when fitting: `--corpus data/corpus/wikitext103-A-n1000.txt --n-prompts 1000` (without
`--n-prompts`, `jlens_gguf fit` silently uses only the first 100 paragraphs).
