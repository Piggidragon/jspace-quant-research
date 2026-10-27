# 03 · Setup und Reproduktion

Getestet unter Linux mit Python 3.12 (venv mit `uv`), CUDA 13 und zwei GPUs mit je 12 GB. Eine GPU
genügt; jedes Modell passt auf eine Karte.

## Versionen

| Komponente | Version |
|---|---|
| llama.cpp | upstream `a37cc4610` (CUDA-Build), unterstützt `spark2_5` |
| [jlens-gguf](https://github.com/igorbarshteyn/jlens-gguf) | `16e5526` + [`patches/jlens-gguf-local.patch`](../../patches/jlens-gguf-local.patch) |
| [jacobian-lens](https://github.com/anthropics/jacobian-lens) | `581d398` (nur `data/evaluations/`) |
| `llama-quantize` | aus einem llama.cpp-Build mit `spark2_5`-Support |

## Eigene Patches an jlens-gguf

1. **`native/jlens_server.cpp`:** Das in jlens-gguf gepinnte llama.cpp kennt `spark2_5` nicht;
   upstream kennt es, hat aber `use_mmap`/`use_mlock` durch `load_mode` ersetzt. Der Patch bildet die
   alten Flags darauf ab.
2. **`jlens_gguf/model_reader.py`:** BF16-Tensoren kamen als rohe Bytes an und wurden falsch in Zahlen
   umgewandelt (doppelte Zeilenzahl, Byte-Werte als Gewichte). Betraf den Readout am BF16-Modell, nicht
   das Fitting. Jetzt über `gguf.quants.dequantize`.

Details: [`patches/README.md`](../../patches/README.md).

## Schritte

```bash
# Fremd-Repos (nicht versioniert)
git clone https://github.com/anthropics/jacobian-lens repos/jacobian-lens
git clone https://github.com/igorbarshteyn/jlens-gguf repos/jlens-gguf
git -C repos/jlens-gguf checkout 16e5526
git -C repos/jlens-gguf apply ../../patches/jlens-gguf-local.patch
LLAMA_DIR=/pfad/zu/llama.cpp repos/jlens-gguf/native/build.sh   # libllama vorher mit CUDA bauen

# Python
uv venv .venv && uv pip install -p .venv/bin/python -e repos/jlens-gguf -e ".[dev]"
.venv/bin/python -m pytest

# Modelle
hf download XHToken/Spark-X2.6-4B-GGUF Spark-X2.6-4B.gguf Spark-X2.6-4B-Q8_0.gguf --local-dir models
export LLAMA_QUANTIZE=/pfad/zu/llama-quantize
scripts/make_flat_q4.sh              # Q4_0
scripts/make_quant_variants.sh       # Erweiterung: Leiter + Ablation
.venv/bin/python scripts/quant_layout.py models/<datei>.gguf   # Typ pro Tensor prüfen

# Lenses (~40 min je Lens) und Analyse
scripts/fit_lenses.sh
.venv/bin/python -m jspace_quant capture --gpus 0,1   # nur noch nicht erfasste Modelle
.venv/bin/python -m jspace_quant analyze              # nur noch nicht berechnete Arme
.venv/bin/python -m jspace_quant plot                 # Abbildungen auf Englisch und Deutsch

# Notebooks
.venv/bin/python scripts/build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.de.ipynb
```

`build.sh` baut ohne `LLAMA_DIR` gegen das gepinnte Submodul, das `spark2_5` nicht kennt; existiert
`libllama` noch nicht, baut das Skript es ohne CUDA. `JSPACE_SMOKE=1` lässt die Pipeline mit wenigen
Items nach `results/smoke/` laufen. `JSPACE_GPU`/`JSPACE_GPUS` wählen die Grafikkarten. Die gewählten
GPUs müssen frei sein.

## Laufzeiten

| Schritt | Dauer |
|---|---|
| Quantisieren | Sekunden bis 4,5 min je Modell |
| Lens-Fit | ~40 min je Lens (CPU-gebunden: Gram-Matrizen in float64) |
| Capture | ~1 min je Modell |
| Analyse | ~35 min für die 11 Arme der Hauptstudie, ~50 min für die 18 Arme der Erweiterung (CPU) |

## Verzeichnisse

| Pfad | Inhalt | im Repo |
|---|---|---|
| `src/jspace_quant/`, `tests/`, `scripts/` | Code | ja |
| `data/corpus/`, `patches/` | Korpora, Patches | ja |
| `results/tables/`, `results/figures/{en,de}/`, `results/logs/` | Ergebnisse | ja |
| `notebooks/walkthrough.{en,de}.ipynb` | Walkthrough mit Ergebnissen | ja |
| `results/activations/` (~7 GB), `lenses/` (1,8 GB), `models/` (~50 GB) | Zwischendaten | nein, reproduzierbar |
