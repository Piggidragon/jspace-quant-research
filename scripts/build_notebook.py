"""Build notebooks/walkthrough.{en,de}.ipynb: method and results step by step, computed on the stored
activations and result tables (no server, no GPU needed).

    .venv/bin/python scripts/build_notebook.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.en.ipynb
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.de.ipynb
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

# (markdown_en, markdown_de) or ("code", source)
CELLS = [
    ("""# J-space readouts under weight quantization — walkthrough

**Question:** Does the global-workspace signal that Anthropic's Jacobian lens reads from a language
model's intermediate layers survive quantization?

This notebook walks through the method and shows the results. It runs on the stored activations
(`results/activations/`) and result tables (`results/tables/`); the pipeline behind it is
`python -m jspace_quant capture | analyze | plot` (see `docs/en/03-setup.md`). Full write-up:
`docs/en/02-methodology.md`, `docs/en/05-results.md`.""",
     """# J-Space-Readouts unter Gewichtsquantisierung — Walkthrough

**Frage:** Bleibt das Global-Workspace-Signal, das Anthropics Jacobian Lens aus den Zwischenschichten
eines Sprachmodells liest, unter Quantisierung erhalten?

Dieses Notebook geht Methode und Ergebnisse Schritt für Schritt durch. Es rechnet auf den gespeicherten
Aktivierungen (`results/activations/`) und Ergebnistabellen (`results/tables/`); dahinter steht die
Pipeline `python -m jspace_quant capture | analyze | plot` (siehe `docs/de/03-setup.md`). Ausführlich:
`docs/de/02-methodik.md`, `docs/de/05-ergebnisse.md`."""),
    ("code", """
import json
import numpy as np
import pandas as pd
from IPython.display import Image, display

from jspace_quant import analysis, capture, config, readout

pd.set_option("display.precision", 3)
T, F = config.TABLES_DIR, config.FIGURES_DIR / LANG
summary = json.loads((T / "summary.json").read_text())
band = tuple(summary["workspace_band"])
"""),
    ("""## 1. Model and quantizations

Spark-X2.6-4B (36 layers, residual width 2560, hybrid sliding-window/full attention). We only use
**flat** quants — every weight tensor has the same type — so that bit width is the only variable.

- **Main study:** BF16 (reference) → Q8_0 → Q4_0
- **Extension:** a bit-width ladder in two families (legacy: Q5_0; K-quants: Q6_K … Q2_K) and a
  component ablation (attention / MLP / embedding alone in Q4_0)

Bits per weight are measured from the GGUF files:""",
     """## 1. Modell und Quantisierungen

Spark-X2.6-4B (36 Schichten, Residual-Breite 2560, hybride Sliding-Window-/Full-Attention). Wir verwenden
nur **flache** Quants — jeder Gewichtstensor hat denselben Typ —, damit die Bitbreite die einzige
Variable ist.

- **Hauptstudie:** BF16 (Referenz) → Q8_0 → Q4_0
- **Erweiterung:** Bitbreiten-Leiter in zwei Familien (Legacy: Q5_0; K-Quants: Q6_K … Q2_K) und
  Komponenten-Ablation (nur Attention / MLP / Embedding in Q4_0)

Bits pro Gewicht sind aus den GGUF-Dateien gemessen:"""),
    ("code", """
pd.read_csv(T / "models.csv")[["model", "family", "bits_per_weight", "tensor_types"]]
"""),
    ("""## 2. What the lens does

For a layer `l` and a position, take the residual-stream vector `h_l` and read it out:

```
logits = Unembed( RMSNorm( A_l · h_l + b_l ) )
```

`A_l` maps the space of layer `l` into the space of the last layer. With `A_l = I` this is the classic
**logit lens**. The paper defines `A_l` as the averaged Jacobian `E[∂h_final/∂h_l]`. llama.cpp has no
autograd, so we fit `A_l, b_l` by **ridge regression** of `h_35` on `h_l` over 1000 WikiText paragraphs
(forward passes only — a correlational surrogate in the tuned-lens family).

**Live example** on the stored activations: the French prompt "Après la nuit vient le" ("after the
night comes the") — answer "jour". The text contains the French "nuit"; the question is whether the
model internally forms the *English* concept "night", and at which layer it becomes readable.""",
     """## 2. Was die Lens tut

Für eine Schicht `l` und eine Position nimmt man den Residual-Stream-Vektor `h_l` und liest ihn aus:

```
logits = Unembed( RMSNorm( A_l · h_l + b_l ) )
```

`A_l` bildet den Raum von Schicht `l` in den Raum der letzten Schicht ab. Mit `A_l = I` ist das der
klassische **Logit Lens**. Das Paper definiert `A_l` als gemittelte Jacobi-Matrix `E[∂h_final/∂h_l]`.
llama.cpp hat kein Autograd, deshalb fitten wir `A_l, b_l` per **Ridge-Regression** von `h_35` auf `h_l`
über 1000 WikiText-Absätze (nur Forward-Pässe — ein korrelatives Surrogat aus der Tuned-Lens-Familie).

**Live-Beispiel** auf den gespeicherten Aktivierungen: der französische Prompt „Après la nuit vient le“
(„nach der Nacht kommt der“) — Antwort „jour“. Im Text steht das französische „nuit“; die Frage ist, ob
das Modell intern das *englische* Konzept „night“ bildet und ab welcher Schicht es lesbar wird."""),
    ("code", """
acts, meta = capture.load("bf16", "evals")
i = next(k for k, it in enumerate(meta) if it["name"] == "fr-time-after-night")
item = meta[i]
print(item["prompt"].split("assistant\\n")[-1], "→ target:", item["target"])
night = next(c for c in item["intermediates"] if c["word"] == "night")["token_ids"]

rows = []
for arm_name in ["bf16|identity", "bf16|bf16-A"]:
    arm = config.ARMS_BY_NAME[arm_name]
    for layer in range(0, 36, 3):
        logits = readout.arm_logits(arm, layer, acts[i, layer][None])
        rows.append({"readout": "logit lens" if arm.lens is None else "lens", "layer": layer,
                     "rank of 'night'": int(readout.min_rank(logits, night)[0])})
pd.DataFrame(rows).pivot(index="layer", columns="readout", values="rank of 'night'")
"""),
    ("""With the logit lens "night" stays far back; with the fitted lens it moves into the top 10 in the
late-middle layers, before the model answers. That is the global-workspace signal.

## 3. How do we measure whether quantization changes anything?

Each **experimental arm** combines one model's activations, a lens and an unembedding:""",
     """Mit dem Logit Lens bleibt „night“ weit hinten; mit der gefitteten Lens rückt es in den späten mittleren
Schichten unter die Top 10, bevor das Modell antwortet. Das ist das Global-Workspace-Signal.

## 3. Wie misst man, ob Quantisierung etwas ändert?

Jeder **Versuchsarm** kombiniert die Aktivierungen eines Modells, eine Lens und ein Unembedding:"""),
    ("code", """
pd.DataFrame([{"arm": a.name, "model": a.model, "lens": a.lens or "identity", "unembedding": a.unembed,
               "role": a.role} for a in analysis.available_arms() if a.model in config.MAIN_MODELS])
"""),
    ("""- **fixed lens** (BF16 lens on the quant): did the space *stay the same*?
- **refitted** (lens fitted on the quant): is it *still linearly readable*?
- **BF16 unembedding**: separates drift in the residual stream from a less precise 4-bit readout layer.
- **noise floor**: a second BF16 lens, fitted on a different corpus only. Every quant effect must be
  clearly larger.

**Why the noise floor matters:** comparing the lens *matrices*, the two BF16 lenses differ more than the
BF16 and Q4_0 lenses. When *reading out* real activations they largely agree — the matrices are barely
determined in directions where activations don't vary. So we only measure readouts.""",
     """- **feste Lens** (BF16-Lens auf dem Quant): Ist der Raum *gleich geblieben*?
- **neu gefittet** (Lens auf dem Quant gefittet): Ist er *noch linear lesbar*?
- **BF16-Unembedding**: trennt Drift im Residual Stream von einer ungenaueren 4-bit-Ausleseschicht.
- **Rauschgrenze**: zweite BF16-Lens, nur auf anderem Korpus gefittet. Jeder Quant-Effekt muss deutlich
  darüber liegen.

**Warum die Rauschgrenze so wichtig ist:** Vergleicht man die *Matrizen*, unterscheiden sich die beiden
BF16-Lenses stärker als BF16- und Q4_0-Lens. Beim *Auslesen* echter Aktivierungen sind sie sich aber
weitgehend einig — die Matrizen sind in Richtungen ohne Aktivierungsvarianz kaum festgelegt. Deshalb
messen wir nur Readouts."""),
    ("code", """
A, B, Q4 = (readout.lens(n) for n in ["bf16-A", "bf16-B", "q4_0-A"])
def rel(L1, L2, l):
    return np.linalg.norm(L1.jacobians[l] - L2.jacobians[l]) / np.linalg.norm(L1.jacobians[l])
pd.DataFrame({"layer": [0, 10, 20, 30],
              "matrix distance A↔B (corpus only)": [rel(A, B, l) for l in (0, 10, 20, 30)],
              "matrix distance A↔Q4_0 lens": [rel(A, Q4, l) for l in (0, 10, 20, 30)]})
"""),
    ("""## 4. Metrics

- **pass@k** (the paper's metric) on its six eval sets: share of hidden intermediate concepts whose best
  rank across layers is ≤ k. `association` and `poetry` are already at floor on BF16 and excluded from
  hypothesis testing; the **core** aggregate covers the other four sets.
- **Agreement with the reference per layer:** top-1 equal, top-10 overlap, KL divergence.
- **Fidelity:** does the readout predict the model's *own* output?
- **Behavior check:** answer accuracy, perplexity.
- **Statistics:** 95 % bootstrap CIs, paired against the reference, clustered by paragraph on text.

## 5. Where do the concepts become readable?""",
     """## 4. Metriken

- **pass@k** (Paper-Metrik) auf den sechs Eval-Sets: Anteil der verborgenen Zwischenkonzepte, deren bester
  Rang über die Schichten ≤ k ist. `association` und `poetry` liegen schon auf BF16 am Boden und gehen
  nicht in die Hypothesen ein; das **Kern**-Aggregat umfasst die übrigen vier Sets.
- **Übereinstimmung mit der Referenz pro Schicht:** Top-1 gleich, Top-10-Overlap, KL-Divergenz.
- **Fidelity:** Sagt der Readout den Output des *eigenen* Modells voraus?
- **Verhaltenskontrolle:** Antwortgenauigkeit, Perplexity.
- **Statistik:** 95-%-Bootstrap-KIs, gepaart gegen die Referenz, bei Fließtext über Absätze geclustert.

## 5. Wo werden die Konzepte lesbar?"""),
    ("code", """
print("workspace band:", band, "| floor sets:", summary["floor_sets"])
display(Image(F / "hit_rate_per_layer.png", width=900))
display(Image(F / "concept_trajectories.png", width=1100))
"""),
    ("""## 6. Do the concepts survive? (pass@10, core)""",
     """## 6. Überleben die Konzepte? (pass@10, Kern)"""),
    ("code", """
pak = pd.read_csv(T / "pass_at_k.csv")
tab = pak[(pak.k == 10) & (pak.set == config.AGG_CORE) & (pak.scope == config.SCOPE_ALL) & (pak.role == "fixed")]
tab[["arm", "pass_at_k", "delta_ref", "delta_ci_lo", "delta_ci_hi"]].reset_index(drop=True)
"""),
    ("""## 7. The readout distribution drifts much earlier""",
     """## 7. Die Readout-Verteilung driftet viel früher"""),
    ("code", """
display(Image(F / "dose_response.png", width=1100))
display(Image(F / "agreement_per_layer_heldout.png", width=1100))
"""),
    ("""## 8. Representations without a lens, per layer""",
     """## 8. Repräsentationen ohne Lens, pro Schicht"""),
    ("code", """
display(Image(F / "ladder_per_layer.png", width=1100))
"""),
    ("""## 9. Which component causes the drift?""",
     """## 9. Welche Komponente verursacht die Drift?"""),
    ("code", """
display(Image(F / "ablation.png", width=1100))
"""),
    ("""## 10. Which concepts suffer?""",
     """## 10. Welche Konzepte leiden?"""),
    ("code", """
display(Image(F / "concept_loss.png", width=1100))
"""),
    ("""## 11. Causal test: do lens directions still steer the model?

On the paper's `probe-swap` set (e.g. "the language spoken in the country where the Amazon River ends
is" → Brazil → Portuguese) we steer the residual stream at L18–L25 away from the bridge concept and
towards a swap concept (Mexico) and check whether the answer follows (Spanish).""",
     """## 11. Kausaler Test: Steuern die Lens-Richtungen das Modell noch?

Auf dem Paper-Set `probe-swap` (z. B. „the language spoken in the country where the Amazon River ends
is“ → Brazil → Portuguese) lenken wir den Residual Stream in L18–L25 vom Brückenkonzept weg und zu einem
Tauschkonzept (Mexico) hin und prüfen, ob die Antwort folgt (Spanish)."""),
    ("code", """
iv = pd.read_csv(T / "interventions_summary.csv")
display(iv[["model", "lens", "baseline_accuracy", "steer_success", "steer_success_vs_ref",
            "steer_success_vs_ref_ci_lo", "steer_success_vs_ref_ci_hi", "steer_control", "ablate_drop"]])
display(Image(F / "interventions.png", width=1100))
"""),
    ("""## 12. Behavior check""",
     """## 12. Verhaltenskontrolle"""),
    ("code", """
beh = pd.read_csv(T / "behavior.csv")
beh[beh.metric == "heldout_ppl"][["model", "value", "vs_ref", "vs_ref_ci_lo", "vs_ref_ci_hi"]].reset_index(drop=True)
"""),
    ("""## 13. Conclusion

- **8 bit** changes nothing measurable.
- **Down to 4 bit** the hidden concepts stay readable, although the readout distribution drifts
  noticeably from ~6 bit on, growing with depth. A refitted lens recovers losses: the space is
  shifted, not destroyed.
- **3 bit** costs concepts, **2 bit** destroys them.
- Steering along the lens directions works causally on the quantized models down to 4 bit as well.

Limitations and reflection: `docs/en/05-results.md`.""",
     """## 13. Fazit

- **8 bit** ändert nichts Messbares.
- **Bis 4 bit** bleiben die verborgenen Konzepte lesbar, obwohl die Readout-Verteilung schon ab ~6 bit
  deutlich und mit der Tiefe wachsend driftet. Eine neu gefittete Lens holt Verluste zurück: Der Raum ist
  verschoben, nicht zerstört.
- **3 bit** kostet Konzepte, **2 bit** zerstört sie.
- Steering entlang der Lens-Richtungen wirkt bis 4 bit auch auf den quantisierten Modellen kausal.

Grenzen und Reflexion: `docs/de/05-ergebnisse.md`."""),
]


def build(lang: str) -> Path:
    cells = [nbf.v4.new_code_cell(f'LANG = "{lang}"')]
    for a, b in CELLS:
        if a == "code":
            cells.append(nbf.v4.new_code_cell(b.strip()))
        else:
            cells.append(nbf.v4.new_markdown_cell((a if lang == "en" else b).strip()))
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
    out = ROOT / "notebooks" / f"walkthrough.{lang}.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    return out


if __name__ == "__main__":
    for lang in ("en", "de"):
        print("written:", build(lang))
