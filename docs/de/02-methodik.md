# 02 · Methodik

Begründungen für die einzelnen Entscheidungen: [`04-entscheidungen.md`](04-entscheidungen.md).
Ausführung: [`03-setup.md`](03-setup.md). Ergebnisse: [`05-ergebnisse.md`](05-ergebnisse.md).

## 1. Frage und Hypothesen

Wie robust sind die Readouts der Jacobian Lens gegenüber Gewichtsquantisierung?

| | Hypothese | geprüft über |
|---|---|---|
| H1 | Die auf BF16 gefittete Lens liest auf Q8_0/Q4_0 dieselben Zwischenkonzepte. | pass@k der festen Lens vs. Referenz |
| H2 | Die Abweichung wächst monoton mit stärkerer Quantisierung. | Agreement-Metriken über die Stufen |
| H3 | Eine neu auf dem Quant gefittete Lens stellt das Signal wieder her (Raum verschoben, nicht zerstört). | neu gefittete vs. feste Lens |
| H0 | Quantisierung ändert nichts Messbares. | alle Differenzen innerhalb der Rauschgrenze |

Erwartung aus der Literatur ([`01-hintergrund.md`](01-hintergrund.md)): Q8_0 ohne Effekt, Q4_0 mit
deutlicher, monotoner Drift, schwache Konzepte zuerst.

## 2. Material

**Modelle** — nur flache Quants, d. h. jeder Gewichtstensor hat denselben Typ:

| Schlüssel | Typ | Bits/Gewicht | Herkunft |
|---|---|---|---|
| `bf16` | BF16 | 16,0 | offiziell, Referenz |
| `q8_0` | Q8_0 | 8,5 | offiziell |
| `q4_0` | Q4_0 | 4,5 | selbst erzeugt (`scripts/make_flat_q4.sh`) |

Erweiterung (§6): Q5_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K und drei Ablationsmodelle
(`scripts/make_quant_variants.sh`).

**Fitting-Korpus:** WikiText-103, eingefroren in `data/corpus/`. Korpus A = die ersten 1000 Absätze
mit ≥ 600 Zeichen (alle Lenses), Korpus B = die nächsten 1000 (nur Rausch-Lens).

**Testdaten:**
- die sechs Eval-Sets des Papers (`anthropics/jacobian-lens/data/evaluations`, 551 Items):
  `multihop`, `multilingual`, `order-ops`, `association`, `typo`, `poetry`. Jedes Item hat einen
  Prompt und ein oder mehrere **Zwischenkonzepte**, die im Text nicht stehen.
- 40 WikiText-Absätze, die in keinem Korpus vorkommen (1120 Positionen), für Verteilungsvergleiche
  und Perplexity.

## 3. Lens-Fitting

Pro Schicht `l` eine Ridge-Regression des Residual Streams der letzten Schicht auf den von Schicht `l`:

```
A_l, b_l = argmin Σ_p ‖A·h_l[p] + b − h_35[p]‖² + λ‖A‖²,   λ = 1e-4 · trace(Cov)/d
```

über alle Positionen der 1000 Absätze (gekürzt auf 128 Tokens, erste 16 und letzte Position
ausgeschlossen ⇒ 110 906 Positionen). Vier Lenses, alle mit identischen Parametern:

| Lens | Modell | Korpus | Rolle |
|---|---|---|---|
| `bf16-A` | BF16 | A | Referenz-Lens |
| `q8_0-A` | Q8_0 | A | neu gefittet |
| `q4_0-A` | Q4_0 | A | neu gefittet |
| `bf16-B` | BF16 | B | Rauschgrenze |

## 4. Versuchsarme

Ein Arm kombiniert die Aktivierungen eines Modells M, eine Lens und das Unembedding eines Modells U:

```
logits(l, p) = Unembed_U( RMSNorm( A_l · h_l^M[p] + b_l ) )
```

| Arm | M | Lens | U | Zweck |
|---|---|---|---|---|
| Referenz | BF16 | bf16-A | BF16 | Bezugspunkt |
| Rauschgrenze | BF16 | bf16-B | BF16 | Abweichung allein durch das Fitting |
| Logit Lens | M | — | M | Baseline, Positivkontrolle |
| feste Lens | Q | bf16-A | Q | Ist der Raum gleich geblieben? (H1, H2) |
| feste Lens, BF16-Unembedding | Q | bf16-A | BF16 | Drift im Residual Stream oder im 4-bit-Unembedding? |
| neu gefittet | Q | q-A | Q | Ist der Raum noch linear lesbar? (H3) |

Die letzte Schicht (35) hat keine Lens-Matrix und wird immer mit Identität gelesen — das ist genau
der Output des Modells.

**Prompt-Format:** Spark ist ein Chat-Modell. Jeder Eval-Prompt steht als begonnene
Assistenten-Antwort im Chat-Template (Anweisung „Continue the text.“), abschließende Leerzeichen
entfernt. Held-out-Text bleibt roh wie beim Fitting.

## 5. Metriken und Statistik

**Konzepte (Paper-Metrik):** Readout-Position ist das letzte Prompt-Token (bei `poetry` das Ende
der ersten Zeile). Rang eines Konzepts = bester Rang über seine Ein-Token-Schreibweisen (bei
`order-ops` zusätzlich Synonyme wie „5“/„five“, „*“/„times“).
- **pass@k** = Mittel über Items des Anteils der Konzepte, deren bester Rang über die Schichten ≤ k
  ist (k = 1, 5, 10); einmal über alle Schichten, einmal nur im Workspace-Band.
- **Trefferrate pro Schicht** = Anteil der Konzepte mit Rang ≤ 10 in dieser Schicht.
- **Workspace-Band** = zusammenhängender Bereich um das Maximum der Referenz-Trefferrate, in dem sie
  ≥ 50 % des Maximums ist (ohne letzte Schicht).

**Verteilung**, pro Arm, Schicht und Position gegen den Referenz-Readout: Top-1 gleich,
Top-10-Overlap, KL(Referenz ‖ Arm).

**Fidelity:** Top-1 des Readouts = Output des eigenen Modells.

**Verhalten:** Anteil korrekter greedy Antworten (`multihop`, `multilingual`, `order-ops`) und
Perplexity auf dem Held-out-Text.

**Statistik:**
- 95-%-Bootstrap-KIs (2000 Ziehungen); bei Held-out-Metriken über Absätze geclustert.
- Vergleiche zur Referenz gepaart (dieselben Items/Positionen).
- Ein Effekt zählt, wenn die KI die 0 ausschließt *und* er deutlich größer ist als bei der Rausch-Lens.
- **Boden-Sets:** Sets, auf denen schon die Referenz pass@10 < 0,10 hat, gehen nicht in H1–H3 ein
  (festgelegt vor dem Hauptlauf). Das Aggregat **„Kern“** mittelt über die übrigen.

## 6. Erweiterung

Nach dem Hauptlauf ergänzt, nur mit der festen BF16-Lens ausgewertet:

- **A — Repräsentationen ohne Lens:** pro Schicht relativer Fehler `‖h_q − h_BF16‖/‖h_BF16‖`,
  Kosinus und lineares CKA gegenüber BF16.
- **B — Welche Konzepte leiden?** Rang-Änderung im Band nach Stärke in der Referenz und nach
  Token-Häufigkeit in Korpus A.
- **C — Komponenten-Ablation:** nur Attention, nur MLP oder nur Embedding in Q4_0, Rest BF16.
- **D — Bitbreiten-Leiter:** Legacy-Familie (Q8_0, Q5_0, Q4_0) und K-Familie (Q6_K bis Q2_K), alle flach.

## 7. Grenzen

- **Surrogat-Lens:** Regression statt kausaler Jacobi-Matrix; konsistent über alle Quants, aber
  nicht die Lens des Papers.
- **Ein Modell, ein Prompt-Format, nur flache Quants.**
- **Rauschen:** Die Konzeptmetrik schwankt schon zwischen zwei BF16-Lenses um bis zu 0,08 pro Set;
  in frühen Schichten weichen ihre Readouts bei ~20 % der Positionen ab.
- **Konzept-Tokenisierung:** 6 % der Konzepte haben keine Ein-Token-Form (Fallback: erstes Token).
