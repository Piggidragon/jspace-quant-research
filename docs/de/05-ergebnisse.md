# 05 · Ergebnisse und Reflexion

Alle Zahlen aus `results/tables/`, Abbildungen in `results/figures/de/`. KIs: 95-%-Bootstrap, gepaart
gegen die Referenz. „Kern“ = die vier Eval-Sets, die nicht schon auf BF16 am Boden liegen
(`multihop`, `multilingual`, `order-ops`, `typo`; 351 Items).

## Kurzfassung

1. **8 bit ändert nichts Messbares.** Q8_0 weicht von BF16 weniger ab als eine zweite BF16-Lens,
   die nur auf anderem Text gefittet wurde.
2. **Die verborgenen Konzepte halten bis 4 bit,** bröckeln bei 3 bit und brechen bei 2 bit zusammen.
3. **Die gesamte Readout-Verteilung reagiert viel früher:** Q6_K liegt noch auf Höhe der Rauschgrenze, ab
   5 bit driftet sie deutlich — monoton mit sinkender Bitbreite und wachsend mit der Schichttiefe.
4. **Die Drift entsteht im Residual Stream,** nicht im quantisierten Unembedding, und keine einzelne
   Komponente erklärt sie allein — Attention, MLP und Embedding tragen jeweils bei.
5. **Bei 4 bit ist der Raum verschoben, nicht zerstört:** Eine neu gefittete Lens holt verlorene
   Konzepte zurück und sagt das Verhalten des quantisierten Modells genauso gut vorher wie beim
   Original.

## 1. Workspace-Band und Positivkontrolle

Die verborgenen Konzepte werden in den Schichten **L22–L34** lesbar (Trefferrate der Referenz ≥ 50 %
ihres Maximums, Spitze ~19 % bei L26–L30). Davor sind sie praktisch nie unter den Top-10
(`hit_rate_per_layer.png`). Das entspricht dem Bild des Papers.

Die **Positivkontrolle ist gemischt**: Auf `multihop` (+0,075) und `order-ops` (+0,109) schlägt
unsere Lens den Logit Lens, auf `typo` verliert sie deutlich (0,44 vs. 0,94), im Kern-Aggregat liegt
der Logit Lens vorn (0,54 vs. 0,43). Im Workspace-Band allein sind beide gleichauf (0,31).

## 2. Hauptstudie: BF16 → Q8_0 → Q4_0

**Konzepte** (pass@10 über alle Schichten, Differenz zur Referenz 0,43):

| Arm | Kern | auffällig |
|---|---|---|
| Rauschgrenze | ±0,000 | `order-ops` −0,082 (signifikant!) |
| Q8_0, feste Lens | +0,005 [−0,003; +0,014] | — |
| Q4_0, feste Lens | −0,019 [−0,047; +0,009] | `typo` −0,115 [−0,198; −0,042] |
| Q4_0, BF16-Unembedding | −0,019 | `typo` −0,094 |
| Q4_0, neu gefittet | +0,007 [−0,023; +0,038] | `typo` −0,031 (n. s.) |

**Verteilung** (Mittel über das Workspace-Band, Held-out-Text):

| Arm | Top-1 = Referenz | KL [nats] | Fidelity |
|---|---|---|---|
| Rauschgrenze | 0,915 | 0,043 | 0,668 |
| Q8_0, feste Lens | 0,949 | 0,036 | 0,668 |
| Q4_0, feste Lens | 0,649 | 0,904 | 0,668 |
| Q4_0, BF16-Unembedding | 0,655 | 0,897 | 0,666 |
| Q4_0, neu gefittet | 0,648 | 0,914 | 0,676 |

Pro Schicht (`agreement_per_layer_heldout.png`) beginnt Q4_0 nahe an der Referenz (0,90 in L0),
fällt ab etwa L10 unter die Rauschgrenze und landet im Band bei ~0,65. Q8_0 bleibt überall über der
Rauschgrenze. Die Rauschgrenze selbst ist in frühen Schichten am niedrigsten (0,78): Dort ist die Lens
am schlechtesten bestimmt.

**Verhalten:** Q8_0 unverändert (Perplexity ×1,02). Q4_0: Perplexity ×1,46 [1,08; 1,94],
`multilingual` −9 Punkte [−17; −3].

## 3. Bitbreiten-Leiter

| Stufe | Bits/Gewicht | Top-1 = Ref. (Band) | pass@10 Kern Δ | Perplexity | rel. Fehler h (Band) |
|---|---|---|---|---|---|
| Rauschgrenze | — | 0,915 | — | — | — |
| Q8_0 | 8,5 | 0,949 | +0,005 | ×1,02 | 0,08 |
| Q6_K | 6,6 | 0,909 | +0,000 | ×0,97 | 0,15 |
| Q5_K | 5,5 | 0,825 | −0,022* | ×0,98 | 0,25 |
| Q5_0 | 5,5 | 0,802 | −0,009 | ×1,31* | 0,28 |
| Q4_K | 4,5 | 0,711 | +0,005 | ×1,33* | 0,37 |
| Q4_0 | 4,5 | 0,649 | −0,019 | ×1,46* | 0,50 |
| Q3_K | 3,4 | 0,501 | **−0,055*** | ×2,81* | 0,69 |
| Q2_K | 2,6 | 0,102 | **−0,258*** | ×393* | 1,15 |

\* KI schließt 0 bzw. 1 aus.

- **Verteilung:** monoton mit der Bitbreite; ab Q6_K auf Höhe der Rauschgrenze, darunter klar
  schlechter (`dose_response.png`).
- **Konzepte:** bis 4 bit innerhalb des Rauschens, ab 3 bit signifikant weniger, bei 2 bit
  Zusammenbruch (`multilingual` 0,41 → 0,01).
- **K- vs. Legacy-Quants:** Bei gleicher Bitbreite weicht das K-Schema weniger ab (Q4_K 0,71 vs.
  Q4_0 0,65; Q5_K 0,83 vs. Q5_0 0,80) — die feinere Skalierung zahlt sich auch intern aus.
- **Ohne Lens:** Der relative Fehler des Residual Streams steigt von Schicht 0 bis ~L12 an und bleibt
  dann etwa konstant (`ladder_per_layer.png`); die Lens-Übereinstimmung folgt demselben Verlauf.

## 4. Komponenten-Ablation

Nur ein Teil des Modells in Q4_0, alles andere BF16 (`ablation.png`):

| Variante | Top-1 = Ref. (Band) | rel. Fehler h (Band) | Perplexity | pass@10 Kern Δ |
|---|---|---|---|---|
| nur Embedding | 0,834 | 0,18 | ×1,10 | −0,008 |
| nur Attention | 0,766 | 0,32 | ×1,15 | −0,019 |
| nur MLP | 0,774 | 0,31 | ×1,22* | −0,009 |
| alles (Q4_0) | 0,649 | 0,50 | ×1,46* | −0,019 |

Attention und MLP tragen etwa gleich viel bei, das Embedding weniger; keine Komponente allein erreicht
die Drift des vollen Q4_0 — die Fehler addieren sich. Schon das quantisierte Embedding allein drückt
die Übereinstimmung unter die Rauschgrenze, obwohl es nur einen Tensor betrifft: Sein Fehler steckt
im Eingang jeder Schicht.

## 5. Welche Konzepte leiden?

Unter den Konzepten, die die Referenz im Band liest (Rang ≤ 10, n = 249), Anteil, der verloren geht:

| | selten | eher selten | eher häufig | häufig |
|---|---|---|---|---|
| Rauschgrenze | 0,14 | 0,02 | 0,02 | 0,02 |
| Q4_0 | 0,21 | 0,10 | 0,09 | 0,07 |
| Q3_K | 0,35 | 0,40 | 0,14 | 0,17 |
| Q2_K | 0,90 | 0,98 | 0,98 | 0,73 |

Seltene Konzepte gehen eher verloren — aber sie schwanken schon zwischen zwei BF16-Lenses am
stärksten. Ein klarer Häufigkeitseffekt über das Rauschen hinaus zeigt sich erst bei 3 bit.

## 6. Hypothesen

| | Ergebnis |
|---|---|
| H1 | Q8_0: ja. 4 bit: Konzepte ja, Verteilung nein. ≤ 3 bit: nein. |
| H2 | Ja: Verteilung monoton über die ganze Leiter; Konzepte erst ab 3 bit messbar. |
| H3 | Teilweise: Refit holt Konzepte zurück und hält die Fidelity; die Abweichung vom BF16-Readout bleibt, weil das Modell selbst anders rechnet. |
| H0 | Für Q8_0 (und Q6_K) beibehalten, ab 5 bit verworfen. |

## Reflexion

**Flache statt gebräuchlicher Quants.** In der Praxis läuft fast jeder mit gemischten Quants wie
`Q4_K_M`, oft mit imatrix. Wir haben bewusst flache Quants gewählt, damit die Bitbreite die einzige
Variable ist und Schichtkurven nicht von einem schichtweise wechselnden Bit-Budget verfälscht werden.
Der Preis: Unsere 4-bit-Modelle sind härter als das, was tatsächlich eingesetzt wird. Q4_0
quantisiert sogar das Embedding auf 4 bit, das `Q4_K_M` unquantisiert lässt (Unembedding als Q8_0),
und die Ablation zeigt, dass genau das Embedding allein messbar beiträgt. Unsere 4-bit-Ergebnisse sind daher eher eine obere Schranke für den
Schaden; ob gemischte und imatrix-Quants die Konzepte besser schützen, haben wir nicht getestet.

**Surrogat statt echter Jacobian Lens.** Ridge-Regression lernt, welche Richtung von `h_l` mit `h_35`
*korreliert*, nicht, welche sie *verursacht*. Die Folge sieht man an der gemischten Positivkontrolle:
Auf `typo` ist unsere Lens schlechter als der Logit Lens. Für den Vergleich zwischen Quants ist das
konsistent, weil alle Lenses gleich gefittet sind; ob eine echte Jacobian Lens anders auf
Quantisierung reagiert, bleibt offen.

**Die Konzeptmetrik ist nachsichtig und verrauscht.** pass@k zählt den *besten* Rang über 36
Schichten; eine Verschiebung um ein paar Schichten bleibt unsichtbar. Gleichzeitig schwankt sie schon
zwischen zwei BF16-Lenses um bis zu 0,08 pro Set. Effekte unter ~0,05 sind mit 55–107 Items pro Set
nicht belastbar. Deshalb haben wir die Verteilungsmetriken ergänzt — sie zeigen die Drift viel früher.

**Verteilung vs. Konzepte.** Dass die Verteilung bei 4 bit deutlich driftet, die gesuchten Konzepte
aber bleiben, ist plausibel, wenn Quantisierungsrauschen vor allem schwache Richtungen und häufige
Füllwörter verschiebt (frühe Top-1 sind oft „ the“), während die starken, aufgabenrelevanten
Richtungen im Band erhalten bleiben. Belegt haben wir diesen Mechanismus nicht.

**Auswertung nach Referenz-Stärke.** Gruppiert man Konzepte danach, wie gut die Referenz sie liest,
können gut gelesene nur schlechter und schlecht gelesene nur besser werden (Regression zur Mitte).
Solche Aufschlüsselungen sind nur gegen die Rauschgrenze interpretierbar; deshalb steht sie in jeder
Tabelle.

**CKA mit Vorsicht.** Q8_0 hat einen Kosinus von 0,995 zur Referenz, aber nur CKA 0,91 — vermutlich,
weil das lineare CKA von wenigen Richtungen mit sehr großer Varianz dominiert wird. Wir berichten es, stützen
Aussagen aber auf den relativen Fehler und die Readouts.

**Prompt-Format.** Die Eval-Sets sind für Textvervollständigung geschrieben; Spark ist ein
Chat-Modell. Das Chat-Format haben wir nur anhand von BF16 gewählt, bevor es Quant-Ergebnisse gab.
Trotzdem gelten die Zahlen für dieses Format; `association` und `poetry` sind damit auf diesem
Modell nicht messbar.

**Ein Modell.** Alle Aussagen betreffen Spark-X2.6-4B. Laut Literatur sind kleine Modelle
empfindlicher; größere könnten robuster sein.
