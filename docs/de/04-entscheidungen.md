# 04 · Methodische Entscheidungen

Was wir unterwegs gefunden haben und wie es das Design verändert hat — in der Reihenfolge, in der
es auftrat. Die Entscheidungen §1–§5 und §7 fielen **vor** dem jeweiligen Hauptlauf, ohne Blick auf Ergebnisse der
quantisierten Modelle; §6 ist eine Korrektur der Statistik nach dem ersten Hauptlauf.

## 1. Spark läuft erst mit Patches

Das in jlens-gguf gepinnte llama.cpp kennt die Architektur `spark2_5` nicht. Mit dem aktuellen
llama.cpp und Patch 1 ([`03-setup.md`](03-setup.md)) lädt das Modell; der Python-Readout stimmt mit
den Logits von llama.cpp überein (Korrelation 0,99998, gleicher Top-Token).

Beim ersten Readout am BF16-Modell tauchten Token-IDs über der Vokabulargröße auf (143 784 > 131 072).
Ursache war ein Bug in jlens-gguf beim Einlesen von BF16-Gewichten ⇒ Patch 2.

## 2. Referenz ist BF16, nicht F16

Die als „F16“ bezeichnete offizielle Datei ist laut Header BF16 — das native Format des Modells,
also eine verlustfreie Referenz.

## 3. Nur flache Quants

Das offizielle `Q4_K_M` mischt Typen pro Schicht (Q6_K/Q4_K, einzelne Tensoren Q8_0/BF16). Eine
Abweichung in Schicht 4 könnte dann schlicht daher kommen, dass Schicht 4 weniger Bits hat als
Schicht 3. ⇒ Wir verwenden nur **flache** Quants (jeder Tensor gleicher Typ) und erzeugen sie mit
`llama-quantize --pure` selbst. Was diese Entscheidung für die Aussagekraft bedeutet, steht in der
Reflexion ([`05-ergebnisse.md`](05-ergebnisse.md) §Reflexion).

„Flach“ ist unabhängig vom **Kodierschema**: Legacy-Quants (Q8_0, Q5_0, Q4_0) speichern Blöcke von
32 Gewichten mit einer Skala; K-Quants (Q6_K … Q2_K) Superblöcke von 256 Gewichten mit eigenen Skalen
und Minima je Teilblock. Gemischt sind erst die Rezepte mit `_M`/`_S` im Namen.

## 4. Matrix-Distanzen täuschen

Vergleicht man die Lens-*Matrizen*, unterscheiden sich zwei BF16-Lenses, die nur auf verschiedenen
Korpora gefittet wurden, stärker (relative Distanz 0,73) als die BF16- und die Q4_0-Lens (0,49).
Beim *Auslesen* echter Aktivierungen sind sich die beiden BF16-Lenses aber weitgehend einig
(Top-1 gleich an 78–95 % der Positionen, je nach Schicht). Die Matrizen sind in Richtungen, in denen
die Aktivierungen kaum variieren, praktisch beliebig. ⇒ Wir messen nur **Readouts**, und die zweite
BF16-Lens dient als **Rauschgrenze**: Ein Quant-Effekt muss größer sein als die ~20 % Abweichung,
die das Fitting allein in frühen Schichten erzeugt.

## 5. Prompt-Format

**Leerzeichen am Ende.** Viele Eval-Prompts enden auf ein Leerzeichen („… is the “). Spark macht
daraus ein einzelnes Leerzeichen-Token, das im normalen Text kaum vorkommt (dort hängt das
Leerzeichen am Folgewort). ⇒ Leerzeichen am Ende entfernen; gelesen wird am letzten echten Token.

**Spark vervollständigt rohen Text nicht.** Auf „Fact: The ocean on the coast of the country where
Carnival is most famously celebrated is the“ antwortet das Modell „ the the the …“ — es arbeitet
die Aufgabe gar nicht ab, also gibt es auch kein Zwischenkonzept zu lesen. Pilottest auf BF16
(pass@10 Lens / Logit Lens / Antwort korrekt):

| Set | roh | Chat-Format |
|---|---|---|
| multihop | 0,13 / 0,06 / 0,01 | **0,46** / 0,39 / **0,34** |
| multilingual | 0,22 / 0,19 / 0,04 | **0,41** / 0,42 / **0,49** |
| order-ops | 0,59 / 0,55 / 0,13 | 0,43 / 0,32 / **0,31** |
| association | 0,00 / 0,03 / — | 0,01 / 0,03 / — |
| typo | 0,31 / 0,76 / — | 0,44 / 0,94 / — |
| poetry | 0,00 / 0,02 / — | 0,01 / 0,02 / — |

⇒ **Chat-Format:** Der Prompt steht als begonnene Assistenten-Antwort hinter „Continue the text.“.

Zwei weitere Folgen des Pilottests:
- `association` und `poetry` liegen schon auf BF16 am Boden. Dort kann Quantisierung nichts
  verschlechtern ⇒ sie werden berichtet, gehen aber nicht in die Hypothesen ein.
- Auf `typo` schlägt der Logit Lens unsere Lens deutlich. Vermutlich, weil das korrigierte Wort dort
  fast das nächste Output-Token ist, das der Logit Lens direkt liest. Das ist eine Schwäche des
  Regressions-Surrogats, keine Folge der Quantisierung.

## 6. Paarweise Statistik

Im ersten Hauptlauf war das Konfidenzintervall der Perplexity unbrauchbar breit (60–230), weil die
Modelle ungepaart verglichen wurden und die Absätze sehr unterschiedlich schwer sind. Da alle Modelle
dieselben Positionen sehen, rechnen wir Differenzen pro Absatz bzw. Item ⇒ deutlich engere,
aussagekräftige Intervalle.

## 7. Form des kausalen Tests

Für den kausalen Test (Interventionen mit Lens-Richtungen auf dem Paper-Set `probe-swap`) haben wir
die Eingriffsform in einem Pilot **nur auf BF16** gewählt, an den ersten 30 der 90 Items und bevor ein
quantisiertes Modell lief. Anteil der Items, deren nächstes Token die Tausch-Antwort ist (ohne
Eingriff: 0,07; richtige Antwort ohne Eingriff: 0,40):

| Eingriff | Schichten | Tausch-Antwort | richtige Antwort | Tauschwort selbst |
|---|---|---|---|---|
| Koordinaten-Tausch (Paper) | L22–L34 (Band) | 0,10 | 0,33 | 0,00 |
| Koordinaten-Tausch | alle | 0,07 | 0,30 | 0,00 |
| Steering α = 0,05 | L22–L34 | 0,03 | 0,00 | 0,73 |
| Steering α = 0,1 | L14–L21 | 0,17 | 0,07 | 0,07 |
| **Steering α = 0,05** | **L18–L25** | **0,27** | **0,00** | **0,10** |

Beobachtungen: Der Koordinaten-Tausch des Papers wirkt mit unserer Regressions-Lens kaum. Steering
wirkt stark, aber je später, desto mehr als bloßer *Ausgabe*-Hebel: Im Band gibt das Modell das
Tauschwort selbst aus („Mexico“ statt „Spanish“), statt damit weiterzudenken. ⇒ **Hauptmaß: Steering
in L18–L25 mit α = 0,05** (α in Einheiten der typischen Residual-Norm der Schicht), dazu als
Sekundärmaß der Koordinaten-Tausch im Band und eine Ablation in L18–L25, jeweils mit Kontrolle.

Weil die Parameter an einem Teil der Items gewählt wurden, ist der absolute BF16-Wert leicht
optimistisch; der Vergleich zwischen den Quants ist davon nicht betroffen (gleiche Parameter für alle).
