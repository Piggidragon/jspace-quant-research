# 01 · Hintergrund

## Global Workspace und Jacobian Lens

Gurnee, Sofroniew et al. (2026), *Verbalizable Representations Form a Global Workspace in Language
Models* ([transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace/index.html)),
zeigen, dass Sprachmodelle in mittleren Schichten Konzepte bereithalten, die sich in Worte übersetzen
lassen — auch solche, die im Text nicht vorkommen (z. B. „Brazil“ als Zwischenschritt einer
Zwei-Schritt-Frage). Ausgelesen wird das mit der **Jacobian Lens**:

```
lens_l(h) = Unembed( J_l · h ),   J_l = E[ ∂h_final / ∂h_l ]
```

`J_l` bildet einen Residual-Stream-Vektor aus Schicht `l` in den Raum der letzten Schicht ab; das
Unembedding macht daraus eine Rangliste von Tokens. Mit `J_l = I` ergibt sich der klassische
**Logit Lens**. Die Original-Lenses werden per Backpropagation über 1000 Sequenzen à 128 Tokens
gefittet; ~100 reichen laut Paper schon für brauchbare Qualität. Selbst benannte Grenzen: nur
Ein-Token-Konzepte, frühe Schichten kaum interpretierbar, die Lens sei „an imperfect tool“.

Referenzcode und Eval-Sets: [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
(Apache 2.0). Fertige Lenses gibt es nur für andere Modelle
([neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens)), nicht für Spark.

## Werkzeug: jlens-gguf

[igorbarshteyn/jlens-gguf](https://github.com/igorbarshteyn/jlens-gguf) wendet die Lens auf jede
GGUF-Datei an, die llama.cpp lädt: Ein C++-Server (`jlens-server`) liefert den Residual Stream jeder
Schicht, Python/numpy rechnet Lens und Readout. Weil llama.cpp keine Gradienten berechnet, fittet
jlens-gguf statt der echten Jacobi-Matrix eine **Ridge-Regression** `h_final ≈ A_l·h_l + b` (nur
Forward-Pässe). Das ist ein korrelatives Surrogat in der Tradition der **Tuned Lens** (Belrose et al.
2023, [arXiv:2303.08112](https://arxiv.org/abs/2303.08112)).

## Modell: Spark-X2.6-4B

[XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B), GGUFs unter
[XHToken/Spark-X2.6-4B-GGUF](https://huggingface.co/XHToken/Spark-X2.6-4B-GGUF).

| | |
|---|---|
| Architektur | `spark2_5`, 36 Schichten, Residual-Breite 2560 |
| Attention | hybrid: Sliding-Window (512) mit Full-Attention in jeder vierten Schicht |
| Vokabular | 131 072, Embedding und Unembedding geteilt |
| natives dtype | bfloat16 |

**Warum Spark:** klein genug für eine Grafikkarte, aber praktisch nutzbar — statt der älteren
Testmodelle (GPT-2, Pythia), auf denen viele Interpretierbarkeits-Arbeiten laufen. Modern trainiert
und keine klassische dichte Full-Attention-Architektur; damit prüfen wir auch, ob die Methode über den
Standardfall hinaus trägt.

## Literatur: Quantisierung und interne Repräsentationen

Eine Arbeit zu genau unserer Frage (Lens-Readouts unter GGUF-Quantisierung, mit neu gefitteten Lenses
je Stufe) haben wir nicht gefunden. Nächste Nachbarn und Erwartungen:

| Arbeit | Befund | Erwartung für uns |
|---|---|---|
| Hu & Ramachandran 2026, [arXiv:2609.01587](https://arxiv.org/abs/2609.01587) | 8 bit ist bei RTN, GPTQ und AWQ nahezu verlustfrei; Schaden verteilt sich über viele Schichten | Q8_0: kein messbarer Effekt |
| Duan 2026, [arXiv:2606.03002](https://arxiv.org/abs/2606.03002) | SAE-Features zerfallen unter Quantisierung gestuft und meist „verwischt“; Perplexity übersieht den Schaden; stark aktive Features überleben eher | Perplexity nur als Kontrolle; starke Konzepte halten länger |
| Wu, Dhiman & Koshiyama 2026, [arXiv:2608.06564](https://arxiv.org/abs/2608.06564) | Low-bit schrumpft Entscheidungs-Margins etwa proportional | systematische Verschiebung ohne Strukturbruch |
| Wang et al. 2025, [arXiv:2505.13963](https://arxiv.org/abs/2505.13963) (TrustNLP @ ACL 2026) | kleinere Modelle verlieren unter Quantisierung mehr Faktenwissen | 4B ist ein eher empfindliches Regime |
| menelly/vspace-jlens (Community, nicht begutachtet) | NF4 erhält die Workspace-Geometrie weitgehend (Kosinus 0,968); auf zwei sehr kleinen Modellen (0,5B, 1,7B) scheitert die J-Lens an der eigenen Positivkontrolle | 4 bit: Geometrie weitgehend erhalten; Positivkontrolle nötig |
| willkn/jlens_re (Analyse auf GPT-2-medium) | Die rohe J-Lens ist bei Next-Token-Treue in jeder Schicht schlechter als der Logit Lens | Logit Lens ist ernstzunehmende Vergleichsgröße, nicht nur Baseline |

Einschränkung: Diese Befunde stammen aus anderen Verfahren (RTN, GPTQ, AWQ, NF4); GGUF-Quants sind
strukturell anders. Grundlagen der Quantisierung: Frantar et al. 2022 (GPTQ,
[arXiv:2210.17323](https://arxiv.org/abs/2210.17323)), Lin et al. 2023 (AWQ,
[arXiv:2306.00978](https://arxiv.org/abs/2306.00978)). Repräsentationsvergleich: Kornblith et al.
2019 (CKA, [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)).
