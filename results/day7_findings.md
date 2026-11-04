**Day 7 Manual inspection**: 
Manual inspection of 20 technical and 20 general-domain translations found accurate technical instructions, preserved placeholders, and generally appropriate terminology, alongside isolated terminology and formatting errors. General-domain outputs showed substantial meaning errors, omissions, incorrect word senses, and named-entity corruption. These observations are consistent with the measured technical-domain gains and general-domain degradation. However, the sample contains only fine-tuned predictions, so paired baseline inspection is needed to attribute individual errors or improvements to fine-tuning. Residual source metadata and source–reference inconsistencies also limit evaluation reliability.

| Domain | Sentences | BLEU | chrF | Sentences/sec |
|---|---:|---:|---:|---:|
| technical | 12623 | 40.5126 | 61.8439 | 9.060 |
| general | 1997 | 30.6843 | 57.3025 | 7.329 |

**Day 7 evaluation report** 
It shows a clear tradeoff:
- Technical translation improved: BLEU rose by 5.41 points, and chrF by 3.94 points.
- General translation degraded: BLEU fell by 9.72 points, and chrF by 8.18 points.
Technical-only fine-tuning improved performance in its training domain but reduced general-domain performance. This is consistent with over-specialization; the scores alone don’t establish its cause.

**Comparing day7_manual_inspection.csv and day5_prediction.csv**:
Paired manual inspection of 20 technical and 20 general-domain sentences identified technical terminology improvements and several general-domain regressions after fine-tuning. The fine-tuned model restored “normal” in a statistical definition and removed appended localization metadata in several technical examples. However, many technical translations remained equivalent, and one output introduced email-address corruption.

General-domain regressions included changing shark species into individual sharks, omitting supermarkets as the responsible actor, losing political details, and mistranslating domestic violence. Some general-domain examples improved, and several errors were shared by both models.

These observations support the automatic evaluation’s finding of technical-domain gains accompanied by reduced general-domain quality. This small, unblinded review does not establish population-wide error rates or statistical significance. Contaminated sources and imperfect references remain evaluation limitations.