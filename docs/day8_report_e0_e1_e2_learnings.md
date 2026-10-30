### Day 8 — Controlled Fine-Tuning: Results and Conclusion

Three German-to-English MarianMT configurations were compared using the same held-out technical and general-domain test datasets and consistent SacreBLEU BLEU and chrF evaluation settings.

**E0 — Pretrained baseline:** The original MarianMT model achieved 35.1068 technical BLEU and 40.4083 general BLEU.

**E1 — Technical-domain fine-tuning:** Fine-tuning on technical translation pairs increased technical BLEU to 40.5126 (+5.4058 points) but reduced general BLEU to 30.6843 (−9.7240 points). This demonstrated improved domain specialization accompanied by substantial general-domain regression.

**E2 — Mixed-domain fine-tuning:** Combining technical and general-domain training examples produced 39.8935 technical BLEU and 35.6534 general BLEU. Compared with E1, general BLEU improved by 4.9691 points, while technical BLEU decreased by only 0.6191 points.

The chrF results showed a consistent pattern, supporting the conclusion that mixed-domain training improved the balance between domain-specific and general translation quality.

**Conclusion:** E2 provided the strongest balance among the fine-tuned models, retaining approximately 88.5% of E1's technical BLEU improvement over the pretrained baseline while recovering approximately 51.1% of the general BLEU regression. The results support mixed-domain fine-tuning as a promising approach for reducing domain over-specialization.

**Limitations:** The experiments were evaluated using fixed held-out datasets and a single observed run per configuration. Further evaluation should include manual translation error analysis, repeated training seeds, dataset provenance verification and statistical significance testing before making stronger generalization claims.