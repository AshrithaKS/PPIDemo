# Wider-context features and patch smoothing (stopped early, 5 of 9 models)

Script: `src/context_experiment.py` (raw log: `context_experiment.log`). Same protocol as `final_evaluation.py`:
protein-level split, 15% validation proteins for early stopping, threshold and smoothing parameters chosen on validation only,
test = the 70 held-out proteins. ESM-2 150M embeddings, sequence-only feature set.

| Variant | Runs | Test MCC | Test PR-AUC | Accuracy |
|---|---|---|---|---|
| base: ESM-2 150M + composition + length + position (662 features) | seeds 0,1,2 | **0.269 ± 0.009** | **0.410** | 75.8% |
| base + patch smoothing (p' = a·p + (1-a)·mean of ±h neighbours, a/h picked on validation) | seeds 0,1,2 | 0.276 ± 0.014 | 0.408 | 74.8% |
| + mean ESM-2 embedding of residues within ±7 (1302 features) | seeds 0,1 | 0.262 (0.256, 0.267) | 0.406 | 73.0% |
| + ±7 with smoothing | seeds 0,1 | 0.260 | 0.400 | 68.8% |
| + means within ±3 and ±15 (1942 features) | not run | – | – | – |

Paired by seed (same train / validation / test proteins), the ±7 context model scored below the base model in both runs
(seed 0: 0.256 vs 0.279; seed 1: 0.267 vs 0.271). Smoothing changed MCC by +0.007 on average, within the seed-to-seed spread
(±0.014), lowered PR-AUC slightly and lowered accuracy by 1 point.

**Conclusion:** neither wider-context embedding features nor patch smoothing improved on the plain ESM-2 150M model, so the
final and demo models use plain ESM-2 150M. The experiment was stopped after 5 of 9 models: the remaining ±3/±15 variant is
larger still and the trend gave no reason to expect a gain. ESM-2 embeddings already encode sequence context through
self-attention, which is a plausible reason why averaging neighbouring embeddings adds little.
