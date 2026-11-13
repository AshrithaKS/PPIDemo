# Explainable, class-balanced PPI-site prediction

Residue-level protein–protein interaction (PPI) site prediction with protein-language-model
embeddings (ESM-2) + PSSM/DSSP, CTGAN balancing, XGBoost, and SHAP.

Data: DeepPPISP benchmark (Dset_72, Dset_186, PDBset_164 = 422 proteins) in `data_cache/` (read-only copy).

## Status
- [x] **Phase 1 – data foundation**
- [x] **Phase 2 – baselines, global features, CV, leakage test, DNN** (results below)
- [x] **Phase 3 – ESM-2 features** (35M, then 150M which is now used; wider-context features tested, no gain; results below)
- [x] **Phase 4 – CTGAN balancing** (results below)
- [x] **Phase 5 – tuning and final evaluation** (results below) · [x] **Phase 6 – SHAP** (results below) · [x] **Phase 7 – demo app** (final report and slides still to write)

## Reproduce: data preparation
```bash
pip install -r requirements.txt
python src/audit_dssp_alignment.py   # writes configs/dssp_shifts.csv + results/dssp_alignment*.{json,csv}
python src/data_loading.py           # writes data/processed/residues.parquet, proteins.csv, results/data_audit.json
python src/eda.py                    # writes results/figures/*.png
python -m pytest tests -q            # data, feature, metric and balancing invariants
```

## Baseline results (handcrafted features)
Features: 7-residue window of one-hot + PSSM + DSSP = 343 (no RSA, no global block). XGBoost with the paper's
settings (depth 9, lr 0.01, min_child_weight 5, 600 trees). Fixed 70-protein test split, 5 seeds
(different validation proteins), mean±std. "val-tuned" = MCC-optimal threshold chosen on held-out TRAIN proteins.

| arm | threshold | accuracy | precision | recall | f1 | mcc | pr_auc |
|---|---|---|---|---|---|---|---|
| all-negative | - | 0.802 | 0.000 | 0.000 | 0.000 | 0.000 | 0.198 |
| unbalanced | 0.5 | 0.802±0.000 | 0.807±0.127 | 0.002±0.001 | 0.004±0.002 | 0.030±0.009 | 0.284±0.004 |
| unbalanced | val-tuned | 0.657±0.064 | 0.278±0.022 | 0.432±0.153 | 0.320±0.054 | 0.128±0.014 | 0.284±0.004 |
| weighted | 0.5 | 0.705±0.005 | 0.292±0.004 | 0.342±0.008 | 0.315±0.003 | 0.130±0.004 | 0.287±0.001 |
| weighted | val-tuned | 0.682±0.050 | 0.285±0.018 | 0.385±0.103 | 0.319±0.024 | 0.128±0.004 | 0.287±0.001 |

Reproduce: `python -X utf8 src/baseline_xgboost.py --seeds 0 1 2 3 4` (about 2 min). Raw numbers: `results/baseline_results.json`.

### Baseline diagnostics (tree count, cross-validation, leakage test, DNN)
`python -X utf8 src/baseline_diagnostics.py all` (about 40 min on CPU; raw numbers in `results/baseline_{sweep,cv,leakage_test,dnn}.json`).
All rows use an MCC-optimal threshold chosen on held-out validation data. Mean±std.

| Setting | Features | Model | MCC | PR-AUC | Recall |
|---|---|---|---|---|---|
| **Protein-level 5-fold CV (422 proteins)** | local | XGB unbalanced | 0.113±0.014 | 0.229±0.018 | 0.325 |
| | local | XGB weighted | 0.104±0.026 | 0.225±0.020 | 0.258 |
| | local+global | XGB unbalanced | **0.204±0.013** | 0.303±0.013 | 0.451 |
| | local+global | XGB weighted | 0.193±0.012 | 0.289±0.011 | 0.443 |
| **Random residue-level 80/20 (paper-style)** | local | XGB unbalanced | 0.134±0.007 | 0.246 | 0.371 |
| | local+global | XGB unbalanced | **0.345±0.005** | 0.440 | 0.564 |
| | local+global | XGB weighted | 0.331±0.007 | 0.430 | 0.476 |
| **Fixed test split (70 proteins)** | local | DNN (1024-256) | 0.106±0.024 | 0.268±0.009 | 0.341 |
| | local+global | DNN | 0.186±0.012 | 0.329±0.013 | 0.468 |

Findings
1. **Not undertrained.** Early stopping (lr 0.01, up to 3000 trees) leaves PR-AUC at about 0.29 (local); best tree count
   ranges 205-2960 across seeds, i.e. a flat optimum. More trees do not fix the low MCC.
2. **Global context helps under a fair protein-level split:** MCC 0.11 -> 0.20 (CV, low fold-to-fold std).
3. **Leakage inflates the global-feature gain.** With a random residue split the same model scores MCC 0.345 vs 0.204
   protein-level (about +0.14). Global features are constant within a protein, so the model can recognise the protein.
   With local features the gap is small (0.134 vs 0.113). This is consistent with, but does not prove, the paper's
   0.305 being partly inflated: our global block is a compact stand-in (51 summary values), not the paper's 500-residue block.
4. **Learner trade-off matches the paper:** the DNN has higher recall at 0.5 (0.50-0.54) but not higher MCC than XGBoost.
5. Simple imbalance fixes (threshold tuning, class weights) give about the same MCC as each other; CTGAN must beat these.

Caveats: no RSA in the data; protein-level test split has 19.8% positives vs 14.7% train; CV thresholds come from a 15%
inner hold-out of training proteins.

## ESM-2 feature results (35M embeddings)
Embeddings: `python src/embed_esm.py --model 35M` (53 s on CPU; 88,040 x 480, float16, `data/processed/esm2_35M.npy`; proteins > 1000 residues
use overlapping chunks, overlaps averaged). Comparison: `python -X utf8 src/feature_set_comparison.py 35M` (about 1 h on CPU).
Protocol for every row: protein-level 5-fold CV over all 422 proteins, XGBoost (depth 9, lr 0.03, colsample 0.5, early stopping on a 15%
protein hold-out), MCC-optimal threshold chosen on that hold-out. No balancing yet. Mean±std over folds.

| Features | # | Accuracy | Precision | Recall | MCC | PR-AUC |
|---|---|---|---|---|---|---|
| H handcrafted (window + global) | 394 | 0.775 | 0.319 | 0.367 | 0.205±0.018 | 0.314±0.018 |
| E ESM-2 only | 480 | 0.758 | 0.300 | 0.411 | 0.206±0.027 | 0.311±0.024 |
| E+G ESM-2 + global | 531 | 0.780 | 0.337 | 0.413 | 0.240±0.018 | 0.340±0.019 |
| **F fused (ESM-2 + window + global)** | 874 | **0.789** | **0.350** | 0.406 | **0.250±0.023** | **0.354±0.023** |

Paired per-fold gain over H: E +0.000 MCC (3/5 folds better); E+G +0.035 (5/5); F +0.045 MCC and +0.039 PR-AUC (5/5 folds).
XGBoost gain share in F: ESM-2 74%, global 19%, window PSSM 7%, DSSP 0.5%, one-hot 0.2% (a rough preview; the SHAP analysis comes later).

Findings
1. ESM-2 alone only **matches** the handcrafted baseline (MCC 0.206 vs 0.205); it does not beat it. 
2. Fusing ESM-2 with handcrafted + global features gives a **modest but consistent** gain (+0.045 MCC, better in all 5 folds, but fold 4 gain is only +0.002).
3. Accuracy stays at 0.76-0.79, below the all-negative baseline (~0.85 pooled); judge models by MCC / PR-AUC.
4. Fused MCC 0.25 is still below the paper's reported 0.305 (random split), consistent with the leakage test above.
Caveats: this comparison used the 35M model (150M was tested later and is now used, see below; 650M untested); no RSA; no class balancing in this comparison.

### ESM-2 size and wider context
- **35M vs 150M** (`src/esm_size_comparison.py`, final-evaluation protocol, 3 seeds, 70 test proteins; `results/esm_size_comparison.json`):
  fused MCC 0.250 -> 0.274, PR-AUC 0.395 -> 0.423, accuracy 0.726 -> 0.759; sequence-only MCC 0.244 -> 0.269. Paired protein bootstrap: the PR-AUC gain
  for the fused model excludes zero (CI +0.003 to +0.054), the MCC gain does not (CI -0.008 to +0.048). Embedding with 150M takes about 3.5 min on CPU.
  650M was not run (2.5 GB, expected small further gain, heavy for a laptop CPU).
- **Wider context and patch smoothing** (`src/context_experiment.py`, summary `results/context_experiment_summary.md`): adding the mean 150M embedding of the
  residues within ±7 lowered MCC in both seeds tested (0.256 vs 0.279, 0.267 vs 0.271); smoothing predicted probabilities over neighbours changed MCC by
  +0.007 (within noise) and lowered accuracy. Neither is used. Stopped after 5 of 9 models; the ±3/±15 variant was not run.

## Class-balancing results (including CTGAN)
`python -X utf8 src/balancing_comparison.py --epochs 200 --tag 35M` (about 1 h on CPU; CTGAN fit ~9 min/fold), then `python -X utf8 src/balancing_summary.py 35M`.
Raw: `results/balancing_comparison_35M.json`, tables: `results/balancing_summary_35M.md`.
Protocol: protein-level 5-fold CV; inside each fold PCA, CTGAN and SMOTE-NC see only that fold's training proteins; a 15% protein hold-out gives
early stopping + the MCC-optimal threshold; test folds stay at the natural class ratio. All arms use the SAME 82 inputs: amino acid + DSSP state
(categorical), 20 PSSM values, 32 PCA components of ESM-2 (47% variance). Window and global features are excluded (synthetic residues have neither).
CTGAN: 200 epochs, default hyper-parameters, 2 discrete columns.

| arm | accuracy | precision | recall | MCC | PR-AUC |
|---|---|---|---|---|---|
| none | 0.663 | 0.237 | 0.494 | 0.150±0.016 | 0.252±0.017 |
| class weight | 0.678 | 0.236 | 0.465 | 0.147±0.019 | 0.244±0.016 |
| random oversampling (1.0) | 0.626 | 0.223 | 0.559 | 0.148±0.015 | 0.243±0.018 |
| SMOTE-NC (0.5) | 0.640 | 0.224 | 0.516 | 0.138±0.014 | 0.231±0.017 |
| SMOTE-NC (1.0) | 0.574 | 0.205 | 0.603 | 0.128±0.012 | 0.224±0.012 |
| CTGAN (0.5) | 0.626 | 0.223 | 0.564 | 0.152±0.012 | 0.246±0.018 |
| CTGAN (1.0) | 0.626 | 0.222 | 0.560 | 0.149±0.016 | 0.245±0.013 |

Paired per-fold differences: CTGAN(0.5) − none = +0.002 MCC (better in 3/5 folds), −0.005 PR-AUC (1/5); CTGAN(1.0) − none = −0.001 MCC (2/5), −0.007 PR-AUC (0/5);
CTGAN − SMOTE-NC = +0.014 to +0.021 MCC and +0.015 to +0.021 PR-AUC (**5/5 folds**).
Synthetic-data fidelity (mean of 5 folds): CTGAN KS 0.141, corr-diff 0.080, real-vs-synthetic detector AUC 1.00, nearest-neighbour ratio 1.77;
SMOTE-NC KS 0.053, corr-diff 0.021, detector AUC 0.976, ratio 0.65 (closer to, partly copying, real rows).

Findings
1. **CTGAN gives no measurable benefit over not balancing** once the threshold is tuned on validation data (MCC 0.152 vs 0.150; PR-AUC slightly lower).
   The 2-fold interim lead (+0.010) disappeared with all 5 folds.
2. **CTGAN is consistently better than SMOTE-NC** (5/5 folds), but SMOTE-NC is the arm that *hurts* (below no-balancing), so this is not a strong claim for CTGAN.
3. Balancing mainly moves the precision/recall trade-off (recall 0.49 -> 0.56-0.60, precision 0.24 -> 0.21-0.22); it does not improve ranking (PR-AUC).
4. Un-tuned, balanced models are miscalibrated at threshold 0.5 (MCC at 0.5: CTGAN 0.06, SMOTE-NC 0.04-0.08, none 0.05; class weight / oversampling 0.15), so always tune the threshold on real validation data.
5. CTGAN rows are trivially separable from real minority rows (detector AUC 1.00) and match marginals worse than SMOTE-NC.
Caveats: reduced feature set (MCC 0.15 here vs 0.25 for the fused model in the ESM-2 feature comparison); CTGAN not tuned (200 epochs, defaults); CTGAN on the fused/global
features, longer training, and conditional generation were NOT tested; only the 35M ESM-2 model.

### Balancing extension (PRELIMINARY, 1 fold only): CTGAN on compact fused features
`python -X utf8 src/balancing_comparison.py --fused --epochs 200 --tag fused_35M` adds global-block PCs (8), chain length, relative position and
neighbour-PSSM PCs (8) to the 82 per-residue inputs (**100 inputs**). The run was **stopped deliberately after fold 0** (CTGAN fit took 49 min on
this CPU; folds 1-4 not run). Raw: `results/balancing_comparison_fused_35M.json`, table: `results/balancing_summary_fused_35M.md`.
Fold 0 only, MCC at validation-tuned threshold: none 0.222, class weight 0.219, random oversampling 0.212, SMOTE-NC 0.214 / 0.174,
**CTGAN 0.164 (ratio 0.5) / 0.191 (ratio 1.0)**; PR-AUC none 0.300 vs CTGAN 0.253 / 0.260. Real-vs-synthetic detector AUC for CTGAN rows 1.00.
Read-out: no sign that CTGAN helps; on this fold it is 0.03-0.06 MCC below no balancing. One fold has no error bars (earlier fold-to-fold spread
was about 0.015-0.02 MCC), so this supports, but does not prove, the 5-fold conclusion above. Also note compact fused features score higher
(fold 0 MCC 0.222) than the reduced 82-input set on the same fold (0.123), confirming that global/neighbour context matters.

## Final evaluation on held-out test proteins
`PPI_N_JOBS=6 python -X utf8 src/final_evaluation.py --seeds 0 1 2` (about 15 min at 6 threads). Raw: `results/final_evaluation.json`,
per-residue test probabilities: `results/test_predictions.parquet`, saved models: `models/final_model_F.json`, `models/demo_model_S.json`
(+ `_meta.json` with feature names and the validation-chosen threshold).
Train on the 352 DeepPPISP training proteins, test on the 70 held-out proteins (19.8% positive residues). Each of 3 seeds changes the 15%
validation proteins used for early stopping and for the MCC-optimal threshold; test proteins never influence training, stopping, thresholds or tuning.
Hyper-parameter search (Optuna, 3 trials, lr 0.1): the starting settings (depth 9, min_child_weight 5, colsample 0.5) were best (PR-AUC 0.331 vs 0.309 / 0.318),
so **tuning found no improvement over the defaults** and the final model uses them (final models trained at lr 0.05 with early stopping). The search was stopped
early because of laptop overheating, so this is a small search, not evidence the defaults are optimal.

**Current models use ESM-2 150M** (`ESM_SIZE` in `src/model_utils.py`; see "ESM-2 size and wider context" below). Per-seed results and the 3-seed ensemble
(probabilities of the three seed models averaged, threshold = mean of the three validation thresholds):

| model | # features | accuracy | precision | recall | MCC (mean±std over seeds) | PR-AUC | ensemble accuracy / MCC / PR-AUC | MCC 95% CI | PR-AUC 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| Handcrafted baseline | 394 | 0.685 | 0.330 | 0.503 | 0.208±0.010 | 0.349±0.002 | 0.708 / 0.223 / 0.364 | 0.18–0.27 | 0.32–0.42 |
| **Fused model (final), ESM-2 150M** | 1034 | **0.759** | 0.409 | 0.446 | **0.274±0.006** | **0.423±0.005** | **0.767 / 0.290 / 0.440** | 0.24–0.34 | 0.38–0.50 |
| Sequence-only (demo), ESM-2 150M | 662 | 0.758 | 0.404 | 0.440 | 0.269±0.009 | 0.410±0.003 | 0.765 / 0.286 / 0.423 | 0.23–0.34 | 0.37–0.48 |
| *previous fused model, ESM-2 35M* | 874 | 0.726 | 0.369 | 0.487 | 0.250±0.006 | 0.395±0.002 | – / 0.27 / 0.41 | 0.22–0.32 | 0.36–0.47 |

95% CIs come from resampling whole test proteins (1000 bootstrap draws) on the 3-seed ensemble. Paired differences (same resampled proteins):
- Fused − baseline: ΔMCC +0.067 (95% CI +0.034 to +0.101), ΔPR-AUC +0.075 (CI +0.041 to +0.110); the interval excludes zero, so the gain is supported on this split.
- Fused − sequence-only: ΔMCC +0.004 (CI -0.015 to +0.024), ΔPR-AUC +0.016 (CI +0.002 to +0.032); MCC cannot be told apart, PR-AUC is slightly higher for the fused model.
  The sequence-only model, which needs no PSSM/DSSP and can score a new FASTA sequence, is almost as good.
- The saved model files are the seed-0 models (fused: MCC 0.279, PR-AUC 0.423, accuracy 0.733; demo: MCC 0.279, PR-AUC 0.413), not the 3-seed ensemble.
The 35M-era models and results are kept in `models/archive_35M/` and `results/archive_35M/`.

Caveats: one fixed 70-protein split (wide absolute intervals: MCC about 0.24–0.34); PR-AUC depends on prevalence, so these test-split numbers are not comparable with cross-validation numbers above (test split has more positives);
the MCC-optimal threshold is unstable across seeds (0.10–0.21 for the fused model; recall 0.34–0.53 at similar MCC), so report threshold-free PR-AUC alongside MCC; no RSA;
published numbers use different splits and are not directly comparable (next section).

### Comparison with the reference study (Kulkarni et al., 2025)
| | split | accuracy | MCC |
|---|---|---|---|
| Reference: XGBoost, CTGAN-balanced, local + global (their best, as reported) | random residue-level 80/20 | 0.831 | 0.305 |
| Reference: deep network, CTGAN-balanced, local + global (as reported) | random residue-level 80/20 | 0.641 | 0.298 |
| Ours: XGBoost, local + global handcrafted, no balancing (leakage test above) | random residue-level 80/20 (paper-style) | 0.794 | **0.345** |
| Ours: same handcrafted baseline | protein-level (70 unseen proteins) | 0.685 | 0.208 |
| **Ours: final fused model, ESM-2 150M, 3-seed ensemble** | **protein-level (70 unseen proteins)** | **0.767** | **0.290** |

Under the reference study's own random residue split, our handcrafted XGBoost already exceeds their best reported MCC without CTGAN; the same model drops to
MCC 0.21 when whole proteins are held out, so random residue splits inflate scores. CTGAN gave no measurable gain under protein-level evaluation (class-balancing
section). Accuracy is not a useful headline here: predicting "non-interface" for every residue scores about 0.80–0.85.

## SHAP explainability and within-protein ranking
`python -X utf8 src/explain_shap.py` (about 1 min) and `python -X utf8 src/per_protein_ranking.py`. Outputs: `results/shap_family_importance.json`,
`results/per_protein_ranking.json`, figures `results/figures/shap_*.png`. SHAP (exact TreeExplainer) explains the final fused model on 3,000 random
held-out test residues; additivity check max error 4.3e-06 (explanations add up to the model's scores). Numbers below are for the ESM-2 150M fused model
(the 35M-era values are in `results/archive_35M/`; the picture is the same).
SHAP describes what the MODEL relies on, not why proteins bind: it does not show biological causation.

Share of total mean |SHAP| by feature family:

| family | # features | share |
|---|---|---|
| ESM-2 embedding (150M) | 640 | 79.6% |
| window: PSSM | 140 | 6.0% |
| global: amino-acid composition | 20 | 4.6% |
| global: length / relative position | 2 | 4.2% |
| global: DSSP fractions | 9 | 2.6% |
| global: mean PSSM | 20 | 2.5% |
| window: DSSP | 63 | 0.5% |
| window: residue identity (one-hot) | 140 | 0.1% |

Findings
1. **ESM-2 carries most of the signal** (80% with 150M, 73% with 35M), spread over many dimensions (the 10 most important dimensions hold only 13% of ESM importance), so individual ESM dimensions are not interpretable.
2. **Evolutionary profile (PSSM) is the main interpretable family** (6%); PSSM column 12 (lysine, BLAST order) at the residue's neighbours is the strongest window feature, and higher values lower the predicted interface probability. DSSP and residue-identity one-hots contribute almost nothing (0.5% and 0.1%).
3. **Local context decays with distance** (share of window |SHAP| by offset: -3: 11%, -2: 15%, -1: 15%, 0: 23%, 1: 18%, 2: 9%, 3: 10%).
4. **Protein length is the single strongest interpretable feature** (mean |SHAP| 0.26, about 3x the next): short proteins get a large push up, long proteins a push down. Across the 70 test proteins, length and interface fraction have Spearman -0.71, so this is a real size prior, but it ranks PROTEINS, not residues.
5. **How much is the size prior?** Scoring every residue with only its protein's mean predicted probability gives pooled PR-AUC 0.304 (chance = 0.198); the full model gives 0.440. So roughly 44% of the pooled gain over chance is reproducible from the per-protein average alone (53% with the 35M model). Pooled metrics therefore overstate within-protein skill.
6. **Within-protein ranking is still genuine:** treating each of the 70 test proteins as its own ranking problem, the fused model's mean average precision is 0.472 versus 0.243 for a random ranking, above random in 66/70 proteins, and above the handcrafted baseline (0.383) in 50/70 proteins (mean gain +0.089).
7. Example per-residue figures (best / median / weakest test protein): `results/figures/shap_protein_3e1z_B.png`, `shap_protein_3AYHB.png`, `shap_protein_3VU9A.png`. Performance varies a lot between proteins.
Caveats: SHAP computed on a 3,000-residue sample of one split; relative chain position also shifts scores (earlier residues score higher), which may reflect dataset-specific artefacts rather than biology (not tested).

## Demo app: BindSpot (Streamlit)
```bash
python src/demo_assets.py                 # once: precompute held-out probabilities, threshold curve, family importance
python -m streamlit run app/streamlit_app.py
```
Opens at http://localhost:8501. `.streamlit/config.toml` binds the server to `localhost` only (not reachable from other devices) and sets light and dark themes.
Held-out proteins load in about 10 s; the first pasted sequence also loads PyTorch and ESM-2 (about 30-45 s once), later ones take seconds.
A guided page in three steps (a clickable stepper at the top, with Back / Next buttons):
- **1 Choose a protein:** one of the 70 held-out test proteins (the experimentally known answer is shown next to the prediction) or your own sequence
  (plain or FASTA, 20 to 1500 residues, 20 standard amino acids, up to 5 records; a "Use example (ubiquitin)" button). Invalid input is reported next to the box.
- **2 See the result (answer first):** a headline ("34 residues of 3e1z_B are likely to touch another protein", hotspots, and for test proteins how many real sites
  were found and what share of flags is correct), then the protein as coloured residue tiles in groups of ten (darker = more likely, outlined = flagged,
  orange underline = real interface). A threshold slider with a "What does this setting do?" pop-over (precision and recall against threshold).
  Collapsed extras: an interactive probability chart (hover, drag to summarise a region) and a table with CSV download.
- **3 Why?:** pick a hotspot or any residue; its probability against an average residue, a plain-language reason, an exact explanation chart
  (XGBoost tree SHAP grouped into ESM-2 / composition / length / position, adds up to the model's log-odds) and the residue's neighbourhood. A collapsed
  "How the model works and how good it is" section holds the pipeline, the held-out metrics and the limits.
- **Design:** ui-ux-pro-max "Biotech / Life Sciences" palette (sky blue + life-science green) and the Poppins + Open Sans pairing. Sky blue means what the
  model says, orange what experiments say; the three data colours pass every dataviz colour-vision check in light and dark (`validate_palette.js`), and every
  text colour is at least 5:1 against its background. Colours resolve with CSS `light-dark()`, so a theme switch updates the page instantly, and a tiny hidden hook
  reruns the app once so the charts redraw. Keyboard focus is visible, buttons are at least 44 px tall, and motion is off for reduced-motion users.
- Model: `models/demo_model_S.json` = sequence-only features (ESM-2 150M embedding + chain composition + log length + relative position, 662 features). It needs no PSSM / DSSP, so it can score any new sequence.
  Held-out test: MCC 0.28, PR-AUC 0.41, precision 0.38, recall 0.53 at the default threshold 0.14 (chosen on validation proteins).
  The app reads the language-model size from `models/demo_model_S_meta.json`; the first pasted sequence loads ESM-2 150M (about 40-60 s once).
- Tests: `tests/test_inference.py` (features identical to training, fresh ESM embedding equals the cached one), `tests/test_app.py` (clicks through all three steps headlessly, checks that choices survive moving between steps, and pastes ubiquitin end to end).
  A bug caught on the way: contributions used all stored trees while predictions stop at the early-stopping iteration (error 0.6 log-odds); fixed and checked (additivity error 6e-6).
- Limits stated in the app: research prototype; partner-independent; short proteins score higher on average (size prior, see the SHAP section), so many residues can be flagged in a short chain (ubiquitin: 27 of 76);
  the threshold is a starting point, not a truth. Not for clinical or experimental decisions.

## Verified facts about the data
| Item | Value |
|---|---|
| Proteins / residues / positives | 422 / 88,040 / 13,536 |
| Positive rate | 15.4% pooled (Dset_72 10.6%, Dset_186 15.2%, PDBset_164 18.1%) |
| Split (DeepPPISP lists, protein-level) | 352 train / 70 test proteins, 0 overlap; positives 14.7% train vs **19.8% test** |
| Sequence encoding | ints → `ACDEFGHIKLMNPQRSTVWY` (DeepPPISP `utils/config.py`) |
| PSSM column order | BLAST order `ARNDCQEGHILKMFPSTWYV` (native residue is row-argmax for 64% of residues) |
| DSSP | 9-column one-hot (one state never occurs, 649 residues all-zero). **No RSA in the data.** |
| Protein lengths | min 39, median 165, max 2000; 14 proteins > 500, only 1 > 1022 (ESM-2 limit) |

## DSSP alignment
46/422 proteins have a DSSP array whose length differs from the sequence (32 longer, 14 shorter,
up to 15). DSSP has no residue identity, so `src/audit_dssp_alignment.py` fits P(DSSP | aa, PSSM)
on the 376 length-matched proteins and scores every offset. A shift is applied only when it beats
offset 0 by > 0.02 mean log-likelihood (noise level on matched proteins: mean 0.005, p90 0.015):
**17 proteins shifted**, others keep DeepPPISP's rule (first N rows, zero-pad). Padded residues are
flagged in `dssp_missing`. This is a heuristic; treat DSSP for those 46 proteins as slightly noisy.

## Layout
```
data_cache/            raw DeepPPISP pickles (do not modify)
data/processed/        residues.parquet (1 row / residue), proteins.csv
configs/dssp_shifts.csv  per-protein DSSP offset
src/                   data_loading.py, audit_dssp_alignment.py, eda.py, features.py, evaluate.py, baseline_xgboost.py,
                       baseline_diagnostics.py, embed_esm.py, feature_set_comparison.py,
                       balancing.py, balancing_comparison.py, balancing_summary.py,
                       model_utils.py, tune_hyperparameters.py, final_evaluation.py, explain_shap.py, per_protein_ranking.py, inference.py, demo_assets.py
app/                   streamlit_app.py (demo)
tests/                 test_data_loading.py, test_features_metrics.py, test_balancing.py, test_final_models.py, test_inference.py, test_app.py
results/               audit JSON/CSV, figures/
```
