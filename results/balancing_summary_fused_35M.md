Folds finished: 1 of 5

### Test metrics at the validation-tuned threshold (mean±std over folds)

| arm | accuracy | precision | recall | f1 | mcc | pr_auc | MCC at 0.5 |
|---|---|---|---|---|---|---|---|
| none | 0.770±0.000 | 0.308±0.000 | 0.416±0.000 | 0.354±0.000 | 0.222±0.000 | 0.300±0.000 | 0.079 |
| weight | 0.803±0.000 | 0.342±0.000 | 0.327±0.000 | 0.335±0.000 | 0.219±0.000 | 0.304±0.000 | 0.224 |
| ros1.0 | 0.799±0.000 | 0.333±0.000 | 0.328±0.000 | 0.330±0.000 | 0.212±0.000 | 0.300±0.000 | 0.225 |
| smotenc0.5 | 0.726±0.000 | 0.276±0.000 | 0.500±0.000 | 0.356±0.000 | 0.214±0.000 | 0.294±0.000 | 0.138 |
| smotenc1.0 | 0.792±0.000 | 0.303±0.000 | 0.289±0.000 | 0.296±0.000 | 0.174±0.000 | 0.269±0.000 | 0.179 |
| ctgan0.5 | 0.737±0.000 | 0.259±0.000 | 0.395±0.000 | 0.313±0.000 | 0.164±0.000 | 0.253±0.000 | 0.094 |
| ctgan1.0 | 0.678±0.000 | 0.247±0.000 | 0.551±0.000 | 0.341±0.000 | 0.191±0.000 | 0.260±0.000 | 0.104 |

### Paired per-fold differences (first arm minus second)

| comparison | ΔMCC | folds better | ΔPR-AUC | folds better |
|---|---|---|---|---|
| ctgan0.5 − none | -0.058 | 0/1 | -0.047 | 0/1 |
| ctgan1.0 − none | -0.031 | 0/1 | -0.040 | 0/1 |
| ctgan1.0 − weight | -0.028 | 0/1 | -0.043 | 0/1 |
| ctgan1.0 − ros1.0 | -0.021 | 0/1 | -0.039 | 0/1 |
| ctgan1.0 − smotenc1.0 | +0.017 | 1/1 | -0.009 | 0/1 |
| ctgan0.5 − smotenc0.5 | -0.050 | 0/1 | -0.042 | 0/1 |
| ros1.0 − none | -0.010 | 0/1 | -0.000 | 0/1 |
| weight − none | -0.003 | 0/1 | +0.004 | 1/1 |

### Synthetic-data fidelity (mean over folds)

| generator | ks_mean | ks_max | corr_diff | cat_tv_aa | cat_tv_dssp | detect_auc | nn_ratio |
|---|---|---|---|---|---|---|---|
| ctgan | 0.143 | 0.371 | 0.075 | 0.095 | 0.158 | 1.000 | 1.563 |
| smotenc | 0.055 | 0.112 | 0.019 | 0.042 | 0.090 | 0.984 | 0.666 |

ks/corr/tv: lower is closer to real. detect_auc: 0.5 = indistinguishable from real, 1.0 = trivially detected. nn_ratio: well below 1 suggests near-copies of real rows.
