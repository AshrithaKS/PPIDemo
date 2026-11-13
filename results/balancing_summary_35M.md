Folds finished: 5 of 5

### Test metrics at the validation-tuned threshold (mean±std over folds)

| arm | accuracy | precision | recall | f1 | mcc | pr_auc | MCC at 0.5 |
|---|---|---|---|---|---|---|---|
| none | 0.663±0.088 | 0.237±0.026 | 0.494±0.175 | 0.302±0.037 | 0.150±0.016 | 0.252±0.017 | 0.054 |
| weight | 0.678±0.079 | 0.236±0.016 | 0.465±0.150 | 0.303±0.025 | 0.147±0.019 | 0.244±0.016 | 0.149 |
| ros1.0 | 0.626±0.064 | 0.223±0.019 | 0.559±0.108 | 0.314±0.012 | 0.148±0.015 | 0.243±0.018 | 0.148 |
| smotenc0.5 | 0.640±0.074 | 0.224±0.023 | 0.516±0.140 | 0.301±0.023 | 0.138±0.014 | 0.231±0.017 | 0.044 |
| smotenc1.0 | 0.574±0.072 | 0.205±0.008 | 0.603±0.124 | 0.302±0.011 | 0.128±0.012 | 0.224±0.012 | 0.080 |
| ctgan0.5 | 0.626±0.069 | 0.223±0.010 | 0.564±0.121 | 0.316±0.013 | 0.152±0.012 | 0.246±0.018 | 0.067 |
| ctgan1.0 | 0.626±0.069 | 0.222±0.013 | 0.560±0.126 | 0.313±0.017 | 0.149±0.016 | 0.245±0.013 | 0.061 |

### Paired per-fold differences (first arm minus second)

| comparison | ΔMCC | folds better | ΔPR-AUC | folds better |
|---|---|---|---|---|
| ctgan0.5 − none | +0.002 | 3/5 | -0.005 | 1/5 |
| ctgan1.0 − none | -0.001 | 2/5 | -0.007 | 0/5 |
| ctgan1.0 − weight | +0.003 | 2/5 | +0.000 | 3/5 |
| ctgan1.0 − ros1.0 | +0.001 | 3/5 | +0.002 | 3/5 |
| ctgan1.0 − smotenc1.0 | +0.021 | 5/5 | +0.021 | 5/5 |
| ctgan0.5 − smotenc0.5 | +0.014 | 5/5 | +0.015 | 5/5 |
| ros1.0 − none | -0.001 | 2/5 | -0.009 | 0/5 |
| weight − none | -0.003 | 2/5 | -0.008 | 0/5 |

### Synthetic-data fidelity (mean over folds)

| generator | ks_mean | ks_max | corr_diff | cat_tv_aa | cat_tv_dssp | detect_auc | nn_ratio |
|---|---|---|---|---|---|---|---|
| ctgan | 0.141 | 0.330 | 0.080 | 0.082 | 0.156 | 1.000 | 1.768 |
| smotenc | 0.053 | 0.097 | 0.021 | 0.036 | 0.104 | 0.976 | 0.645 |

ks/corr/tv: lower is closer to real. detect_auc: 0.5 = indistinguishable from real, 1.0 = trivially detected. nn_ratio: well below 1 suggests near-copies of real rows.
