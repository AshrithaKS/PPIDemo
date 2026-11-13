"""Balancing-helper (and bootstrap) invariants (small synthetic data; no CTGAN training)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from balancing import (AA_LIST, DSSP_LIST, encode, fidelity, n_synthetic, random_oversample,  # noqa: E402
                       smote_nc)


def toy(n=600, pos_rate=0.15, seed=0, k=4):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    df = pd.DataFrame({"aa": rng.choice(AA_LIST, n), "dssp_state": rng.choice(DSSP_LIST, n)})
    for i in range(1, 21):
        df[f"pssm_{i}"] = rng.normal(y * 0.8, 1.0, n).astype(np.float32)
    for i in range(1, k + 1):
        df[f"pc_{i}"] = rng.normal(y * 0.5, 1.0, n).astype(np.float32)
    return df, y


def test_n_synthetic():
    assert n_synthetic(100, 900, 1.0) == 800
    assert n_synthetic(100, 900, 0.5) == 350
    assert n_synthetic(900, 900, 1.0) == 0 and n_synthetic(1000, 900, 1.0) == 0


def test_encode_shape_and_onehot():
    df, _ = toy()
    X = encode(df)
    assert X.shape == (len(df), 20 + 10 + 20 + 4)
    assert (X[:, :20].sum(1) == 1).all() and (X[:, 20:30].sum(1) == 1).all()


def test_random_oversample_reaches_ratio_and_copies_minority_only():
    df, y = toy()
    syn = random_oversample(df, y, 1.0)
    assert len(syn) == n_synthetic(int(y.sum()), int((y == 0).sum()), 1.0)
    minority = df[y == 1]
    key = lambda d: d[["pssm_1", "pc_1"]].round(6).apply(tuple, axis=1)
    assert set(key(syn)) <= set(key(minority))


def test_smote_nc_adds_only_valid_synthetic_rows():
    df, y = toy()
    syn = smote_nc(df, y, 0.5)
    assert len(syn) == n_synthetic(int(y.sum()), int((y == 0).sum()), 0.5)
    assert set(syn["aa"]) <= set(AA_LIST) and set(syn["dssp_state"]) <= set(DSSP_LIST)
    assert list(syn.columns) == list(df.columns) and np.isfinite(syn.select_dtypes("number").to_numpy()).all()


def test_fidelity_detects_bad_vs_good_synthetic():
    real, y = toy(3000, 0.5, seed=1)
    real = real[y == 1].reset_index(drop=True)
    good, y2 = toy(3000, 0.5, seed=2)
    good = good[y2 == 1].reset_index(drop=True)
    bad = good.copy()
    for c in bad.columns:
        if c.startswith(("pssm_", "pc_")):
            bad[c] = bad[c] + 3.0
    fg, fb = fidelity(real, good, max_n=500), fidelity(real, bad, max_n=500)
    assert fg["ks_mean"] < fb["ks_mean"] and fg["detect_auc"] < fb["detect_auc"]
    assert fg["detect_auc"] < 0.65


def _fake_folds():
    def arm(mcc, pr):
        m = dict(accuracy=.7, precision=.3, recall=.4, f1=.3, mcc=mcc, pr_auc=pr)
        return {"test_at_val_threshold": m, "test_at_0.5": m}
    return {str(k): {"arms": {"none": arm(.10, .20), "ctgan1.0": arm(.10 + d, .20 - d)}, "fidelity": {}}
            for k, d in enumerate([.02, .01, -.01])}


def test_paired_differences_and_partial_summary():
    from balancing_summary import paired, summarise
    f = _fake_folds()
    p = paired(f, "ctgan1.0", "none", "mcc")
    assert p["n"] == 3 and p["better"] == 2 and p["mean"] == pytest.approx(0.00667, abs=1e-4)
    assert paired(f, "ctgan1.0", "none", "pr_auc")["better"] == 1
    text = summarise({"folds": f})
    assert "Folds finished: 3 of 5" in text and "ctgan1.0 − none" in text and "smotenc" not in text


def test_protein_bootstrap_paired_differences_and_ci():
    from model_utils import protein_bootstrap
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(60), 30)
    y = (rng.random(len(g)) < 0.2).astype(int)
    good = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.15, len(y)), 0, 1)
    noise = rng.random(len(y))
    ci, diffs = protein_bootstrap(y, {"good": good, "noise": noise}, g, {"good": 0.45, "noise": 0.5}, n_boot=200)
    assert ci["good"]["mcc"][0] > ci["noise"]["mcc"][1]                       # intervals separate
    d = diffs["good - noise"]["mcc"]
    assert d["mean"] > 0.3 and d["ci"][0] > 0 and d["p_better"] == pytest.approx(1.0)
    assert ci["good"]["pr_auc"][0] <= ci["good"]["pr_auc"][1]
