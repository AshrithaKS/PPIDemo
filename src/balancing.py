"""Class-balancing helpers: reduced per-residue feature table, encoders, balancing arms, synthetic-data fidelity.

Reduced feature table (what CTGAN models; every arm uses the SAME 82 encoded inputs):
  aa          categorical (20)           dssp_state  categorical ('0'..'8' or 'none')
  pssm_1..20  continuous                 pc_1..pc_K  PCA of the ESM-2 embedding (K=32, fitted on TRAIN rows only)
Window and global features are NOT used here: synthetic residues have no neighbours and no protein identity.
All balancing is applied to training rows only; validation/test stay at the natural class ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

from data_loading import AA

AA_LIST = list(AA)
DSSP_LIST = [str(i) for i in range(9)] + ["none"]
PSSM_COLS = [f"pssm_{i}" for i in range(1, 21)]
# continuous columns = anything with these prefixes (pssm_/pc_ = per-residue; gpc_/g_ = global; wpc_ = neighbour PSSM)
CONT_PREFIXES = ("pssm_", "pc_", "gpc_", "g_", "wpc_")


def pc_cols(k: int) -> list[str]:
    return [f"pc_{i}" for i in range(1, k + 1)]


def fit_pca(esm_train: np.ndarray, k: int = 32, seed: int = 0) -> PCA:
    return PCA(n_components=k, random_state=seed).fit(esm_train)


def make_table(res: pd.DataFrame, esm: np.ndarray, pca: PCA) -> pd.DataFrame:
    """Reduced residue table for the rows of `res` (row-aligned with `esm`)."""
    d = res[[f"dssp_{i}" for i in range(1, 10)]].to_numpy()
    state = np.where(d.sum(1) > 0, d.argmax(1).astype(str), "none")
    df = pd.DataFrame({"aa": res["aa"].to_numpy(), "dssp_state": state})
    for c in PSSM_COLS:
        df[c] = res[c].to_numpy(np.float32)
    pcs = pca.transform(esm).astype(np.float32)
    for j, c in enumerate(pc_cols(pca.n_components_)):
        df[c] = pcs[:, j]
    return df.reset_index(drop=True)


def encode(df: pd.DataFrame) -> np.ndarray:
    """One-hot the two categoricals with FIXED category lists -> (n, 20 + 10 + 20 + K) float32."""
    n = len(df)
    oh_aa = np.zeros((n, len(AA_LIST)), np.float32)
    oh_aa[np.arange(n), df["aa"].map({a: i for i, a in enumerate(AA_LIST)}).to_numpy()] = 1
    oh_d = np.zeros((n, len(DSSP_LIST)), np.float32)
    oh_d[np.arange(n), df["dssp_state"].map({a: i for i, a in enumerate(DSSP_LIST)}).to_numpy()] = 1
    cont = df[[c for c in df.columns if c.startswith(CONT_PREFIXES)]].to_numpy(np.float32)
    return np.hstack([oh_aa, oh_d, cont])


def _to_numeric(df: pd.DataFrame) -> np.ndarray:
    a = df["aa"].map({x: i for i, x in enumerate(AA_LIST)}).to_numpy(np.float64)[:, None]
    d = df["dssp_state"].map({x: i for i, x in enumerate(DSSP_LIST)}).to_numpy(np.float64)[:, None]
    cont = df[[c for c in df.columns if c.startswith(CONT_PREFIXES)]].to_numpy(np.float64)
    return np.hstack([a, d, cont])


def _from_numeric(a: np.ndarray, columns) -> pd.DataFrame:
    cont_cols = [c for c in columns if c.startswith(CONT_PREFIXES)]
    out = pd.DataFrame({"aa": [AA_LIST[int(round(v))] for v in a[:, 0]],
                        "dssp_state": [DSSP_LIST[int(round(v))] for v in a[:, 1]]})
    for j, c in enumerate(cont_cols):
        out[c] = a[:, 2 + j].astype(np.float32)
    return out[list(columns)]


def n_synthetic(n_min: int, n_maj: int, ratio: float) -> int:
    """Rows to add so that minority/majority == ratio."""
    return max(0, int(ratio * n_maj) - n_min)   # floor, same as imblearn


def random_oversample(df: pd.DataFrame, y: np.ndarray, ratio: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    add = rng.choice(pos, n_synthetic(len(pos), int((y == 0).sum()), ratio), replace=True)
    return df.iloc[add].reset_index(drop=True)


def smote_nc(df: pd.DataFrame, y: np.ndarray, ratio: float, seed: int = 0) -> pd.DataFrame:
    """Synthetic minority rows from SMOTE-NC (categoricals = aa, dssp_state)."""
    from imblearn.over_sampling import SMOTENC
    X = _to_numeric(df)
    sm = SMOTENC(categorical_features=[0, 1], sampling_strategy=ratio, k_neighbors=5, random_state=seed)
    Xr, yr = sm.fit_resample(X, y)
    new = Xr[len(X):]
    return _from_numeric(new, df.columns)


def _patch_ctgan_serial():
    """CTGAN's transformer uses joblib/loky worker processes, which crash on Windows (BrokenProcessPool).
    Force serial execution; the data transform is a one-off cost per fit."""
    import joblib
    import ctgan.data_transformer as dt
    if not getattr(dt, "_serial_patched", False):
        dt.Parallel = lambda *a, **k: joblib.Parallel(n_jobs=1)
        dt._serial_patched = True


def fit_ctgan(minority: pd.DataFrame, epochs: int = 200, seed: int = 0, batch_size: int = 500):
    import torch
    from ctgan import CTGAN
    _patch_ctgan_serial()
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = CTGAN(epochs=epochs, batch_size=batch_size, pac=10, cuda=False, verbose=False)
    g.fit(minority, discrete_columns=["aa", "dssp_state"])
    return g


def ctgan_sample(g, n: int) -> pd.DataFrame:
    s = g.sample(n)
    s["aa"] = s["aa"].astype(str)
    s["dssp_state"] = s["dssp_state"].astype(str)
    return s.reset_index(drop=True)


def fidelity(real_min: pd.DataFrame, syn: pd.DataFrame, seed: int = 0, max_n: int = 4000) -> dict:
    """How realistic are synthetic minority rows?  (lower is better unless noted)
    ks_mean/ks_max : two-sample KS statistic over continuous columns (0 = identical marginals)
    corr_diff      : mean |corr_real - corr_syn| over continuous columns
    cat_tv_aa/dssp : total-variation distance of categorical frequencies
    detect_auc     : AUC of a classifier separating real from synthetic (0.5 = indistinguishable, 1.0 = trivially detected)
    nn_ratio       : median NN distance synthetic->real divided by real->other-real (<<1 suggests copying real rows)
    """
    rng = np.random.default_rng(seed)
    r = real_min.sample(min(len(real_min), max_n), random_state=seed).reset_index(drop=True)
    s = syn.sample(min(len(syn), max_n), random_state=seed).reset_index(drop=True)
    cont = [c for c in r.columns if c.startswith(CONT_PREFIXES)]
    ks = [ks_2samp(r[c], s[c]).statistic for c in cont]
    corr = float(np.abs(np.corrcoef(r[cont].to_numpy().T) - np.corrcoef(s[cont].to_numpy().T)).mean())
    tv = {}
    for c in ("aa", "dssp_state"):
        pr, ps = r[c].value_counts(normalize=True), s[c].value_counts(normalize=True)
        idx = pr.index.union(ps.index)
        tv[c] = float(0.5 * np.abs(pr.reindex(idx, fill_value=0) - ps.reindex(idx, fill_value=0)).sum())
    n = min(len(r), len(s))
    Xd = np.vstack([encode(r.iloc[:n]), encode(s.iloc[:n])])
    yd = np.r_[np.zeros(n), np.ones(n)]
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=100, random_state=seed), Xd, yd, cv=5, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(yd, p))
    Z = r[cont].to_numpy()
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    Zr, Zs = (Z - mu) / sd, (s[cont].to_numpy() - mu) / sd
    nn = NearestNeighbors(n_neighbors=2).fit(Zr)
    d_ss = nn.kneighbors(Zs, n_neighbors=1)[0][:, 0]
    d_rr = nn.kneighbors(Zr, n_neighbors=2)[0][:, 1]
    return {"ks_mean": float(np.mean(ks)), "ks_max": float(np.max(ks)), "corr_diff": corr,
            "cat_tv_aa": tv["aa"], "cat_tv_dssp": tv["dssp_state"], "detect_auc": auc,
            "nn_ratio": float(np.median(d_ss) / max(np.median(d_rr), 1e-9))}


def fit_context_pcas(G_train: np.ndarray, Wm_train: np.ndarray, k: int = 8, seed: int = 0):
    """PCA (after standardising) of the 49 protein-constant global columns and of the neighbour-mean PSSM (20)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    gp = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=seed)).fit(G_train[:, :49])
    wp = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=seed)).fit(Wm_train)
    return gp, wp


def make_table_fused(res, esm, pca, gp, wp, G, Wm) -> pd.DataFrame:
    """Reduced residue table + compact context: global PCs, log length, relative position, neighbour-PSSM PCs."""
    df = make_table(res, esm, pca)
    gpc, wpc = gp.transform(G[:, :49]).astype(np.float32), wp.transform(Wm).astype(np.float32)
    for j in range(gpc.shape[1]):
        df[f"gpc_{j + 1}"] = gpc[:, j]
    df["g_log_len"] = G[:, 49]
    df["g_rel_pos"] = G[:, 50]
    for j in range(wpc.shape[1]):
        df[f"wpc_{j + 1}"] = wpc[:, j]
    return df
