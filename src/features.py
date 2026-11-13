"""Handcrafted per-residue features with a sliding window (baseline features).

Per residue: one-hot amino acid (20) + PSSM (20) + DSSP one-hot (9) = 49 values.
(The reference paper also uses RSA -> 50 values; RSA is NOT available in the DeepPPISP pickles.)
Window = 2*half+1 residues centred on the target (default 7 -> 343 features). Positions that fall
outside the protein are zero-filled; windows never cross protein boundaries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_loading import AA

PSSM_COLS = [f"pssm_{i}" for i in range(1, 21)]
DSSP_COLS = [f"dssp_{i}" for i in range(1, 10)]
PER_RES = 49


def per_residue_matrix(res: pd.DataFrame) -> np.ndarray:
    """(n_residues, 49) float32: one-hot aa, PSSM, DSSP."""
    n = len(res)
    oh = np.zeros((n, 20), dtype=np.float32)
    oh[np.arange(n), res["aa_idx"].to_numpy()] = 1.0
    return np.hstack([oh, res[PSSM_COLS].to_numpy(np.float32), res[DSSP_COLS].to_numpy(np.float32)])


def feature_names(half: int = 3) -> list[str]:
    base = [f"aa_{a}" for a in AA] + PSSM_COLS + DSSP_COLS
    return [f"{b}@{o:+d}" for o in range(-half, half + 1) for b in base]


def window_features(res: pd.DataFrame, half: int = 3) -> np.ndarray:
    """(n_residues, (2*half+1)*49). `res` must be ordered by protein then position."""
    base = per_residue_matrix(res)
    out = np.zeros((len(res), (2 * half + 1) * PER_RES), dtype=np.float32)
    pid = res["protein_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(pid[1:] != pid[:-1]) + 1, len(res)]
    for s, e in zip(starts[:-1], starts[1:]):
        block = base[s:e]
        n = e - s
        for k, off in enumerate(range(-half, half + 1)):
            src = np.arange(n) + off
            ok = (src >= 0) & (src < n)
            out[s:e][ok, k * PER_RES : (k + 1) * PER_RES] = block[src[ok]]
    return out


def window_mean(X: np.ndarray, groups: np.ndarray, half: int) -> np.ndarray:
    """Mean of the rows of X over a window of +-half residues, never crossing protein boundaries (edges use the part that exists).

    Used for wider-context ESM-2 features: interface residues form surface patches, so the neighbourhood's embedding carries signal.
    `groups` holds the protein id per row (rows ordered by protein then position). Works for a single protein with groups = zeros.
    """
    out = np.empty_like(X, dtype=np.float32)
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(X)]
    for s, e in zip(starts[:-1], starts[1:]):
        n = e - s
        c = np.vstack([np.zeros((1, X.shape[1]), np.float64), np.cumsum(X[s:e], axis=0, dtype=np.float64)])
        lo, hi = np.clip(np.arange(n) - half, 0, n), np.clip(np.arange(n) + half + 1, 0, n)
        out[s:e] = ((c[hi] - c[lo]) / (hi - lo)[:, None]).astype(np.float32)
    return out


def smooth_probs(p: np.ndarray, groups: np.ndarray, half: int, alpha: float) -> np.ndarray:
    """alpha * own probability + (1 - alpha) * mean probability of the +-half neighbourhood (within each protein)."""
    if half == 0 or alpha >= 1:
        return p
    return alpha * p + (1 - alpha) * window_mean(p[:, None], groups, half)[:, 0]


GLOBAL_NAMES = (
    [f"g_comp_{a}" for a in AA] + [f"g_pssm_{i}" for i in range(1, 21)]
    + [f"g_dssp_{i}" for i in range(1, 10)] + ["g_log_len", "g_rel_pos"]
)


def global_features(res: pd.DataFrame) -> np.ndarray:
    """(n_residues, 51) whole-chain context: amino-acid composition (20), mean PSSM (20),
    DSSP state fractions (9), log length, relative position of the residue in the chain.

    This is a compact stand-in for the paper's whole-chain block (500 residues x 50 values, whose
    dimensionality the paper does not reconcile with its stated 507-unit input layer). All but the
    last column are CONSTANT within a protein, so they can leak under residue-level random splits.
    """
    base = per_residue_matrix(res)
    out = np.zeros((len(res), 51), dtype=np.float32)
    pid = res["protein_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(pid[1:] != pid[:-1]) + 1, len(res)]
    for s, e in zip(starts[:-1], starts[1:]):
        n = e - s
        out[s:e, :49] = base[s:e].mean(axis=0)
        out[s:e, 49] = np.log(n)
        out[s:e, 50] = np.arange(n) / max(n - 1, 1)
    return out
