"""Quantify the DSSP/sequence length mismatch and test whether a better offset exists.

For the 46 proteins whose DSSP length differs from the sequence, DSSP has no residue identity.
We fit a classifier P(DSSP state | amino acid, PSSM) on the 376 length-matched proteins, then
score every candidate offset of the DSSP array for each mismatched protein. If offset 0
(DeepPPISP's implicit rule: keep the first N rows) is not the best, a shift fits better.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from data_loading import AA, N_DSSP, N_PSSM, load_raw, protein_index

root = Path(__file__).resolve().parents[1]
seqs, pssm, dssp, labels, dset_of = load_raw(root / "data_cache")
prot = protein_index(root / "data_cache")


def feats(k):
    s = np.asarray(seqs[k])
    oh = np.zeros((len(s), 20), dtype=np.float32)
    oh[np.arange(len(s)), s] = 1
    return np.hstack([oh, np.asarray(pssm[k], dtype=np.float32)])


def state(d):  # state index, -1 for all-zero rows
    d = np.asarray(d)
    return np.where(d.sum(1) > 0, d.argmax(1), -1)


matched = [k for k in range(len(seqs)) if len(dssp[k]) == len(seqs[k])]
mism = [k for k in range(len(seqs)) if len(dssp[k]) != len(seqs[k])]
X = np.vstack([feats(k) for k in matched])
y = np.concatenate([state(dssp[k]) for k in matched])
keep = y >= 0
clf = LogisticRegression(max_iter=1000, C=1.0).fit(X[keep], y[keep])
classes = list(clf.classes_)


def score(k, shift):
    n = len(seqs[k])
    st = state(dssp[k])
    idx = np.arange(n) + shift
    ok = (idx >= 0) & (idx < len(st))
    s = np.full(n, -1)
    s[ok] = st[idx[ok]]
    use = s >= 0
    if use.sum() < 10:
        return -np.inf
    lp = clf.predict_log_proba(feats(k)[use])
    cols = np.array([classes.index(v) if v in classes else 0 for v in s[use]])
    return float(lp[np.arange(use.sum()), cols].mean())


rows = []
for k in mism:
    d = len(dssp[k]) - len(seqs[k])
    shifts = list(range(min(0, d), max(0, d) + 1))
    sc = {s: score(k, s) for s in shifts}
    best = max(sc, key=sc.get)
    rows.append(
        dict(
            protein_id=prot.loc[prot.id_idx == k, "protein_id"].iloc[0],
            idx=k,
            len_diff=d,
            best_shift=best,
            gain_vs_shift0=round(sc[best] - sc[0], 4),
        )
    )
df = pd.DataFrame(rows)
# reference: how noisy is this score for properly matched proteins? (shift 0 vs best nearby shift)
ref_gain = []
rng = np.random.default_rng(0)
for k in rng.choice(matched, 40, replace=False):
    sc = {s: score(k, s) for s in range(-3, 4)}
    ref_gain.append(max(sc.values()) - sc[0])
summary = {
    "n_mismatched": len(df),
    "best_shift_is_0": int((df.best_shift == 0).sum()),
    "best_shift_nonzero": int((df.best_shift != 0).sum()),
    "mean_gain_nonzero": round(float(df.loc[df.best_shift != 0, "gain_vs_shift0"].mean()), 4),
    "reference_gain_on_matched_proteins_mean": round(float(np.mean(ref_gain)), 4),
    "reference_gain_on_matched_proteins_p90": round(float(np.percentile(ref_gain, 90)), 4),
    "worst_proteins": df.sort_values("gain_vs_shift0", ascending=False).head(8).to_dict("records"),
}
# Apply a shift only when the gain clearly exceeds noise seen on properly matched proteins.
THRESH = 0.02
df["applied_shift"] = np.where(df.gain_vs_shift0 > THRESH, df.best_shift, 0)
summary["shift_threshold"] = THRESH
summary["n_shifts_applied"] = int((df.applied_shift != 0).sum())
(root / "configs").mkdir(exist_ok=True)
df[["protein_id", "len_diff", "applied_shift", "gain_vs_shift0"]].to_csv(
    root / "configs" / "dssp_shifts.csv", index=False
)
(root / "results").mkdir(exist_ok=True)
df.to_csv(root / "results" / "dssp_alignment_per_protein.csv", index=False)
(root / "results" / "dssp_alignment.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
