"""Precompute the assets the demo app shows for the 70 held-out test proteins, using the saved demo model (models/demo_model_S.json).

Writes
  results/demo_test_probabilities.parquet   per-residue probabilities (so held-out examples load instantly, no language model needed)
  results/demo_threshold_curve.csv          precision / recall / MCC / share-flagged for every threshold on the 70 held-out proteins
  results/demo_family_importance.json       mean |contribution| of each feature family for the demo model on held-out residues
Usage: python -X utf8 src/demo_assets.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from inference import families, family_contributions, load_demo_model
from model_utils import load_sets, root


def main():
    model, meta = load_demo_model()
    FAMILIES = families(meta)
    res, sets, y, g, is_train = load_sets(meta["esm_size"], tuple(meta["context_halves"]))
    X = sets["S"][~is_train]
    te = res.loc[~is_train, ["protein_id", "position", "aa", "label"]].reset_index(drop=True)
    te["prob"] = model.predict_proba(X)[:, 1].astype(np.float32)
    te.to_parquet(root / "results" / "demo_test_probabilities.parquet", index=False)

    yt, p = te["label"].to_numpy(), te["prob"].to_numpy()
    rows = []
    for t in np.round(np.arange(0.02, 0.81, 0.01), 2):
        pred = p >= t
        tp = int((pred & (yt == 1)).sum())
        rows.append({"threshold": float(t), "precision": tp / max(int(pred.sum()), 1), "recall": tp / int(yt.sum()),
                     "mcc": float(matthews_corrcoef(yt, pred)) if pred.any() else 0.0, "share_flagged": float(pred.mean())})
    curve = pd.DataFrame(rows)
    curve.to_csv(root / "results" / "demo_threshold_curve.csv", index=False)

    fam, bias = family_contributions(model, X, meta)
    imp = np.abs(fam).mean(0)
    out = {"families": FAMILIES, "mean_abs_contribution": [float(v) for v in imp], "share": [float(v) for v in imp / imp.sum()],
           "base_value_logodds": bias, "n_residues": int(len(X))}
    (root / "results" / "demo_family_importance.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    at = curve.loc[(curve.threshold - meta["threshold"]).abs().idxmin()]
    print(f"curve at default threshold {meta['threshold']}: precision {at.precision:.3f} recall {at.recall:.3f} mcc {at.mcc:.4f} "
          f"(meta says mcc {meta['test_metrics_at_threshold']['mcc']:.4f})")
    print("family share of mean |contribution|:", {f: round(s, 3) for f, s in zip(FAMILIES, out["share"])})
    margin = model.predict(X, output_margin=True)
    print("additivity max error:", float(np.abs(fam.sum(1) + bias - margin).max()))


if __name__ == "__main__":
    main()
