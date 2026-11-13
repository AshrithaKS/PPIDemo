"""Summarise results/balancing_comparison_<tag>.json: per-arm table, paired per-fold deltas, synthetic-data fidelity.

Usage: python -X utf8 src/balancing_summary.py [tag]      -> prints and writes results/balancing_summary_<tag>.md
Works on partial results (any number of finished folds), and says how many folds it used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
ARMS = ["none", "weight", "ros1.0", "smotenc0.5", "smotenc1.0", "ctgan0.5", "ctgan1.0"]
METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "pr_auc"]


def arm_values(folds: dict, arm: str, key: str, metric: str) -> np.ndarray:
    return np.array([folds[k]["arms"][arm][key][metric] for k in sorted(folds) if arm in folds[k]["arms"]])


def paired(folds: dict, a: str, b: str, metric: str, key: str = "test_at_val_threshold") -> dict:
    """Per-fold metric(a) - metric(b), using only folds that contain both arms."""
    ks = [k for k in sorted(folds) if a in folds[k]["arms"] and b in folds[k]["arms"]]
    d = np.array([folds[k]["arms"][a][key][metric] - folds[k]["arms"][b][key][metric] for k in ks])
    return {"mean": float(d.mean()) if len(d) else float("nan"), "better": int((d > 0).sum()), "n": int(len(d)),
            "diffs": [round(float(x), 3) for x in d]}


def summarise(data: dict) -> str:
    folds = data["folds"]
    L = [f"Folds finished: {len(folds)} of 5", "",
         "### Test metrics at the validation-tuned threshold (mean±std over folds)", "",
         "| arm | " + " | ".join(METRICS) + " | MCC at 0.5 |", "|---|" + "---|" * (len(METRICS) + 1)]
    for arm in ARMS:
        if not any(arm in f["arms"] for f in folds.values()):
            continue
        cells = []
        for m in METRICS:
            v = arm_values(folds, arm, "test_at_val_threshold", m)
            cells.append(f"{v.mean():.3f}±{v.std():.3f}")
        v5 = arm_values(folds, arm, "test_at_0.5", "mcc")
        L.append(f"| {arm} | " + " | ".join(cells) + f" | {v5.mean():.3f} |")

    L += ["", "### Paired per-fold differences (first arm minus second)", "",
          "| comparison | ΔMCC | folds better | ΔPR-AUC | folds better |", "|---|---|---|---|---|"]
    for a, b in [("ctgan0.5", "none"), ("ctgan1.0", "none"), ("ctgan1.0", "weight"), ("ctgan1.0", "ros1.0"),
                 ("ctgan1.0", "smotenc1.0"), ("ctgan0.5", "smotenc0.5"), ("ros1.0", "none"), ("weight", "none")]:
        pm, pp = paired(folds, a, b, "mcc"), paired(folds, a, b, "pr_auc")
        if pm["n"]:
            L.append(f"| {a} − {b} | {pm['mean']:+.3f} | {pm['better']}/{pm['n']} | {pp['mean']:+.3f} | {pp['better']}/{pp['n']} |")

    fid = {g: pd.DataFrame([f["fidelity"][g] for f in folds.values() if g in f.get("fidelity", {})]) for g in ("ctgan", "smotenc")}
    if len(fid["ctgan"]):
        cols = list(fid["ctgan"].columns)
        L += ["", "### Synthetic-data fidelity (mean over folds)", "",
              "| generator | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for g, df in fid.items():
            L.append(f"| {g} | " + " | ".join(f"{df[c].mean():.3f}" for c in cols) + " |")
        L += ["", "ks/corr/tv: lower is closer to real. detect_auc: 0.5 = indistinguishable from real, 1.0 = trivially detected. "
                  "nn_ratio: well below 1 suggests near-copies of real rows."]
    return "\n".join(L)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "35M"
    data = json.loads((root / "results" / f"balancing_comparison_{tag}.json").read_text(encoding="utf-8"))
    text = summarise(data)
    (root / "results" / f"balancing_summary_{tag}.md").write_text(text + "\n", encoding="utf-8")
    print(text)
