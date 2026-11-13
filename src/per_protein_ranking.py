"""Does the final model rank residues WITHIN each protein, or mostly just rank proteins by size?

Pooled MCC / PR-AUC mix two abilities: (a) telling which proteins have many interface residues (protein length alone does this:
shorter proteins have a larger interface fraction) and (b) telling which residues inside one protein are interface residues.
This script scores (b) separately using results/test_predictions.parquet (written by final_evaluation.py):
  * per-protein average precision of each model, averaged over test proteins (each protein is its own ranking problem)
  * how often that beats the protein's own base rate (what a random ranking within the protein would score)
  * how often the fused model beats the handcrafted baseline protein by protein
  * Spearman correlation between protein length and interface fraction (the size prior)
Writes results/per_protein_ranking.json.   Usage: python -X utf8 src/per_protein_ranking.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

root = Path(__file__).resolve().parents[1]


def per_protein_table(te: pd.DataFrame, cols: list[str], min_pos: int = 3) -> pd.DataFrame:
    rows = []
    for pid, g in te.groupby("protein_id"):
        if g["label"].sum() < min_pos or g["label"].sum() == len(g):
            continue
        r = {"protein_id": pid, "n": len(g), "n_pos": int(g["label"].sum()), "base_rate": float(g["label"].mean())}
        for c in cols:
            r[c] = float(average_precision_score(g["label"], g[c]))
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    te = pd.read_parquet(root / "results" / "test_predictions.parquet")
    cols = [c for c in te.columns if c.startswith("prob_")]
    t = per_protein_table(te, cols)
    out = {"n_proteins": int(len(t)), "mean_base_rate": float(t["base_rate"].mean()), "models": {}}
    for c in cols:
        out["models"][c.replace("prob_", "")] = {
            "mean_within_protein_AP": float(t[c].mean()),
            "median_within_protein_AP": float(t[c].median()),
            "n_proteins_above_own_base_rate": int((t[c] > t["base_rate"]).sum()),
        }
    if {"prob_F_final", "prob_H_baseline"} <= set(cols):
        out["fused_beats_baseline_in_n_proteins"] = int((t["prob_F_final"] > t["prob_H_baseline"]).sum())
        out["mean_within_protein_AP_gain_fused_minus_baseline"] = float((t["prob_F_final"] - t["prob_H_baseline"]).mean())
    al = te.groupby("protein_id").agg(n=("label", "size"), rate=("label", "mean"))
    out["spearman_length_vs_interface_fraction"] = float(al["n"].corr(al["rate"], method="spearman"))
    # how much of the pooled signal is the size prior alone? score every residue with its protein's mean predicted probability
    for c in cols:
        prot_mean = te.groupby("protein_id")[c].transform("mean")
        out["models"][c.replace("prob_", "")]["pooled_PR_AUC"] = float(average_precision_score(te["label"], te[c]))
        out["models"][c.replace("prob_", "")]["pooled_PR_AUC_protein_mean_only"] = float(average_precision_score(te["label"], prot_mean))
    (root / "results" / "per_protein_ranking.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    t.to_csv(root / "results" / "per_protein_ranking.csv", index=False)


if __name__ == "__main__":
    main()
