"""EDA: class balance, length distribution, split composition. Writes PNGs to results/figures."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
out = root / "results" / "figures"; out.mkdir(parents=True, exist_ok=True)
r = pd.read_parquet(root / "data" / "processed" / "residues.parquet")
p = pd.read_csv(root / "data" / "processed" / "proteins.csv")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
# 1 class balance per dataset / split
g = pd.concat([r.groupby("dset")["label"].mean(), r.groupby("split")["label"].mean(), pd.Series({"pooled": r.label.mean()})])
ax[0].bar(g.index, g.values * 100, color="#4c78a8"); ax[0].axhline(15.4, ls="--", c="gray")
for i, v in enumerate(g.values): ax[0].text(i, v * 100 + .3, f"{v*100:.1f}%", ha="center")
ax[0].set_ylabel("% interface residues"); ax[0].set_title("Positive rate"); ax[0].tick_params(axis="x", rotation=30)
# 2 length distribution
ax[1].hist(p.length, bins=50, color="#59a14f"); ax[1].axvline(500, c="red", ls="--", label="500 (paper cap)")
ax[1].axvline(1022, c="purple", ls="--", label="1022 (ESM-2 limit)"); ax[1].set_yscale("log")
ax[1].set_xlabel("protein length"); ax[1].set_ylabel("proteins (log)"); ax[1].set_title("Protein length"); ax[1].legend()
# 3 per-protein positive rate
for s, c in (("train", "#4c78a8"), ("test", "#e15759")):
    ax[2].hist(p.loc[p.split == s, "pos_rate"] * 100, bins=25, alpha=.6, label=s, color=c, density=True)
ax[2].set_xlabel("% positives per protein"); ax[2].set_title("Per-protein positive rate"); ax[2].legend()
plt.tight_layout(); plt.savefig(out / "data_overview.png", dpi=130); plt.close()

# amino-acid composition: positives vs negatives
comp = pd.crosstab(r.aa, r.label, normalize="columns") * 100
comp.plot.bar(figsize=(9, 4), color=["#9ecae9", "#e15759"]); plt.ylabel("% of class"); plt.title("Amino-acid composition by class")
plt.legend(["non-interface", "interface"]); plt.tight_layout(); plt.savefig(out / "aa_composition_by_class.png", dpi=130); plt.close()

# DSSP state frequency by class
d = [f"dssp_{i}" for i in range(1, 10)]
ds = r.groupby("label")[d].mean().T * 100
ds.plot.bar(figsize=(8, 4), color=["#9ecae9", "#e15759"]); plt.ylabel("% of residues in class"); plt.title("DSSP state by class")
plt.legend(["non-interface", "interface"]); plt.tight_layout(); plt.savefig(out / "dssp_by_class.png", dpi=130); plt.close()
print("wrote", sorted(x.name for x in out.glob("*.png")))
print(comp.round(1).assign(diff=lambda t: (t[1]-t[0]).round(2)).sort_values("diff").tail(4)[["diff"]].T.to_string())
