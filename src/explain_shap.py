"""SHAP explainability for the final fused model (models/final_model_F.json), on the held-out test proteins.

What it produces (results/):
  shap_family_importance.json   mean |SHAP| share by feature family and by window offset
  figures/shap_family_importance.png   which FAMILIES of features drive predictions (ESM-2 vs PSSM vs DSSP vs global ...)
  figures/shap_top_interpretable.png   top individual non-ESM features (PSSM / DSSP / one-hot / global); ESM dimensions are
                                      not individually interpretable, so they are only reported as a family
  figures/shap_beeswarm_interpretable.png   direction of effect for the top non-ESM features
  figures/shap_protein_<id>.png        per-residue view for example proteins: predicted probability along the chain,
                                      true interface residues, and which feature families push each residue up or down
SHAP values are in log-odds of the model's output (TreeExplainer, exact for tree ensembles). They describe the MODEL's behaviour,
not biology: a feature that drives predictions is not thereby shown to cause interaction.
Usage: python -X utf8 src/explain_shap.py [--n-sample 3000] [--proteins 3]
"""
from __future__ import annotations

import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from model_utils import load_sets, root

FIG = root / "results" / "figures"


def family_of(name: str) -> str:
    """Map a feature name to an interpretable family."""
    if name.startswith("esm_"):
        return "ESM-2 embedding"
    if name.startswith("ctx"):
        return "ESM-2 neighbourhood (mean of nearby residues)"
    if name.startswith("g_comp_"):
        return "global: amino-acid composition"
    if name.startswith("g_pssm_"):
        return "global: mean PSSM"
    if name.startswith("g_dssp_"):
        return "global: DSSP fractions"
    if name in ("g_log_len", "g_rel_pos"):
        return "global: length / relative position"
    base = name.split("@")[0]
    if base.startswith("aa_"):
        return "window: residue identity (one-hot)"
    if base.startswith("pssm_"):
        return "window: PSSM"
    if base.startswith("dssp_"):
        return "window: DSSP"
    return "other"


def offset_of(name: str):
    m = re.search(r"@([+-]\d+)$", name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=3000)
    ap.add_argument("--proteins", type=int, default=3)
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    meta = json.loads((root / "models" / "final_model_F_meta.json").read_text(encoding="utf-8"))
    names, thr = meta["feature_names"], meta["threshold"]
    model = xgb.XGBClassifier()
    model.load_model(str(root / "models" / "final_model_F.json"))
    res, sets, y, g, is_train = load_sets(meta.get("esm_size", "35M"), tuple(meta.get("context_halves", ())))
    X = sets[meta["feature_set"]]
    assert X.shape[1] == len(names) == meta["n_features"]
    te_idx = np.flatnonzero(~is_train)
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(te_idx, min(a.n_sample, len(te_idx)), replace=False))
    print(f"explaining {len(sample):,} random test residues (of {len(te_idx):,}); positive rate in sample {y[sample].mean():.3f}", flush=True)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X[sample])
    fam = np.array([family_of(n) for n in names])
    # sanity: SHAP values + base value must reproduce the model's raw margin
    margin = model.predict(X[sample], output_margin=True)
    err = float(np.abs(sv.sum(1) + explainer.expected_value - margin).max())
    print(f"additivity check, max |sum(SHAP)+base - margin| = {err:.2e}", flush=True)

    abs_sv = np.abs(sv)
    fam_share = {f: float(abs_sv[:, fam == f].sum() / abs_sv.sum()) for f in np.unique(fam)}
    fam_n = {f: int((fam == f).sum()) for f in np.unique(fam)}
    off = np.array([offset_of(n) if offset_of(n) is not None else 99 for n in names])
    win = np.array([f.startswith("window") for f in fam])
    off_share = {int(o): float(abs_sv[:, win & (off == o)].sum() / abs_sv[:, win].sum()) for o in sorted(set(off[win]))}
    mean_abs = abs_sv.mean(0)
    non_esm = np.flatnonzero(fam != "ESM-2 embedding")
    top_non_esm = [(names[i], float(mean_abs[i])) for i in non_esm[np.argsort(-mean_abs[non_esm])][:20]]
    esm_top = float(np.sort(mean_abs[fam == "ESM-2 embedding"])[::-1][:10].sum() / mean_abs[fam == "ESM-2 embedding"].sum())

    out = {"n_explained": int(len(sample)), "additivity_max_error": err, "family_share_of_total_abs_shap": fam_share,
           "family_n_features": fam_n, "window_offset_share": off_share, "top_non_esm_features": top_non_esm,
           "esm_top10_dims_share_of_esm_importance": esm_top, "model_threshold": thr}
    (root / "results" / "shap_family_importance.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for f, v in sorted(fam_share.items(), key=lambda t: -t[1]):
        print(f"  {f:40s} {v*100:5.1f}%  ({fam_n[f]} features)")

    # ---- figure 1: family importance
    order = sorted(fam_share, key=fam_share.get)
    plt.figure(figsize=(8, 4.2))
    plt.barh(order, [fam_share[f] * 100 for f in order], color="#4c78a8")
    for i, f in enumerate(order):
        plt.text(fam_share[f] * 100 + 0.4, i, f"{fam_share[f]*100:.1f}%", va="center", fontsize=8)
    plt.xlabel("share of total mean |SHAP| (%)"); plt.title("Which feature families drive the final model?")
    plt.tight_layout(); plt.savefig(FIG / "shap_family_importance.png", dpi=140); plt.close()

    # ---- figure 2: top interpretable (non-ESM) features
    top = non_esm[np.argsort(-mean_abs[non_esm])][:15][::-1]
    plt.figure(figsize=(8, 5))
    plt.barh([names[i] for i in top], mean_abs[top], color="#59a14f")
    plt.xlabel("mean |SHAP| (log-odds)"); plt.title("Top interpretable features (ESM-2 dimensions excluded)")
    plt.tight_layout(); plt.savefig(FIG / "shap_top_interpretable.png", dpi=140); plt.close()

    # ---- figure 3: beeswarm for the same features (direction of effect)
    top_desc = non_esm[np.argsort(-mean_abs[non_esm])][:15]
    shap.summary_plot(sv[:, top_desc], X[sample][:, top_desc], feature_names=[names[i] for i in top_desc], show=False, max_display=15)
    plt.tight_layout(); plt.savefig(FIG / "shap_beeswarm_interpretable.png", dpi=140); plt.close()

    # ---- figure 4+: per-protein residue views (test proteins with decent length and several true interface residues)
    te = res.loc[~is_train].reset_index(drop=True)
    probs = model.predict_proba(X[~is_train])[:, 1]
    te["prob"] = probs
    stats = te.groupby("protein_id").agg(n=("label", "size"), pos=("label", "sum"))
    cand = stats[(stats.n.between(80, 260)) & (stats.pos >= 12)].index.tolist()
    from sklearn.metrics import average_precision_score
    scored = sorted(cand, key=lambda p: -average_precision_score(te.loc[te.protein_id == p, "label"], te.loc[te.protein_id == p, "prob"]))
    chosen = [scored[0], scored[len(scored) // 2], scored[-1]][: a.proteins] if len(scored) >= 3 else scored[: a.proteins]
    for pid in chosen:
        rows = np.flatnonzero((res["protein_id"] == pid).to_numpy())
        svp = explainer.shap_values(X[rows])
        fams = sorted(np.unique(fam))
        mat = np.stack([svp[:, fam == f].sum(1) for f in fams])                # (families, residues)
        yy, pp = y[rows], model.predict_proba(X[rows])[:, 1]
        ap_ = average_precision_score(yy, pp)
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
        pos = np.arange(len(rows))
        ax[0].plot(pos, pp, color="#e15759", lw=1.2, label="predicted probability")
        ax[0].axhline(thr, color="gray", ls="--", lw=0.8, label=f"threshold {thr:.2f}")
        for i in np.flatnonzero(yy == 1):
            ax[0].axvspan(i - .5, i + .5, color="#4c78a8", alpha=.25)
        ax[0].set_ylabel("probability"); ax[0].legend(loc="upper right", fontsize=7)
        ax[0].set_title(f"{pid}: blue bands = true interface residues   (per-protein PR-AUC {ap_:.2f}, {int(yy.sum())}/{len(yy)} positive)")
        lim = np.abs(mat).max()
        im = ax[1].imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        ax[1].set_yticks(range(len(fams))); ax[1].set_yticklabels(fams, fontsize=7)
        ax[1].set_xlabel("residue position"); plt.colorbar(im, ax=ax[1], label="SHAP (red = pushes toward interface)")
        plt.tight_layout(); plt.savefig(FIG / f"shap_protein_{pid}.png", dpi=140); plt.close()
        print(f"  wrote shap_protein_{pid}.png", flush=True)
    print("done")


if __name__ == "__main__":
    main()
