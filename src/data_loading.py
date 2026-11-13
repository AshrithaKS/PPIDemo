"""Load the DeepPPISP benchmark pickles into one aligned residue-level table.

Source: https://github.com/CSUBioGroup/DeepPPISP  (data_cache folder, copied read-only).

Facts verified from the DeepPPISP code and the files themselves:
  * Sequences are integer-encoded with  "ACDEFGHIKLMNPQRSTVWY"  (utils/config.py: acid_idex).
  * `all_dset_list.pkl` rows are (count, id_idx, position, dset, protein_id, seq_length).
    `id_idx` is a GLOBAL protein index in the order dset186 -> dset164 -> dset72.
  * `training_list.pkl` / `testing_list.pkl` hold indices into `all_dset_list` (residue level),
    so we collapse them to a PROTEIN-level split (352 train / 70 test proteins, no overlap).

DSSP alignment rule (46 of 422 proteins have a DSSP length different from the sequence):
  DSSP rows carry no residue identity. Default is DeepPPISP's rule (keep the first len(sequence)
  rows, zero-pad if shorter). For proteins listed in configs/dssp_shifts.csv (produced by
  src/audit_dssp_alignment.py) a data-driven offset is applied instead, only where it fits the
  sequence/PSSM clearly better than offset 0. Rows with no DSSP are zero and flagged in
  `dssp_missing`; the per-protein length difference is kept in `dssp_len_diff`.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
# Order matters: id_idx in all_dset_list is global in this order.
DSETS = [("dset186", "Dset_186"), ("dset164", "PDBset_164"), ("dset72", "Dset_72")]
N_DSSP = 9
N_PSSM = 20


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_raw(data_cache: str | Path):
    """Return per-protein lists in global (dset186, dset164, dset72) order."""
    d = Path(data_cache)
    seqs, pssm, dssp, labels, dset_of = [], [], [], [], []
    for key, name in DSETS:
        s = _load(d / f"{key}_sequence_data.pkl")
        seqs += s
        pssm += _load(d / f"{key}_pssm_data.pkl")
        dssp += _load(d / f"{key}_dssp_data.pkl")
        labels += _load(d / f"{key}_label.pkl")
        dset_of += [name] * len(s)
    return seqs, pssm, dssp, labels, dset_of


def protein_index(data_cache: str | Path) -> pd.DataFrame:
    """One row per protein: global idx, dataset, PDB-chain id, and train/test split."""
    d = Path(data_cache)
    all_list = _load(d / "all_dset_list.pkl")
    train_idx = _load(d / "training_list.pkl")
    test_idx = _load(d / "testing_list.pkl")

    rows = {}
    for r in all_list:
        rows.setdefault(int(r[1]), (str(r[4]), str(r[3]), int(r[5])))
    prot = pd.DataFrame(
        [(i, *rows[i]) for i in sorted(rows)],
        columns=["id_idx", "protein_id", "dset_key", "list_seq_len"],
    )

    split_of = {}
    for name, idxs in (("train", train_idx), ("test", test_idx)):
        for i in idxs:
            pid = int(all_list[i][1])
            if split_of.setdefault(pid, name) != name:
                raise ValueError(f"protein idx {pid} appears in both train and test")
    prot["split"] = prot["id_idx"].map(split_of)
    return prot


def _load_shifts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    t = pd.read_csv(path)
    return {r.protein_id: int(r.applied_shift) for r in t.itertuples()}


def build_residue_table(data_cache: str | Path, shifts_csv: str | Path | None = None):
    """Build (residue_df, protein_df, audit) with DSSP aligned to sequence length."""
    if shifts_csv is None:
        shifts_csv = Path(__file__).resolve().parents[1] / "configs" / "dssp_shifts.csv"
    shifts = _load_shifts(Path(shifts_csv))
    seqs, pssm, dssp, labels, dset_of = load_raw(data_cache)
    prot = protein_index(data_cache)
    assert len(prot) == len(seqs) == 422, (len(prot), len(seqs))

    frames, p_rows = [], []
    for i, row in prot.iterrows():
        k = int(row["id_idx"])
        s = np.asarray(seqs[k], dtype=np.int64)
        n = len(s)
        ps = np.asarray(pssm[k], dtype=np.float32)
        ds = np.asarray(dssp[k], dtype=np.float32)
        lb = np.asarray(labels[k], dtype=np.int8)
        if not (len(ps) == len(lb) == n):
            raise ValueError(f"{row['protein_id']}: seq/pssm/label length mismatch")
        diff = len(ds) - n
        shift = shifts.get(row["protein_id"], 0)
        idx = np.arange(n) + shift
        ok = (idx >= 0) & (idx < len(ds))
        aligned = np.zeros((n, N_DSSP), dtype=np.float32)
        aligned[ok] = ds[idx[ok]]
        missing = (~ok).astype(np.int8)

        df = pd.DataFrame(
            {
                "protein_id": row["protein_id"],
                "dset": dset_of[k],
                "split": row["split"],
                "position": np.arange(n, dtype=np.int32),
                "aa_idx": s.astype(np.int8),
                "aa": [AA[a] for a in s],
                "label": lb,
                "dssp_missing": missing,
            }
        )
        for j in range(N_PSSM):
            df[f"pssm_{j + 1}"] = ps[:, j]
        for j in range(N_DSSP):
            df[f"dssp_{j + 1}"] = aligned[:, j]
        frames.append(df)
        p_rows.append(
            {
                "protein_id": row["protein_id"],
                "dset": dset_of[k],
                "split": row["split"],
                "length": n,
                "n_pos": int(lb.sum()),
                "dssp_len_diff": diff,
                "dssp_shift": shift,
            }
        )

    residues = pd.concat(frames, ignore_index=True)
    proteins = pd.DataFrame(p_rows)
    proteins["pos_rate"] = proteins["n_pos"] / proteins["length"]

    # ---- audit ----
    dcols = [f"dssp_{j + 1}" for j in range(N_DSSP)]
    raw_dssp_rows = np.concatenate([np.asarray(x) for x in dssp])
    audit = {
        "n_proteins": len(proteins),
        "n_residues": len(residues),
        "n_positive": int(residues["label"].sum()),
        "pos_rate_pooled": float(residues["label"].mean()),
        "pos_rate_by_dset": residues.groupby("dset")["label"].mean().round(4).to_dict(),
        "proteins_by_split": proteins["split"].value_counts().to_dict(),
        "residues_by_split": residues["split"].value_counts().to_dict(),
        "pos_rate_by_split": residues.groupby("split")["label"].mean().round(4).to_dict(),
        "dssp_mismatch_proteins": int((proteins["dssp_len_diff"] != 0).sum()),
        "dssp_longer": int((proteins["dssp_len_diff"] > 0).sum()),
        "dssp_shorter": int((proteins["dssp_len_diff"] < 0).sum()),
        "dssp_max_abs_diff": int(proteins["dssp_len_diff"].abs().max()),
        "dssp_shifts_applied": int((proteins["dssp_shift"] != 0).sum()),
        "dssp_padded_residues": int(residues["dssp_missing"].sum()),
        "dssp_row_sum_counts_raw": pd.Series(raw_dssp_rows.sum(1)).value_counts().to_dict(),
        "dssp_col_freq_raw": raw_dssp_rows.sum(0).astype(int).tolist(),
        "length_min_median_max": [
            int(proteins["length"].min()),
            int(proteins["length"].median()),
            int(proteins["length"].max()),
        ],
        "proteins_over_500": int((proteins["length"] > 500).sum()),
        "protein_id_overlap_train_test": len(
            set(proteins.loc[proteins.split == "train", "protein_id"])
            & set(proteins.loc[proteins.split == "test", "protein_id"])
        ),
        "dset_vs_split": pd.crosstab(proteins["dset"], proteins["split"]).to_dict(),
    }
    # round-trip sanity: decoded sequence must re-encode to the stored ints
    assert (residues["aa"].map(AA.index).values == residues["aa_idx"].values).all()
    return residues, proteins, audit


if __name__ == "__main__":
    import json
    import sys

    root = Path(__file__).resolve().parents[1]
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data_cache"
    out = root / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    residues, proteins, audit = build_residue_table(cache)
    residues.to_parquet(out / "residues.parquet", index=False)
    proteins.to_csv(out / "proteins.csv", index=False)
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "data_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    print(json.dumps(audit, indent=2, default=str))
    print(f"\nwrote {out / 'residues.parquet'} ({len(residues):,} rows x {residues.shape[1]} cols)")
