"""Data-loading invariants. Run: python -m pytest tests -q"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_loading import AA, build_residue_table  # noqa: E402


@pytest.fixture(scope="module")
def built():
    return build_residue_table(ROOT / "data_cache")


def test_counts(built):
    r, p, a = built
    assert a["n_proteins"] == 422 and len(r) == 88040 and a["n_positive"] == 13536


def test_split_is_protein_level_and_disjoint(built):
    r, p, a = built
    assert p.split.isin(["train", "test"]).all()
    assert (p.split == "train").sum() == 352 and (p.split == "test").sum() == 70
    assert a["protein_id_overlap_train_test"] == 0
    assert r.groupby("protein_id")["split"].nunique().max() == 1


def test_lengths_and_positions_consistent(built):
    r, p, _ = built
    assert (r.groupby("protein_id").size().reindex(p.protein_id).values == p.length.values).all()
    assert (r.groupby("protein_id")["position"].max().reindex(p.protein_id).values == p.length.values - 1).all()


def test_labels_binary(built):
    assert set(built[0].label.unique()) == {0, 1}


def test_aa_mapping_matches_pssm_blast_order(built):
    """Independent check: native residue is the PSSM row-argmax far above chance (BLAST column order)."""
    r = built[0]
    blast = "ARNDCQEGHILKMFPSTWYV"
    P = r[[f"pssm_{i}" for i in range(1, 21)]].values
    cols = r.aa.map(blast.index).values
    assert (P.argmax(1) == cols).mean() > 0.55          # observed 0.64; wrong mapping ~0.13
    assert P[np.arange(len(P)), cols].mean() > 3.0       # observed 4.6; wrong mapping ~ -0.6
    assert set(r.aa.unique()) <= set(AA)


def test_dssp_is_onehot_or_missing(built):
    r = built[0]
    s = r[[f"dssp_{i}" for i in range(1, 10)]].sum(axis=1)
    assert set(np.unique(s)) <= {0.0, 1.0}


def test_dssp_padding_flag_matches_zero_rows_only_when_padded(built):
    r = built[0]
    d = r[[f"dssp_{i}" for i in range(1, 10)]].sum(axis=1)
    assert (d[r.dssp_missing == 1] == 0).all()
