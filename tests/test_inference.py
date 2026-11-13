"""The demo pipeline must build exactly the features the model was trained on."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inference import MAX_LEN, features_S, parse_fasta, validate_sequence  # noqa: E402


def _meta():
    p = ROOT / "models" / "demo_model_S_meta.json"
    if not p.exists():
        pytest.skip("run final_evaluation.py first")
    m = json.loads(p.read_text(encoding="utf-8"))
    return m.get("esm_size", "35M"), int(m.get("esm_dim", 480)), tuple(m.get("context_halves", []))


def test_parse_fasta_variants():
    assert parse_fasta("") == []
    assert parse_fasta("mkt ayi\nakq") == [("sequence_1", "MKTAYIAKQ")]
    recs = parse_fasta(">p1 some description\nMKTA\nYIAK\n>p2\nGGSS\n")
    assert recs == [("p1", "MKTAYIAK"), ("p2", "GGSS")]


def test_validate_sequence():
    ok = "ACDEFGHIKLMNPQRSTVWY" * 2
    assert validate_sequence(ok) == []
    assert any("not one of the 20" in p for p in validate_sequence(ok + "XBZ"))
    assert any("too short" in p for p in validate_sequence("ACDE"))
    assert any("too long" in p for p in validate_sequence("A" * (MAX_LEN + 1)))


@pytest.mark.parametrize("d,halves", [(480, ()), (640, (7,)), (640, (3, 15))])
def test_features_shape_order_and_values(d, halves):
    seq = "ACDEFGHIKLMNPQRSTVWY" * 3
    emb = np.random.default_rng(0).normal(size=(len(seq), d)).astype(np.float32)
    X = features_S(seq, emb, halves)
    assert X.shape == (60, d + 22 + d * len(halves)) and X.dtype == np.float32
    assert np.array_equal(X[:, :d], emb)
    assert np.allclose(X[0, d:d + 20], 0.05)                       # each amino acid is 3/60 of this chain
    assert np.allclose(X[:, d + 20], np.log(60)) and X[0, d + 21] == 0 and X[-1, d + 21] == 1
    for k, h in enumerate(halves):                                  # context block = mean embedding of residues within +-h
        blk = X[:, d + 22 + k * d: d + 22 + (k + 1) * d]
        assert np.allclose(blk[30], emb[30 - h: 31 + h].mean(0), atol=1e-5)
        assert np.allclose(blk[0], emb[: h + 1].mean(0), atol=1e-5)   # edge: only the part of the window inside the chain


@pytest.fixture(scope="module")
def protein():
    size, _, _ = _meta()
    p = ROOT / "data" / "processed" / f"esm2_{size}.npy"
    if not p.exists():
        pytest.skip("run embed_esm.py first")
    res = pd.read_parquet(ROOT / "data" / "processed" / "residues.parquet")
    te = res[res["split"] == "test"]
    pid = te.groupby("protein_id").size().loc[lambda s: s.between(90, 140)].index[0]
    rows = np.flatnonzero((res["protein_id"] == pid).to_numpy())
    return res.iloc[rows], np.load(p, mmap_mode="r")[rows].astype(np.float32)


def test_features_identical_to_training_feature_set_S(protein):
    from model_utils import load_sets
    size, _, halves = _meta()
    sub, emb = protein
    res, sets, *_ = load_sets(size, halves)
    rows = np.flatnonzero((res["protein_id"] == sub["protein_id"].iloc[0]).to_numpy())
    assert np.array_equal(features_S("".join(sub["aa"]), emb, halves), sets["S"][rows])


def test_fresh_esm_embedding_matches_cached_and_prediction_matches(protein):
    from inference import embed, load_demo_model, load_esm
    sub, cached = protein
    seq = "".join(sub["aa"])
    model, meta = load_demo_model()
    tok, esm = load_esm(meta["esm_size"])
    fresh = embed(seq, tok, esm)
    assert np.allclose(fresh, cached, atol=5e-3)                  # same model, same chunking, float16 rounding only
    halves = tuple(meta["context_halves"])
    p_fresh = model.predict_proba(features_S(seq, fresh, halves))[:, 1]
    p_cached = model.predict_proba(features_S(seq, cached, halves))[:, 1]
    assert np.abs(p_fresh - p_cached).max() < 0.02 and 0 <= p_fresh.min() and p_fresh.max() <= 1
