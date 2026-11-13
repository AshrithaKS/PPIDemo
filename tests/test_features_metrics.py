"""Feature-window and metric invariants. Run: python -m pytest tests -q"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import best_mcc_threshold, metrics_at, trivial_baseline  # noqa: E402
from features import PER_RES, per_residue_matrix, window_features  # noqa: E402


@pytest.fixture(scope="module")
def res():
    return pd.read_parquet(ROOT / "data" / "processed" / "residues.parquet")


def test_window_shape(res):
    assert window_features(res, 3).shape == (len(res), 7 * PER_RES)


def test_centre_block_equals_own_features(res):
    X = window_features(res, 3)
    assert np.array_equal(X[:, 3 * PER_RES : 4 * PER_RES], per_residue_matrix(res))


def test_window_never_crosses_protein_boundaries(res):
    X = window_features(res, 3)
    first = res.index[res.position == 0].to_numpy()          # first residue of each protein
    assert (X[first, : 3 * PER_RES] == 0).all()                # left context is zero padding
    last = res.groupby("protein_id").tail(1).index.to_numpy()
    assert (X[last, 4 * PER_RES :] == 0).all()                 # right context is zero padding


def test_neighbour_block_is_actual_neighbour(res):
    X = window_features(res, 3)
    base = per_residue_matrix(res)
    i = res.index[(res.position == 10)][0]
    assert np.array_equal(X[i, 2 * PER_RES : 3 * PER_RES], base[i - 1])
    assert np.array_equal(X[i, 4 * PER_RES : 5 * PER_RES], base[i + 1])


def test_metrics_and_trivial_baseline():
    y = np.array([0] * 85 + [1] * 15)
    assert trivial_baseline(y)["accuracy"] == pytest.approx(0.85)
    perfect = metrics_at(y, y.astype(float), 0.5)
    assert perfect["mcc"] == pytest.approx(1.0) and perfect["recall"] == 1.0
    assert 0.02 <= best_mcc_threshold(y, y.astype(float)) <= 0.9


def test_global_features_constant_within_protein_except_relpos(res):
    from features import global_features
    G = global_features(res)
    assert G.shape == (len(res), 51)
    df = pd.DataFrame(G[:, :50]).assign(p=res.protein_id.values)
    assert (df.groupby("p").nunique().max().max()) == 1
    rel = pd.Series(G[:, 50]).groupby(res.protein_id.values)
    assert (rel.min() == 0).all() and (rel.max() == 1).all()
    assert np.allclose(G[:, :20].reshape(len(res), 20).sum(1), 1.0, atol=1e-4)


def test_chunk_spans_cover_exactly_and_overlap():
    from embed_esm import chunk_spans
    assert chunk_spans(300) == [(0, 300)]
    for n in (1001, 1500, 2000, 2599):
        sp = chunk_spans(n, 1000, 200)
        assert sp[0][0] == 0 and sp[-1][1] == n
        assert all(e - s <= 1000 for s, e in sp)
        assert all(a[1] - b[0] >= 200 for a, b in zip(sp[:-1], sp[1:]))   # consecutive windows overlap
        covered = np.zeros(n, bool)
        for s, e in sp:
            covered[s:e] = True
        assert covered.all()


def test_esm_embeddings_aligned_and_finite():
    p = ROOT / "data" / "processed" / "esm2_35M.npy"
    if not p.exists():
        pytest.skip("run src/embed_esm.py first")
    E = np.load(p)
    n = len(pd.read_parquet(ROOT / "data" / "processed" / "residues.parquet", columns=["label"]))
    assert E.shape == (n, 480) and np.isfinite(E.astype(np.float32)).all()
