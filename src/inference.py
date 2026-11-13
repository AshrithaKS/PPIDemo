"""Score a NEW protein sequence with the sequence-only demo model (models/demo_model_S.json).

Pipeline for a sequence of length n (the sizes come from the model's metadata, models/demo_model_S_meta.json):
  ESM-2 per-residue embedding (d = 480 for 35M, 640 for 150M)
  + amino-acid composition of the whole chain (20) + log(n) + relative position
  + optional wider context: mean embedding of the residues within +-h, one block of d per h in meta["context_halves"]
identical in construction and order to feature set "S" used in training (model_utils.load_sets).
Embeddings are rounded through float16 first, because the training embeddings were cached as float16.
No PSSM / DSSP is needed, which is why this model (not the fused model) is the one the app can use.
PyTorch / transformers are imported lazily, so code that only needs parsing or the XGBoost model starts fast.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from data_loading import AA
from features import window_mean

root = Path(__file__).resolve().parents[1]
MAX_LEN = 1500          # CPU time guard for the demo (training proteins go up to 2000)
MIN_LEN = 20
BASE_FAMILIES = ["ESM-2 embedding", "Amino-acid composition", "Chain length", "Position in chain"]
CONTEXT_FAMILY = "Neighbouring residues (ESM-2)"
FAMILIES = BASE_FAMILIES            # kept for older callers; use families(meta) for the current model


def families(meta: dict) -> list[str]:
    return BASE_FAMILIES + ([CONTEXT_FAMILY] if meta.get("context_halves") else [])


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Parse FASTA text (one or more records) or a bare sequence. Returns [(name, sequence)] with whitespace removed, upper-cased."""
    text = text.strip()
    if not text:
        return []
    if not text.startswith(">"):
        return [("sequence_1", re.sub(r"\s+", "", text).upper())]
    out = []
    for i, block in enumerate(re.split(r"(?m)^>", text)[1:], start=1):
        head, _, body = block.partition("\n")
        out.append((head.strip().split()[0] if head.strip() else f"sequence_{i}", re.sub(r"\s+", "", body).upper()))
    return out


def validate_sequence(seq: str) -> list[str]:
    """Human-readable problems with a sequence (empty list = OK)."""
    problems = []
    bad = sorted(set(seq) - set(AA))
    if bad:
        problems.append(f"contains letters that are not one of the 20 standard amino acids: {', '.join(bad)}")
    if len(seq) < MIN_LEN:
        problems.append(f"too short ({len(seq)} residues; minimum {MIN_LEN})")
    if len(seq) > MAX_LEN:
        problems.append(f"too long ({len(seq)} residues; this demo accepts at most {MAX_LEN})")
    return problems


def features_S(seq: str, emb: np.ndarray, context_halves=()) -> np.ndarray:
    """(n, d + 22 + d*len(context_halves)) float32 features for feature set S from the sequence and its (n, d) ESM-2 embedding."""
    n, d = len(seq), emb.shape[1]
    assert emb.shape[0] == n, (emb.shape, n)
    comp = np.array([seq.count(a) for a in AA], dtype=np.float32) / n
    X = np.empty((n, d + 22 + d * len(context_halves)), dtype=np.float32)
    X[:, :d] = emb
    X[:, d:d + 20] = comp
    X[:, d + 20] = np.log(n)
    X[:, d + 21] = np.arange(n) / max(n - 1, 1)
    g = np.zeros(n, dtype=np.int8)
    for k, h in enumerate(context_halves):
        X[:, d + 22 + k * d: d + 22 + (k + 1) * d] = window_mean(emb, g, h)
    return X


def load_esm(name: str = "35M"):
    import torch
    from transformers import AutoTokenizer, EsmModel

    from embed_esm import MODELS
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(MODELS[name])
    model = EsmModel.from_pretrained(MODELS[name], add_pooling_layer=False).eval()
    return tok, model


def load_demo_model():
    import xgboost as xgb
    meta = json.loads((root / "models" / "demo_model_S_meta.json").read_text(encoding="utf-8"))
    meta.setdefault("esm_size", "35M")
    meta.setdefault("esm_dim", 480)
    meta.setdefault("context_halves", [])
    m = xgb.XGBClassifier()
    m.load_model(str(root / "models" / "demo_model_S.json"))
    assert m.n_features_in_ == meta["n_features"] == meta["esm_dim"] * (1 + len(meta["context_halves"])) + 22
    return m, meta


def embed(seq: str, tok, esm) -> np.ndarray:
    from embed_esm import embed_sequence
    return embed_sequence(seq, tok, esm).astype(np.float16).astype(np.float32)


def predict_residues(seq: str, tok, esm, model, meta: dict | None = None) -> np.ndarray:
    """Per-residue probability of being a protein-protein interface residue."""
    halves = tuple((meta or {}).get("context_halves", ()))
    return model.predict_proba(features_S(seq, embed(seq, tok, esm), halves))[:, 1]


def family_contributions(model, X: np.ndarray, meta: dict | None = None) -> tuple[np.ndarray, float]:
    """Exact per-residue explanation of the model output, grouped into feature families (see families(meta)).

    Returns (contrib, bias): contrib is (n, n_families) in log-odds; probability = sigmoid(bias + contrib.sum(axis=1)).
    Uses XGBoost's built-in tree SHAP (pred_contribs), so no extra package is needed.
    """
    import xgboost as xgb
    d = int((meta or {}).get("esm_dim", 480))
    booster = model.get_booster()
    try:                                  # use exactly the trees predict_proba uses (early stopping keeps extra trailing trees)
        end = int(booster.attr("best_iteration")) + 1
    except (TypeError, ValueError):
        end = 0
    c = booster.predict(xgb.DMatrix(X), pred_contribs=True, iteration_range=(0, end))   # (n, n_features + 1): features + bias
    cols = [c[:, :d].sum(1), c[:, d:d + 20].sum(1), c[:, d + 20], c[:, d + 21]]
    if X.shape[1] > d + 22:
        cols.append(c[:, d + 22:-1].sum(1))
    return np.stack(cols, axis=1), float(c[0, -1])
