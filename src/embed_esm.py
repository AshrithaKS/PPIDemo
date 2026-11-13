"""Per-residue ESM-2 embeddings for all 422 proteins (CPU friendly).

Writes data/processed/esm2_<name>.npy  -> float16 array (n_residues, hidden), row-aligned with
residues.parquet, plus esm2_<name>.json (model, dim, chunking, sanity stats).

Proteins longer than the model window are processed in overlapping chunks and the overlaps are
averaged (ESM-2 supports at most 1022 residues + 2 special tokens).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel

root = Path(__file__).resolve().parents[1]
MODELS = {
    "35M": "facebook/esm2_t12_35M_UR50D",     # 480-dim, 12 layers  (default, CPU friendly)
    "150M": "facebook/esm2_t30_150M_UR50D",   # 640-dim
    "650M": "facebook/esm2_t33_650M_UR50D",   # 1280-dim (GPU / Colab recommended)
}


def chunk_spans(n: int, max_len: int = 1000, overlap: int = 200) -> list[tuple[int, int]]:
    """Cover [0, n) with windows of <= max_len, consecutive windows overlapping by `overlap`."""
    if n <= max_len:
        return [(0, n)]
    step = max_len - overlap
    spans, s = [], 0
    while True:
        e = min(s + max_len, n)
        spans.append((s, e))
        if e == n:
            return spans
        s += step


@torch.no_grad()
def embed_sequence(seq: str, tok, model, max_len: int = 1000, overlap: int = 200) -> np.ndarray:
    n = len(seq)
    acc = np.zeros((n, model.config.hidden_size), dtype=np.float32)
    cnt = np.zeros(n, dtype=np.float32)
    for s, e in chunk_spans(n, max_len, overlap):
        enc = tok(seq[s:e], return_tensors="pt", add_special_tokens=True)
        h = model(**enc).last_hidden_state[0, 1:-1].float().numpy()   # drop <cls>, <eos>
        assert h.shape[0] == e - s, (h.shape, e - s)
        acc[s:e] += h
        cnt[s:e] += 1
    return acc / cnt[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="35M", choices=list(MODELS))
    ap.add_argument("--limit", type=int, default=0, help="only first N proteins (smoke test)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    res = pd.read_parquet(root / "data" / "processed" / "residues.parquet", columns=["protein_id", "position", "aa"])
    tok = AutoTokenizer.from_pretrained(MODELS[a.model])
    model = EsmModel.from_pretrained(MODELS[a.model], add_pooling_layer=False).eval()
    dim = model.config.hidden_size

    pids = list(dict.fromkeys(res["protein_id"]))
    if a.limit:
        pids = pids[: a.limit]
    sub = res[res["protein_id"].isin(set(pids))]
    out = np.zeros((len(sub), dim), dtype=np.float16)
    row = 0
    t0 = time.time()
    for i, (pid, g) in enumerate(sub.groupby("protein_id", sort=False)):
        assert (g["position"].to_numpy() == np.arange(len(g))).all()
        seq = "".join(g["aa"])
        emb = embed_sequence(seq, tok, model)
        assert np.isfinite(emb).all()
        out[row : row + len(g)] = emb.astype(np.float16)
        row += len(g)
        if (i + 1) % 25 == 0 or i + 1 == len(pids):
            print(f"{i+1}/{len(pids)} proteins  {time.time()-t0:.0f}s", flush=True)

    tag = f"esm2_{a.model}" + ("_smoke" if a.limit else "")
    d = root / "data" / "processed"
    np.save(d / f"{tag}.npy", out)
    stats = {"model": MODELS[a.model], "dim": dim, "n_residues": int(len(sub)), "n_proteins": len(pids),
             "dtype": "float16", "chunk": {"max_len": 1000, "overlap": 200},
             "mean": float(out.astype(np.float32).mean()), "std": float(out.astype(np.float32).std()),
             "seconds": round(time.time() - t0, 1)}
    (d / f"{tag}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
