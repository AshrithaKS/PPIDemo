"""Saved final models and SHAP outputs: files exist, are consistent, and reproduce the reported test numbers."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
M = ROOT / "models"


def _need(*files):
    for f in files:
        if not (ROOT / f).exists():
            pytest.skip(f"{f} missing - run final_evaluation.py / explain_shap.py first")


def test_saved_models_match_their_metadata():
    _need("models/final_model_F.json", "models/final_model_F_meta.json", "models/demo_model_S_meta.json")
    import xgboost as xgb
    for name in ("final_model_F", "demo_model_S"):
        meta = json.loads((M / f"{name}_meta.json").read_text(encoding="utf-8"))
        d, k = meta.get("esm_dim", 480), len(meta.get("context_halves", []))
        expected = {"F": d * (1 + k) + 343 + 51, "S": d * (1 + k) + 22}[meta["feature_set"]]    # window 7x49, global 51
        assert meta["n_features"] == expected == len(meta["feature_names"]) == len(set(meta["feature_names"]))
        assert 0.0 < meta["threshold"] < 1.0
        m = xgb.XGBClassifier()
        m.load_model(str(M / f"{name}.json"))
        assert m.n_features_in_ == expected


def test_final_model_reproduces_reported_test_mcc():
    _need("models/final_model_F.json", "models/final_model_F_meta.json")
    import xgboost as xgb
    from sklearn.metrics import matthews_corrcoef
    from model_utils import load_sets
    meta = json.loads((M / "final_model_F_meta.json").read_text(encoding="utf-8"))
    m = xgb.XGBClassifier()
    m.load_model(str(M / "final_model_F.json"))
    _, sets, y, _, is_train = load_sets(meta.get("esm_size", "35M"), tuple(meta.get("context_halves", ())))
    p = m.predict_proba(sets["F"][~is_train])[:, 1]
    mcc = matthews_corrcoef(y[~is_train], (p >= meta["threshold"]).astype(int))
    assert mcc == pytest.approx(meta["test_metrics_at_threshold"]["mcc"], abs=1e-4)


def test_shap_outputs_are_consistent():
    _need("results/shap_family_importance.json")
    d = json.loads((ROOT / "results" / "shap_family_importance.json").read_text(encoding="utf-8"))
    assert d["additivity_max_error"] < 1e-3                       # SHAP + base value reproduces the model margin
    assert sum(d["family_share_of_total_abs_shap"].values()) == pytest.approx(1.0, abs=1e-6)
    assert sum(d["window_offset_share"].values()) == pytest.approx(1.0, abs=1e-6)


def test_within_protein_ranking_beats_random():
    _need("results/per_protein_ranking.json")
    d = json.loads((ROOT / "results" / "per_protein_ranking.json").read_text(encoding="utf-8"))
    assert d["models"]["F_final"]["mean_within_protein_AP"] > d["mean_base_rate"]
