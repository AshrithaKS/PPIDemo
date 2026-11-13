"""Click through the Streamlit app headlessly (no browser): every step and path must render without an exception."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
STEP1, STEP2, STEP3 = "1  Choose a protein", "2  See the result", "3  Why?"
OWN = "My own sequence"


def _need():
    for f in ("results/demo_test_probabilities.parquet", "results/demo_threshold_curve.csv", "results/demo_family_importance.json",
              "models/demo_model_S.json"):
        if not (ROOT / f).exists():
            pytest.skip(f"{f} missing - run src/demo_assets.py first")


def _fresh():
    _need()
    return AppTest.from_file(APP, default_timeout=180).run()


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _result_text(at):
    return " ".join(m.value for m in at.markdown if "is-result" in m.value)


def _expected_3e1z_B():
    """What the headline must say for 3e1z_B with the saved demo model and its default threshold (computed, not hard-coded)."""
    import json
    import pandas as pd
    thr = json.loads((ROOT / "models" / "demo_model_S_meta.json").read_text(encoding="utf-8"))["threshold"]
    d = pd.read_parquet(ROOT / "results" / "demo_test_probabilities.parquet").query("protein_id == '3e1z_B'")
    flagged = d["prob"].to_numpy() >= thr
    return thr, int(flagged.sum()), int((flagged & (d["label"].to_numpy() == 1)).sum()), int(d["label"].sum())


def test_starts_on_step_1_and_the_test_protein_result_renders():
    at = _fresh()
    _ok(at)
    assert at.session_state["step"] == STEP1
    assert at.selectbox(key="pid").value == "3e1z_B"
    at.button(key="go_test").click().run()
    _ok(at)
    assert at.session_state["step"] == STEP2
    thr, n_flag, tp, n_true = _expected_3e1z_B()
    txt = _result_text(at)
    assert "3e1z_B" in txt and "out of 212 residues" in txt and f"found {tp} of {n_true}" in txt and f">{n_flag}<" in txt
    assert at.slider(key="thr").value == pytest.approx(thr)


def test_choices_survive_moving_between_steps():
    """Regression: Streamlit drops the state of widgets that are not drawn, so the chosen protein used to reset on step 2."""
    at = _fresh()
    at.selectbox(key="pid").select("2d5r_A").run()
    at.button(key="go_test").click().run()
    _ok(at)
    assert "2d5r_A" in _result_text(at)
    at.slider(key="thr").set_value(0.35).run()
    _ok(at)
    at.button(key="next_2").click().run()                     # step 3
    _ok(at)
    assert at.session_state["step"] == STEP3
    at.button(key="back_2").click().run()                     # back to step 2: protein and threshold kept
    _ok(at)
    assert "2d5r_A" in _result_text(at)
    assert at.slider(key="thr").value == pytest.approx(0.35)


def test_why_step_explains_a_held_out_residue():
    at = _fresh()
    at.session_state["step"] = STEP3
    at.run()
    _ok(at)
    assert "Residue" in _result_text(at)
    at.slider(key="why_3e1z_B").set_value(20).run()
    _ok(at)
    assert "Residue 20 " in _result_text(at)


def test_example_button_fills_the_box():
    at = _fresh()
    at.session_state["source"] = OWN
    at.run()
    _ok(at)
    at.button(key="use_example").click().run()
    _ok(at)                                                    # regression: this button used to raise StreamlitAPIException
    assert at.text_area(key="seq_text").value.startswith(">ubiquitin")


def test_invalid_and_empty_input_give_clear_errors_and_stay_on_step_1():
    at = _fresh()
    at.session_state["source"] = OWN
    at.run()
    at.button(key="predict").click().run()
    assert any("Paste a sequence first" in e.value for e in at.error)
    at.text_area(key="seq_text").set_value("MKTAYIAKQRQISFVKSHFSRQXXZ").run()
    at.button(key="predict").click().run()
    assert any("not one of the 20 standard amino acids" in e.value and "X, Z" in e.value for e in at.error)
    assert at.session_state["step"] == STEP1


def test_pasting_ubiquitin_scores_end_to_end():
    """Loads the real ESM-2 model (slow, ~30 s): parsing -> embedding -> model -> result -> explanation."""
    at = _fresh()
    at.session_state["source"] = OWN
    at.run()
    at.button(key="use_example").click().run()
    at.button(key="predict").click().run()
    _ok(at)
    assert at.session_state["step"] == STEP2
    txt = _result_text(at)
    assert "ubiquitin" in txt and "out of 76 residues" in txt
    at.button(key="next_2").click().run()
    _ok(at)
    assert at.slider(key="why_ubiquitin").value >= 1
