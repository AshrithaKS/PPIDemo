"""BindSpot: predict protein-protein interface residues from a sequence.   Run:  python -m streamlit run app/streamlit_app.py

A guided, three-step page:  1 choose a protein  ->  2 see the result (the answer first)  ->  3 why (per-residue explanation).
Sequence-only demo model (ESM-2 embeddings [+ neighbourhood means] + amino-acid composition + chain length + relative position, XGBoost);
the language-model size and context windows are read from models/demo_model_S_meta.json.
Held-out test performance (70 proteins): MCC 0.24, PR-AUC 0.38 -> a research prototype, not a validated predictor.

Design (ui-ux-pro-max "Biotech / Life Sciences" palette, Poppins + Open Sans): sky blue = what the model says, orange = what
experiments say. The three data colours pass the dataviz colour-vision checks in light and dark mode.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score

from inference import (MAX_LEN, families, family_contributions, features_S, load_demo_model, load_esm, parse_fasta,
                       validate_sequence)

st.set_page_config(page_title="BindSpot", page_icon=None, layout="wide")

# ----------------------------------------------------------------------------------------------------------- design tokens
# Every text token is >= 4.5:1 on its surface. Data colours (model, truth, third) are identical in both modes and validated.
LIGHT = dict(page="#f0f9ff", surface="#ffffff", soft="#e6f4fd", ink="#0c3b5c", ink2="#41566b", muted="#55687d", grid="#e2eef7",
             axis="#bcd3e6", border="#cfe3f3", brand="#0369a1", on_brand="#ffffff", accent="#047857",
             model="#0284c7", model_hi="#075985", truth="#ea580c", third="#059669", pos="#dc2626", neg="#0284c7",
             ramp=("#e0f2fe", "#075985"), note_bg="#fff7e6", note_ink="#6b4300", focus="#0369a1")
DARK = dict(page="#061523", surface="#0b2235", soft="#0f2b42", ink="#e0f2fe", ink2="#b3c8da", muted="#93a9bd", grid="#16334b",
            axis="#24496a", border="#1d3d59", brand="#38bdf8", on_brand="#04121f", accent="#34d399",
            model="#0284c7", model_hi="#7dd3fc", truth="#ea580c", third="#059669", pos="#f87171", neg="#38bdf8",
            ramp=("#0f3350", "#7dd3fc"), note_bg="#2a2412", note_ink="#f1d98a", focus="#7dd3fc")
try:
    MODE = st.context.theme.type or "light"
except Exception:
    MODE = "light"
T = DARK if MODE == "dark" else LIGHT
PMAX = 0.6                     # probability that maps to the strongest tile colour (fixed, so proteins are comparable)
# CSS colours resolve in the browser with light-dark(), keyed to .stApp's live color-scheme, so the page follows a theme switch
# instantly (Streamlit does not rerun the script on a theme change; the hook further down reruns once for the charts).
V = {k: f"var(--is-{k.replace('_', '-')})" for k, v in LIGHT.items() if isinstance(v, str)}
CSS_VARS = " ".join(f"--is-{k.replace('_', '-')}: light-dark({LIGHT[k]}, {DARK[k]});" for k in V)
STEPS = ["1  Choose a protein", "2  See the result", "3  Why?"]


def mix(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(round(x + (y - x) * min(max(t, 0), 1))):02x}" for x, y in zip(a, b))


def ramp(p: float) -> str:
    return mix(T["ramp"][0], T["ramp"][1], p / PMAX)


def ink_on(bg: str) -> str:
    """Dark or white text, whichever reads better on this tile colour."""
    r, g, b = (int(bg[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#04121f" if lum > 0.42 else "#ffffff"


st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Open+Sans:wght@400;500;600;700&display=swap');
.stApp {{ {CSS_VARS} }}
html, body, [class*="st-"], .stApp {{ font-family: "Open Sans", system-ui, "Segoe UI", sans-serif; }}
[data-testid="stIconMaterial"], [data-testid="stIconMaterial"] * {{ font-family: "Material Symbols Rounded" !important; }}
.stApp h1, .stApp h2, .stApp h3, .is-big, .is-brand b {{ font-family: "Poppins", "Open Sans", system-ui, sans-serif; }}
.block-container {{ padding-top: 4rem; max-width: 1100px; }}   /* clears Streamlit's floating toolbar (Deploy / menu), which overlaps anything higher */

/* header */
.is-top {{ display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap; margin-bottom:6px; }}
.is-brand {{ display:flex; align-items:center; gap:14px; }}
.is-brand b {{ display:block; font-size:1.75rem; font-weight:700; letter-spacing:-0.02em; color:{V['ink']}; line-height:1.25; }}
.is-brand span {{ color:{V['ink2']}; font-size:0.98rem; }}
.is-pill {{ display:inline-block; border:1px solid {V['border']}; background:{V['surface']}; color:{V['ink2']}; border-radius:999px;
  padding:5px 12px; font-size:0.8rem; font-weight:600; }}

/* stepper: the "step" segmented control (inside st.container(key="stepper")) restyled as three big, equal steps */
.st-key-stepper [data-testid="stElementContainer"], .st-key-stepper [data-testid="stButtonGroup"],
.st-key-stepper [data-testid="stButtonGroup"] > div {{ width:100% !important; }}
.st-key-stepper [data-testid="stButtonGroup"] > div {{ display:grid !important; grid-template-columns:repeat(3, 1fr); gap:10px; }}
.st-key-stepper button {{ min-height:52px; width:100%; border-radius:12px !important; font-family:"Poppins", "Open Sans", sans-serif; font-weight:600;
  font-size:0.98rem; border:1.5px solid {V['border']} !important; background:{V['surface']}; color:{V['ink2']}; }}
.st-key-stepper button[kind*="Active"] {{ background:{V['brand']} !important; color:{V['on_brand']} !important; border-color:{V['brand']} !important;
  box-shadow:0 8px 20px -10px {V['brand']}; }}

/* cards */
.is-card {{ background:{V['surface']}; border:1px solid {V['border']}; border-radius:16px; padding:20px 22px; }}
.is-result {{ display:flex; gap:22px; align-items:center; flex-wrap:wrap; background:linear-gradient(135deg, {V['soft']}, {V['surface']} 70%);
  border:1px solid {V['border']}; border-radius:18px; padding:20px 24px; margin:4px 0 12px 0; }}
.is-big {{ font-size:3.1rem; font-weight:700; color:{V['brand']}; line-height:1; }}
.is-result .t {{ font-size:1.15rem; font-weight:600; color:{V['ink']}; }}
.is-result .s {{ color:{V['ink2']}; font-size:0.92rem; margin-top:4px; }}
.is-chip {{ display:inline-flex; align-items:center; gap:6px; background:{V['surface']}; border:1px solid {V['border']}; border-radius:999px;
  padding:4px 11px; font-size:0.84rem; color:{V['ink']}; margin:6px 6px 0 0; }}
.is-chip i {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}
.is-note {{ background:{V['note_bg']}; color:{V['note_ink']}; border-radius:10px; padding:8px 12px; font-size:0.85rem; margin:8px 0 2px 0; }}
.is-empty {{ text-align:center; padding:34px 20px; }}
.is-empty b {{ font-family:"Poppins", sans-serif; font-size:1.2rem; color:{V['ink']}; display:block; margin-bottom:6px; }}
.is-empty span {{ color:{V['ink2']}; }}

/* residue tiles: the main answer */
.is-legend {{ display:flex; flex-wrap:wrap; gap:6px 18px; font-size:0.84rem; color:{V['ink2']}; margin:2px 0 10px 0; align-items:center; }}
.is-legend i {{ display:inline-block; vertical-align:middle; margin-right:6px; border-radius:4px; }}
.is-tiles {{ display:flex; flex-wrap:wrap; gap:12px 14px; }}
.is-chunk small {{ display:block; color:{V['muted']}; font-size:0.7rem; margin:0 0 2px 2px; font-variant-numeric:tabular-nums; }}
.is-chunk div {{ display:flex; gap:2px; }}
.is-t {{ width:22px; height:28px; display:inline-flex; align-items:center; justify-content:center; border-radius:5px;
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size:13px; border-bottom:4px solid transparent; }}
.is-t.f {{ font-weight:800; outline:2px solid {V['ink']}; outline-offset:-2px; }}
.is-t.k {{ border-bottom-color:{V['truth']}; }}

/* controls: friendly targets, visible focus, gentle motion */
button, [role="tab"], [data-baseweb="select"], [role="radio"], label {{ cursor:pointer; }}
.stButton > button, .stDownloadButton > button {{ min-height:44px; border-radius:10px; font-weight:600; transition:transform 120ms ease, box-shadow 150ms ease; }}
.stButton > button[kind="primary"] {{ background:{V['brand']}; border-color:{V['brand']}; color:{V['on_brand']}; }}
.stButton > button[kind="primary"]:hover {{ transform:translateY(-1px); box-shadow:0 8px 20px -10px {V['brand']}; }}
button:focus-visible, [role="tab"]:focus-visible, [role="radio"]:focus-visible, textarea:focus-visible {{ outline:3px solid {V['focus']} !important; outline-offset:2px; }}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ transition:none !important; animation:none !important; }} }}
.st-key-theme_hook {{ position:absolute; width:1px; height:1px; overflow:hidden; opacity:0; pointer-events:none; }}
.is-result > div:last-child {{ flex:1 1 320px; min-width:0; }}
</style>""", unsafe_allow_html=True)

# Theme switch -> rerun once so the Altair charts (coloured in Python) redraw. Uses Streamlit's own "R" rerun key, which keeps the
# current step, protein, threshold and pasted sequence. Does nothing if the parent page is not reachable.
with st.container(key="theme_hook"):
    st.iframe("""<script>
try {
  const doc = window.parent.document, app = () => doc.querySelector('.stApp');
  const scheme = () => app() ? getComputedStyle(app()).colorScheme : '';
  let last = scheme();
  setInterval(() => { const now = scheme();
    if (now && now !== last) { last = now;
      doc.dispatchEvent(new KeyboardEvent('keydown', {key: 'r', code: 'KeyR', keyCode: 82, which: 82, bubbles: true})); } }, 400);
} catch (e) {}
</script>""", height=1)


# ----------------------------------------------------------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading the ESM-2 protein language model (first sequence only, about 30 s)...")
def get_esm():
    return load_esm(get_model()[1]["esm_size"])


@st.cache_resource(show_spinner=False)
def get_model():
    return load_demo_model()


@st.cache_data(show_spinner=False)
def score_pasted(seq: str) -> dict:
    tok, esm = get_esm()
    from inference import embed
    model, meta = get_model()
    X = features_S(seq, embed(seq, tok, esm), tuple(meta["context_halves"]))
    fam, bias = family_contributions(model, X, meta)
    return {"p": model.predict_proba(X)[:, 1], "fam": fam, "bias": bias}


@st.cache_data(show_spinner=False)
def held_out() -> dict:
    """Held-out TEST proteins (never used for training, early stopping, thresholds or tuning): sequence, labels, probabilities."""
    df = pd.read_parquet(ROOT / "results" / "demo_test_probabilities.parquet")
    return {pid: {"seq": "".join(g["aa"]), "labels": g["label"].to_numpy(), "p": g["prob"].to_numpy(dtype=float)}
            for pid, g in df.groupby("protein_id", sort=True)}


@st.cache_data(show_spinner=False)
def contrib_held_out(pid: str) -> dict:
    ids = pd.read_parquet(ROOT / "data" / "processed" / "residues.parquet", columns=["protein_id"])["protein_id"].to_numpy()
    rows = np.flatnonzero(ids == pid)
    meta = get_model()[1]
    emb = np.load(ROOT / "data" / "processed" / f"esm2_{meta['esm_size']}.npy", mmap_mode="r")[rows].astype(np.float32)
    model, _ = get_model()
    X = features_S(held_out()[pid]["seq"], emb, tuple(meta["context_halves"]))
    fam, bias = family_contributions(model, X, meta)
    return {"fam": fam, "bias": bias}


@st.cache_data(show_spinner=False)
def threshold_curve() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "demo_threshold_curve.csv")


@st.cache_data(show_spinner=False)
def family_importance() -> dict:
    return json.loads((ROOT / "results" / "demo_family_importance.json").read_text(encoding="utf-8"))


_, META = get_model()
DEFAULT_THR = float(META["threshold"])
FAMILIES = families(META)
ESM_DIM, ESM_SIZE, N_CTX = int(META["esm_dim"]), META["esm_size"], len(META["context_halves"])
HELD = held_out()
UBIQUITIN = ">ubiquitin\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
SRC_TEST, SRC_OWN = "Test protein (answer known)", "My own sequence"

ss = st.session_state
# Streamlit forgets a widget's value on any run where that widget is not drawn (e.g. the protein picker while on step 2).
# Re-assigning the keys at the top of every run keeps the choices across steps. Widgets therefore get their start values
# here, never through default=/index=/value= (which would clash with values set through session state).
ss.setdefault("step", STEPS[0])
ss.setdefault("source", SRC_TEST)
ss.setdefault("pid", "3e1z_B" if "3e1z_B" in HELD else next(iter(HELD)))
ss.setdefault("seq_text", "")
ss.setdefault("thr", DEFAULT_THR)
for _k in [k for k in ss.keys() if k in ("step", "source", "pid", "seq_text", "thr") or str(k).startswith("why_")]:
    ss[_k] = ss[_k]


def go(step: int):
    ss["step"] = STEPS[step]


def keep_step():
    """Clicking the active step again would deselect it; stay where we are instead."""
    if ss.get("step") is None:
        ss["step"] = ss.get("_last_step", STEPS[0])


def on_predict():
    """Validate pasted text before leaving step 1, so errors are shown right next to the box."""
    recs = parse_fasta(ss.get("seq_text", ""))
    ss["pasted_n"] = len(recs)
    ss["pasted"] = recs[:5]
    ss["paste_errors"] = [f"{nm}: " + "; ".join(pr) for nm, sq in recs[:5] if (pr := validate_sequence(sq))]
    if not recs:
        ss["paste_errors"] = ["Paste a sequence first (or press Use example)."]
    if recs and len(ss["paste_errors"]) < len(recs[:5]):
        go(1)


# ------------------------------------------------------------------------------------------------------------ charts
def style(chart: alt.Chart) -> alt.Chart:
    return (chart.configure(background=T["surface"], font="Open Sans, system-ui, Segoe UI, sans-serif")
            .configure_axis(labelColor=T["ink2"], titleColor=T["ink2"], gridColor=T["grid"], domainColor=T["axis"],
                            tickColor=T["axis"], labelFontSize=11, titleFontSize=11, titleFontWeight="normal")
            .configure_view(stroke=None)
            .configure_legend(labelColor=T["ink2"], titleColor=T["ink2"], labelFontSize=11, titleFontSize=11, orient="top"))


def residue_df(seq, p, labels, thr):
    df = pd.DataFrame({"position": np.arange(1, len(seq) + 1), "residue": list(seq), "probability": p})
    df["flagged"] = df["probability"] >= thr
    df["x0"], df["x1"] = df["position"] - 0.5, df["position"] + 0.5
    df["label"] = df["position"].astype(str) + " " + df["residue"]
    df["known"] = (labels == 1) if labels is not None else False
    return df


def profile_chart(df, thr, has_labels, ymax):
    """Probability along the chain + flagged points (+ the known interface lane). Hover shows a residue; drag selects a region."""
    x0, x1 = float(df["position"].min()) - 0.5, float(df["position"].max()) + 0.5
    ylo = -0.11 if has_labels else -0.02
    xscale, yscale = alt.Scale(domain=[x0, x1], nice=False), alt.Scale(domain=[ylo, ymax], nice=False)
    xs = alt.X("position:Q", scale=xscale, axis=alt.Axis(title="Residue position", grid=False, tickMinStep=1))
    ys = alt.Y("probability:Q", scale=yscale, axis=alt.Axis(title="Interface probability", values=[v for v in (0, 0.2, 0.4, 0.6, 0.8) if v <= ymax], format=".1f"))
    tip = [alt.Tooltip("label:N", title="Residue"), alt.Tooltip("probability:Q", title="Probability", format=".2f")]
    if has_labels:
        tip.append(alt.Tooltip("known:N", title="Known interface"))
    hover = alt.selection_point(name="hover", nearest=True, on="mouseover", fields=["position"], empty=False, clear="mouseout")
    brush = alt.selection_interval(name="brush", encodings=["x"], mark=alt.BrushConfig(fill=T["model"], fillOpacity=0.10, stroke=T["model"]))
    base = alt.Chart(df)
    layers = [base.mark_area(color=T["model"], opacity=0.15, clip=True).encode(x=xs, y=ys, y2=alt.datum(0)),
              base.mark_line(color=T["model"], strokeWidth=2, clip=True).encode(x=xs, y=ys),
              alt.Chart(pd.DataFrame({"y": [thr]})).mark_rule(color=T["muted"], strokeDash=[5, 4]).encode(y=alt.Y("y:Q", scale=yscale))]
    if len(df) <= 450:
        layers.append(base.transform_filter(alt.datum.flagged == True).mark_point(                       # noqa: E712
            filled=True, size=58, color=T["model_hi"], stroke=T["surface"], strokeWidth=1.5).encode(x=xs, y=ys))
    if has_labels:
        layers.append(base.transform_filter(alt.datum.known == True).mark_rect(color=T["truth"], clip=True).encode(      # noqa: E712
            x=alt.X("x0:Q", scale=xscale), x2="x1:Q", y=alt.Y("lo:Q", scale=yscale), y2="hi:Q",
            tooltip=[alt.Tooltip("label:N", title="Known interface residue")]).transform_calculate(lo="-0.095", hi="-0.04"))
    layers.append(base.mark_point(opacity=0, size=260).encode(x=xs, y=ys, tooltip=tip).add_params(hover))
    layers.append(base.mark_rule(color=T["muted"]).encode(x=xs, opacity=alt.condition(hover, alt.value(0.7), alt.value(0))))
    layers.append(base.transform_filter(hover).mark_point(filled=True, size=90, color=T["model"], stroke=T["surface"], strokeWidth=2).encode(x=xs, y=ys))
    return style(alt.layer(*layers).add_params(brush).properties(height=250))


def tiles_html(seq, p, thr, known=None, start=1):
    """The sequence as coloured tiles in groups of ten: darker = more likely interface, outlined = flagged, orange underline = known."""
    out, chunk = [], []
    for i, (a, v) in enumerate(zip(seq, p)):
        bg = ramp(v)
        cls = "is-t" + (" f" if v >= thr else "") + (" k" if (known is not None and known[i]) else "")
        chunk.append(f'<span class="{cls}" title="{start + i} {a}: probability {v:.2f}" style="background:{bg};color:{ink_on(bg)}">{a}</span>')
        if len(chunk) == 10 or i == len(seq) - 1:
            out.append(f'<div class="is-chunk"><small>{start + i - len(chunk) + 1}</small><div>{"".join(chunk)}</div></div>')
            chunk = []
    return '<div class="is-tiles">' + "".join(out) + "</div>"


def legend_html(has_labels):
    items = [f'<span><i style="width:46px;height:12px;background:linear-gradient(90deg,{T["ramp"][0]},{T["ramp"][1]})"></i>less &rarr; more likely</span>',
             f'<span><i style="width:14px;height:14px;outline:2px solid {T["ink"]};outline-offset:-2px"></i>flagged (above your threshold)</span>']
    if has_labels:
        items.append(f'<span><i style="width:14px;height:5px;background:{T["truth"]};border-radius:2px"></i>real interface (from experiments)</span>')
    return '<div class="is-legend">' + "".join(items) + "</div>"


def logo_svg():
    cols = [ramp(v) for v in (0.05, 0.15, 0.5, 0.6, 0.35, 0.1)]
    rects = "".join(f'<rect x="{i * 9}" y="{8 if i in (2, 3) else 12}" width="7" height="{24 if i in (2, 3) else 20}" rx="2" fill="{c}"/>' for i, c in enumerate(cols))
    return (f'<svg width="54" height="42" viewBox="0 0 54 42" aria-hidden="true">{rects}'
            f'<rect x="18" y="36" width="16" height="4" rx="2" fill="{T["truth"]}"/></svg>')


# ---------------------------------------------------------------------------------------------------------------- header
st.markdown(f'<div class="is-top"><div class="is-brand">{logo_svg()}<div><b>BindSpot</b>'
            f'<span>Find the residues of a protein that are likely to touch another protein.</span></div></div>'
            f'<span class="is-pill" title="About {META["test_metrics_at_threshold"]["precision"] * 100:.0f}% of flagged residues are real interface residues (70 held-out proteins). Not for clinical or lab decisions.">'
            f'Research prototype</span></div>', unsafe_allow_html=True)

with st.container(key="stepper"):
    step_label = st.segmented_control("Step", STEPS, key="step", on_change=keep_step, label_visibility="collapsed",
                                      width="stretch") or STEPS[0]
step = STEPS.index(step_label)
ss["_last_step"] = step_label

# --------------------------------------------------------------------------------------------- which protein(s) are active
source = ss.get("source") or SRC_TEST
records = []
if source == SRC_TEST:
    d = HELD[ss["pid"]]
    pid = ss["pid"]
    records = [{"name": pid, "seq": d["seq"], "labels": d["labels"], "p": d["p"], "kind": "held"}]

# =========================================================================================================== STEP 1
if step == 0:
    st.markdown("### Which protein should we scan?")
    st.segmented_control("Source", [SRC_TEST, SRC_OWN], key="source", label_visibility="collapsed")
    source = ss.get("source") or SRC_TEST
    with st.container(border=True):
        if source == SRC_TEST:
            st.selectbox("Test protein", list(HELD), key="pid",
                         format_func=lambda k: f"{k}  ·  {len(HELD[k]['seq'])} residues  ·  {int(HELD[k]['labels'].sum())} real interface sites")
            st.caption("70 proteins the model never saw while learning. You will see the real answer next to the prediction. "
                       "Tip: 3e1z_B is predicted well, 2d5r_A poorly.")
            st.button("See the result  ▶", type="primary", on_click=go, args=(1,), key="go_test")
        else:
            st.text_area("Paste a protein sequence (plain letters or FASTA, up to 5 sequences)", height=140, key="seq_text",
                         placeholder=f">my_protein\nMKTAYIAKQRQISFVKSHFSRQ...   (20 to {MAX_LEN} residues, standard amino acids)")
            c1, c2, _ = st.columns([1.1, 1, 2])
            c1.button("Use example (ubiquitin)", on_click=lambda: ss.update(seq_text=UBIQUITIN), key="use_example",
                      help="A 76-residue protein that is not in the benchmark")
            c2.button("Predict  ▶", type="primary", on_click=on_predict, key="predict")
            for e in ss.get("paste_errors", []):
                st.error(e)
            st.caption("The first sequence takes about 30 seconds while the language model loads; later ones take seconds.")
    st.stop()

# --------------------------------------------------------------------------------------------- score pasted sequences
if source == SRC_OWN:
    for nm, sq in ss.get("pasted", []):
        if validate_sequence(sq):
            continue
        with st.spinner(f"Scanning {nm} ({len(sq)} residues)..."):
            r = score_pasted(sq)
        records.append({"name": nm, "seq": sq, "labels": None, "p": r["p"], "kind": "pasted", "contrib": r})

if not records:
    st.markdown('<div class="is-card is-empty"><b>Nothing to show yet</b><span>Choose a test protein or paste your own sequence first.</span></div>',
                unsafe_allow_html=True)
    st.button("◀  Go to step 1", on_click=go, args=(0,), key="back_empty")
    st.stop()

if len(records) > 1:
    names = [r["name"] for r in records]
    pick = st.segmented_control("Sequence", names, default=names[0], key="active") or names[0]
    rec = records[names.index(pick)]
else:
    rec = records[0]
seq, p, labels, name = rec["seq"], rec["p"], rec["labels"], rec["name"]
n = len(seq)
thr = float(ss.get("thr", DEFAULT_THR))

# =========================================================================================================== STEP 2
if step == 1:
    flagged = p >= thr
    top = np.argsort(-p)[:3]
    hot = ", ".join(f"{i + 1} {seq[i]}" for i in top)
    extra = ""
    if labels is not None:
        tp = int((flagged & (labels == 1)).sum())
        extra = (f'<span class="is-chip"><i style="background:{T["truth"]}"></i>found {tp} of {int(labels.sum())} real interface residues</span>'
                 f'<span class="is-chip"><i style="background:{T["accent"]}"></i>{tp / max(int(flagged.sum()), 1) * 100:.0f}% of flags are correct</span>')
    st.markdown(f'<div class="is-result"><div class="is-big">{int(flagged.sum())}</div><div><div class="t">residues of {name} are likely to touch another protein</div>'
                f'<div class="s">out of {n} residues · strongest hotspots: {hot}</div>{extra}</div></div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Your protein, residue by residue**")
        st.markdown(legend_html(labels is not None), unsafe_allow_html=True)
        st.markdown(tiles_html(seq, p, thr, labels == 1 if labels is not None else None), unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1], vertical_alignment="center")
    with c1:
        st.slider("Flag residues with a probability of at least", 0.05, 0.60, step=0.01, key="thr",
                  help="Lower = finds more interface residues but also more false alarms. Higher = fewer, more confident flags.")
    with c2:
        cv = threshold_curve()
        at = cv.iloc[(cv["threshold"] - thr).abs().argmin()]
        with st.popover("What does this setting do?", width="stretch"):
            st.markdown(f"Across the 70 test proteins, at **{thr:.2f}**: **{at.precision * 100:.0f}%** of flagged residues are real interface residues "
                        f"and **{at.recall * 100:.0f}%** of real ones are found. The default {DEFAULT_THR:.2f} was chosen on validation data.")
            long = cv.melt("threshold", ["precision", "recall"], "metric", "value")
            long["metric"] = long["metric"].map({"precision": "flags that are correct", "recall": "real sites found"})
            dom = ["flags that are correct", "real sites found"]
            ln = alt.Chart(long).mark_line(strokeWidth=2.2).encode(
                x=alt.X("threshold:Q", title="Threshold", scale=alt.Scale(domain=[0.05, 0.6], nice=False)),
                y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("metric:N", scale=alt.Scale(domain=dom, range=[T["model"], T["truth"]]), legend=alt.Legend(title=None)),
                strokeDash=alt.StrokeDash("metric:N", scale=alt.Scale(domain=dom, range=[[1, 0], [6, 3]]), legend=None),
                tooltip=[alt.Tooltip("threshold:Q", format=".2f"), alt.Tooltip("metric:N"), alt.Tooltip("value:Q", format=".0%")])
            rule = alt.Chart(pd.DataFrame({"x": [thr]})).mark_rule(color=T["muted"], strokeDash=[5, 4]).encode(x="x:Q")
            st.altair_chart(style(alt.layer(ln, rule).properties(height=200)), theme=None, width="stretch")

    with st.expander("Interactive chart: probability along the chain"):
        df = residue_df(seq, p, labels, thr)
        ev = st.altair_chart(profile_chart(df, thr, labels is not None, max(0.6, float(p.max()) * 1.12)), theme=None, on_select="rerun",
                             selection_mode=["brush"], key=f"map_{name}", width="stretch")
        brush = (ev.selection.get("brush") or {}).get("position") if ev and ev.selection else None
        if brush:
            a, b = int(np.ceil(brush[0])), int(np.floor(brush[1]))
            sub = df[(df["position"] >= a) & (df["position"] <= b)]
            if len(sub):
                m1, m2, m3 = st.columns(3)
                m1.metric(f"Residues {a}-{b}: mean probability", f"{sub['probability'].mean():.2f}")
                m2.metric("Flagged", f"{int(sub['flagged'].sum())} of {len(sub)}")
                m3.metric("Real interface" if labels is not None else "Highest", f"{int(sub['known'].sum())} of {len(sub)}" if labels is not None else f"{sub['probability'].max():.2f}")
        else:
            st.caption("Hover for a residue; drag across the chart to summarise a region.")
        if labels is not None and labels.sum() > 0:
            st.caption(f"Ranking quality inside this protein: average precision {average_precision_score(labels, p):.2f} (random would be {labels.mean():.2f}).")

    with st.expander("Table and download"):
        full = residue_df(seq, p, labels, thr)[["position", "residue", "probability", "flagged", "known"]].round({"probability": 4})
        if labels is None:
            full = full.drop(columns="known")
        st.dataframe(full.sort_values("probability", ascending=False).head(25), hide_index=True)
        st.download_button("Download every residue (CSV)", full.to_csv(index=False).encode(), f"{name}_interface_predictions.csv", "text/csv")

    st.markdown(f'<div class="is-note">Research prototype: on 70 test proteins about {META["test_metrics_at_threshold"]["precision"] * 100:.0f}% of flagged residues '
                f'are real interface residues. Short proteins get more flags. Do not use it for clinical or lab decisions.</div>', unsafe_allow_html=True)
    b1, _, b2 = st.columns([1, 2, 1])
    b1.button("◀  Change protein", on_click=go, args=(0,), key="back_1", width="stretch")
    b2.button("Why these residues?  ▶", type="primary", on_click=go, args=(2,), key="next_2", width="stretch")

# =========================================================================================================== STEP 3
if step == 2:
    c = rec["contrib"] if rec["kind"] == "pasted" else contrib_held_out(name)
    order = np.argsort(-p)
    wkey = f"why_{name}"
    ss.setdefault(wkey, int(order[0]) + 1)
    st.markdown(f"### Why does the model flag a residue in {name}?")
    st.pills("Quick pick: strongest hotspots", [f"{i + 1} {seq[i]}" for i in order[:6]], key=f"hot_{name}",
             on_change=lambda: ss.update({wkey: int(ss[f"hot_{name}"].split()[0])}) if ss.get(f"hot_{name}") else None)
    pos = st.slider("Or choose any residue", 1, n, key=wkey)
    i = pos - 1
    fam, bias = c["fam"][i], c["bias"]
    avg = 1 / (1 + np.exp(-bias))
    biggest = FAMILIES[int(np.argmax(np.abs(fam)))]
    biggest = biggest if biggest.startswith("ESM") else biggest.lower()
    direction = "raises" if fam[int(np.argmax(np.abs(fam)))] > 0 else "lowers"
    known_txt = ""
    if labels is not None:
        known_txt = f" · experiments say: {'real interface residue' if labels[i] == 1 else 'not an interface residue'}"
    st.markdown(f'<div class="is-result"><div class="is-big">{p[i]:.2f}</div><div><div class="t">Residue {pos} ({seq[i]}): interface probability</div>'
                f'<div class="s">An average residue scores {avg:.2f}. The biggest influence is the <b>{biggest}</b>, which {direction} it{known_txt}.</div></div></div>',
                unsafe_allow_html=True)

    with st.container(border=True):
        cdf = pd.DataFrame({"family": FAMILIES, "push": fam, "dir": np.where(fam >= 0, "raises the score", "lowers the score")})
        cdf["text"] = cdf["push"].map(lambda v: f"{v:+.2f}")
        bars = alt.Chart(cdf).mark_bar(cornerRadiusEnd=5, height=28).encode(
            y=alt.Y("family:N", sort=FAMILIES, title=None, axis=alt.Axis(labelLimit=220, labelFontSize=12)),
            x=alt.X("push:Q", title="effect on the score (log-odds)", scale=alt.Scale(nice=True)),
            color=alt.Color("dir:N", scale=alt.Scale(domain=["raises the score", "lowers the score"], range=[T["pos"], T["neg"]]), legend=alt.Legend(title=None)),
            tooltip=[alt.Tooltip("family:N", title="Inputs"), alt.Tooltip("push:Q", format="+.3f", title="Effect")])
        lbl = alt.layer(   # value labels just past the bar end: right of positive bars, left of negative ones
            alt.Chart(cdf[cdf["push"] >= 0]).mark_text(align="left", dx=5, color=T["ink2"], fontSize=11).encode(y=alt.Y("family:N", sort=FAMILIES), x="push:Q", text="text:N"),
            alt.Chart(cdf[cdf["push"] < 0]).mark_text(align="right", dx=-5, color=T["ink2"], fontSize=11).encode(y=alt.Y("family:N", sort=FAMILIES), x="push:Q", text="text:N"))
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=T["axis"]).encode(x="x:Q")
        st.markdown("**What pushed the score up or down**")
        st.altair_chart(style(alt.layer(bars, lbl, zero).properties(height=180, padding={"right": 44, "left": 8, "top": 6, "bottom": 6})), theme=None, width="stretch")
        lo, hi = max(1, pos - 10), min(n, pos + 10)
        st.markdown(f"**Its neighbourhood** (residues {lo}-{hi})")
        st.markdown(tiles_html(seq[lo - 1:hi], p[lo - 1:hi], thr, (labels[lo - 1:hi] == 1) if labels is not None else None, start=lo), unsafe_allow_html=True)
        st.caption(f"The language-model bar sums {ESM_DIM} numbers that describe the residue's context in the chain; none has a meaning on its own. "
                   "These are the model's reasons, not proof of why a residue binds.")

    with st.expander("How the model works and how good it is"):
        fi = family_importance()
        st.markdown(f"""
1. **Sequence** - the 20 standard amino acids, up to {MAX_LEN} residues.
2. **ESM-2 protein language model** ({ESM_SIZE} parameters) turns each residue into {ESM_DIM} numbers describing its context{" (plus the average of those numbers over nearby residues)" if N_CTX else ""}.
3. **Chain context** - amino-acid mix of the whole chain, its length and the residue's position (22 numbers).
4. **XGBoost** turns those {META["n_features"]} numbers into one probability per residue.

On average the model relies on: """ + ", ".join(f"{f if f.startswith('ESM') else f.lower()} {s * 100:.0f}%" for f, s in zip(fi["families"], fi["share"])) + ".")
        m = META["test_metrics_at_threshold"]
        st.markdown(f"""
**How good is it?** On 70 test proteins at threshold {META['threshold']:.2f}: MCC {m['mcc']:.2f}, PR-AUC {m['pr_auc']:.2f},
precision {m['precision']:.2f}, recall {m['recall']:.2f}. Trained on 352 benchmark proteins (DeepPPISP).

**Know the limits**
- It does not know the partner protein, so it predicts any interface the protein might have.
- Short chains score higher on average, so a short protein can get many flags.
- Quality varies a lot between proteins; scores are for ranking residues, not calibrated chances.
""")
    b1, _ = st.columns([1, 3])
    b1.button("◀  Back to the result", on_click=go, args=(1,), key="back_2", width="stretch")
