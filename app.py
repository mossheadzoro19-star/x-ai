"""
Clinician-facing XAI dashboard (Streamlit) - Team 19, KLE Tech.

Run:  pip install -r requirements.txt   then   streamlit run app.py

Model weights: upload best_densenet121.pth in the sidebar (or place it next to app.py).
Gemini key:    paste it in the sidebar (or set GEMINI_API_KEY / .streamlit/secrets.toml).
Nothing is written to disk except the clinician-ratings CSV (no images, no keys).
Research prototype - not a diagnostic device.
"""
import hashlib
import io
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from xai_core import (CLASS_NAMES, generate_llm_explanation, load_model_from_bytes, load_model,
                      predict_and_explain, read_temperature, test_gemini_key, _fallback_summary)
import report

HERE = Path(__file__).parent
RATINGS_CSV = HERE / "clinician_ratings.csv"
DEFAULT_T = 1.284          # temperature found during training; editable in the sidebar
LOW_CONF = 0.70            # below this -> low-confidence warning
CLOSE_GAP = 0.20           # top-2 probability gap below this -> "uncertain" banner
GOOD_LATENCY_S = 10.0      # NFR2

st.set_page_config(page_title="Chest X-ray XAI · Team 19", page_icon="🫁", layout="wide")

# ----------------------------------------------------------------------------
# Design (Olea-style: Hanken Grotesk, forest + lime, soft rounded cards)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@300;400;500&display=swap');
:root{--forest:#2C3A2E;--deep:#26322A;--lime:#CEE39B;--mist:#F2F3EE;--line:#E3E2DD;--muted:#6B6F69;--ink:#1D1E1A;
      --warn:#B4532A;--warnbg:#FBEDE6;--amber:#8A6A12;--amberbg:#FBF3D9;--note:#3D5A66;--notebg:#E8F0F3}
html,body,[class*="css"],.stApp,button,input,textarea{font-family:'Hanken Grotesk',system-ui,sans-serif!important}
.stApp{background:#fff;color:var(--ink)}
.block-container{padding-top:1.4rem;max-width:1240px}
h1,h2,h3{font-weight:300!important;letter-spacing:-.02em;color:var(--ink)}
header[data-testid="stHeader"]{background:transparent}
/* hero */
.hero{background:radial-gradient(1200px 400px at 85% -20%,#40573F 0%,transparent 60%),var(--deep);border-radius:28px;
      padding:34px 40px 30px;color:#fff;margin-bottom:18px}
.hero .badge{display:flex;align-items:center;gap:10px;font:500 11px/1.2 'Hanken Grotesk';letter-spacing:.04em;text-transform:uppercase}
.dot{position:relative;width:8px;height:8px;border-radius:50%;background:var(--lime);flex:none}
.dot::after{content:'';position:absolute;inset:0;border-radius:50%;background:var(--lime);animation:pulse 2s ease-out infinite}
@keyframes pulse{from{transform:scale(1);opacity:.8}to{transform:scale(3);opacity:0}}
.hero h1{color:#fff;font:300 44px/106% 'Hanken Grotesk'!important;letter-spacing:-.025em;margin:18px 0 12px;max-width:820px}
.hero p{color:#D8DDD3;font:400 15px/150% 'Hanken Grotesk';max-width:640px;margin:0}
.eyebrow{display:flex;justify-content:space-between;font:500 11px/1.2 'Hanken Grotesk';letter-spacing:.04em;text-transform:uppercase;
         color:var(--muted);border-top:1px solid var(--line);padding-top:12px;margin:26px 0 14px}
/* cards */
.card{border-radius:24px;padding:26px 28px;background:var(--mist);height:100%}
.card.dark{background:var(--deep);color:#fff}
.card.lime{background:var(--lime)}
.micro{font:500 10px/1.2 'Hanken Grotesk';letter-spacing:.05em;text-transform:uppercase;opacity:.7}
.big{font:300 52px/105% 'Hanken Grotesk';letter-spacing:-.025em;margin:10px 0 4px;font-variant-numeric:tabular-nums}
.mid{font:300 34px/110% 'Hanken Grotesk';letter-spacing:-.015em;margin:10px 0 4px}
.sub{font:400 14px/150% 'Hanken Grotesk';opacity:.85}
.meter{height:8px;border-radius:4px;background:rgba(255,255,255,.2);margin-top:14px;overflow:hidden}
.meter.light{background:rgba(44,58,46,.15)}
.meter>div{height:100%;border-radius:4px;background:var(--lime)}
.meter.light>div{background:var(--forest)}
/* banners */
.banner{border-radius:16px;padding:14px 18px;margin:10px 0;font:400 14px/150% 'Hanken Grotesk'}
.banner b{font-weight:500}
.banner.warn{background:var(--warnbg);color:var(--warn)}
.banner.amber{background:var(--amberbg);color:var(--amber)}
.banner.note{background:var(--notebg);color:var(--note)}
/* probability bars */
.prow{display:flex;align-items:center;gap:12px;margin:9px 0;font:400 14px 'Hanken Grotesk'}
.prow .n{width:104px}.prow .v{width:56px;text-align:right;font-variant-numeric:tabular-nums}
.prow .t{flex:1;height:8px;border-radius:4px;background:#E3E2DD;overflow:hidden}
.prow .t>div{height:100%;border-radius:4px;background:var(--forest)}
.prow.top .t>div{background:#6FA05A}
/* streamlit widgets */
section[data-testid="stSidebar"]{background:var(--mist);border-right:1px solid var(--line)}
section[data-testid="stSidebar"] .block-container{padding-top:1.2rem}
div[data-testid="stFileUploader"] section{border:1px dashed #A2A59F;border-radius:20px;background:#fff}
.stButton>button,.stDownloadButton>button{background:var(--forest);color:#fff;border:0;border-radius:12px;padding:.55rem 1.1rem;font-weight:500;
      transition:transform .35s cubic-bezier(.16,1,.3,1),background .35s,color .35s}
.stButton>button:hover,.stDownloadButton>button:hover{background:var(--lime);color:var(--ink);transform:translateY(-2px)}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:var(--mist);padding:5px;border-radius:14px;width:fit-content}
.stTabs [data-baseweb="tab"]{border-radius:10px;padding:8px 18px;height:auto;color:var(--ink)}
.stTabs [aria-selected="true"]{background:var(--forest);color:#fff!important}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:24px!important;border-color:var(--line)!important}
div[data-testid="stImage"] img{border-radius:20px}
div[data-testid="stExpander"]{border-radius:18px;border-color:var(--line)}
.pill{display:inline-block;padding:5px 12px;border-radius:10px;font:500 12px 'Hanken Grotesk';background:#fff;border:1px solid var(--line)}
.pill.ok{background:var(--lime);border-color:var(--lime);color:var(--ink)}
.disc{background:#fff;border:1px solid var(--line);border-radius:16px;padding:12px 14px;font:400 12px/150% 'Hanken Grotesk';color:var(--muted)}
.disc b{color:var(--ink);font-weight:500}
</style>
""", unsafe_allow_html=True)


def eyebrow(label: str, idx: str):
    st.markdown(f'<div class="eyebrow"><span>// {label}</span><span>[ {idx} ]</span></div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Model + key handling
# ----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner="Loading DenseNet-121 weights...", max_entries=2)
def _cached_uploaded_model(_weights_bytes: bytes, content_hash: str):
    return load_model_from_bytes(_weights_bytes, DEVICE)


@st.cache_resource(show_spinner="Loading DenseNet-121 weights...", max_entries=1)
def _cached_disk_model(path: str, mtime: float):
    return load_model(str(Path(path).parent), DEVICE)[0]


def sha(b: bytes, n: int = 10) -> str:
    return hashlib.sha256(b).hexdigest()[:n]


def secret_key() -> str:
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "") or st.secrets.get("GOOGLE_API_KEY", ""))
    except Exception:
        return ""


with st.sidebar:
    st.markdown("### Setup")
    eyebrow("Model", "01")
    w_up = st.file_uploader("Upload trained weights (.pth)", type=["pth", "pt"], key="w_up",
                            help="best_densenet121.pth from your training notebook. Loaded in memory only.")
    c_up = st.file_uploader("Calibration file (optional)", type=["json"], key="c_up",
                            help="calibration_params.json - contains the temperature T.")

    model, model_src, model_id, detected_T, T_src = None, "", "", DEFAULT_T, "default from training"
    disk_w = HERE / "best_densenet121.pth"
    try:
        if w_up is not None:
            wb = w_up.getvalue()
            model_id = sha(wb)
            model = _cached_uploaded_model(wb, model_id)
            model_src = f"uploaded · {w_up.name}"
        elif disk_w.exists():
            model = _cached_disk_model(str(disk_w), disk_w.stat().st_mtime)
            model_id = f"disk{int(disk_w.stat().st_mtime)}"
            model_src = "local folder · best_densenet121.pth"
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Could not load weights ({type(e).__name__}). Is this a PyTorch state_dict for DenseNet-121?")

    if c_up is not None:
        try:
            detected_T = float(json.loads(c_up.getvalue())["temperature"])
            T_src = "uploaded calibration file"
        except Exception:
            st.warning("Could not read a 'temperature' from that JSON; using the default.")
    elif (HERE / "calibration_params.json").exists():
        detected_T = read_temperature(str(HERE / "calibration_params.json"), DEFAULT_T)
        T_src = "local calibration_params.json"
    T = st.number_input("Calibration temperature T", 0.1, 10.0, float(detected_T), 0.001, format="%.3f",
                        key=f"T_{T_src}_{detected_T}", help=f"Detected from: {T_src}. Confidence = softmax(logits / T).")

    if model is not None:
        st.markdown(f'<span class="pill ok">● Model ready</span> <span class="pill">{DEVICE.type.upper()}</span>',
                    unsafe_allow_html=True)
        st.caption(f"{model_src} · T = {T:.3f}")
    else:
        st.markdown('<span class="pill">○ No weights loaded</span>', unsafe_allow_html=True)

    eyebrow("Gemini explanation", "02")
    key_in = st.text_input("Gemini API key", type="password", placeholder="Paste key here (AIza...)",
                           help="Used only for this session, held in memory, never written to disk or logs. "
                                "The model receives numbers only - never your X-ray.")
    eff_key = (key_in or "").strip() or secret_key()
    if st.button("Test key", width="stretch"):
        with st.spinner("Checking key..."):
            ok, msg = test_gemini_key(eff_key)
        st.session_state["key_test"] = (ok, msg)
    kt = st.session_state.get("key_test")
    if kt:
        (st.success if kt[0] else st.error)(kt[1])
    use_llm = st.toggle("Generate Gemini explanation", value=True)
    st.caption("No key → a deterministic safe summary is shown instead.")

    eyebrow("Options", "03")
    run_occ = st.toggle("Occlusion baseline", value=DEVICE.type == "cuda",
                        help="Slow on CPU (can exceed the 10 s target). Default ON only with a GPU.")

    st.markdown('<div class="disc"><b>Educational research prototype.</b> Not a diagnostic device. Heatmaps show '
                'model sensitivity, not verified pathology. The clinician keeps final decision authority.</div>',
                unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="badge"><span class="dot"></span>Research prototype · Team 19 · KLE Tech</div>
  <h1>AI support that shows its confidence, and its reasoning.</h1>
  <p>Upload a chest X-ray to see the predicted class, calibrated confidence, a Grad-CAM heatmap, an Explanation
  Usefulness Score and a plain-language summary, on one screen.</p>
</div>
""", unsafe_allow_html=True)

tab_analyze, tab_validate, tab_perf = st.tabs(["Analyze X-ray", "Clinician validation", "Model performance"])


# ----------------------------------------------------------------------------
# helpers for results
# ----------------------------------------------------------------------------
def us_reading(norm: float) -> str:
    if norm >= 0.67:
        return "High: on the prototype concept table the predicted class is strongly separated from its closest alternative."
    if norm >= 0.34:
        return "Moderate: the predicted class is only partly separated from its closest alternative."
    return "Low: the predicted class and its closest alternative overlap on the prototype concept table."


def banners(res):
    conf, fact, foil = res["confidence"], res["fact_class"], res["foil_class"]
    gap = conf - res["foil_confidence"]
    if conf < LOW_CONF:
        st.markdown(f'<div class="banner warn"><b>Low confidence ({conf*100:.0f}% &lt; 70%).</b> '
                    f'Treat this prediction as uncertain and rely on clinical judgement.</div>', unsafe_allow_html=True)
    if gap < CLOSE_GAP:
        st.markdown(f'<div class="banner amber"><b>Close call.</b> {fact.capitalize()} ({conf*100:.0f}%) and '
                    f'{foil.capitalize()} ({res["foil_confidence"]*100:.0f}%) are within {gap*100:.0f} points of each other. '
                    f'The result is uncertain.</div>', unsafe_allow_html=True)
    if fact in ("normal", "tuberculosis"):
        extra = " The model is known to confuse these two classes." if {fact, foil} == {"normal", "tuberculosis"} else ""
        st.markdown(f'<div class="banner note"><b>Extra caution for {fact.capitalize()} predictions.</b> '
                    f'On the test set this model is weaker on Normal (recall 0.59) and Tuberculosis (recall 0.73) '
                    f'than on Pneumonia (0.99).{extra}</div>', unsafe_allow_html=True)


def prob_bars(probs: dict, top: str):
    html = ""
    for name in CLASS_NAMES:
        p = probs[name]
        html += (f'<div class="prow {"top" if name == top else ""}"><span class="n">{name.capitalize()}</span>'
                 f'<span class="t"><div style="width:{p*100:.1f}%"></div></span><span class="v">{p*100:.1f}%</span></div>')
    st.markdown(html, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Analyze tab
# ----------------------------------------------------------------------------
with tab_analyze:
    if model is None:
        st.markdown("""
<div class="card"><div class="micro">Step 1</div><div class="mid">Load the trained model</div>
<p class="sub">Upload <b>best_densenet121.pth</b> in the sidebar (and optionally <b>calibration_params.json</b>).
Or place the files next to <code>app.py</code> and refresh. Add your Gemini key below it for the AI-written explanation.</p></div>
""", unsafe_allow_html=True)
    else:
        eyebrow("Upload", "01")
        up = st.file_uploader("Chest X-ray (JPG / PNG)", type=["jpg", "jpeg", "png"], key="xray",
                              label_visibility="collapsed")
        if up is None:
            st.info("Upload a chest X-ray to begin. Files are processed in memory and never saved.")
        else:
            img_bytes = up.getvalue()
            case_id = sha(img_bytes, 8)
            rkey = (case_id, model_id, round(T, 4), run_occ)
            if st.session_state.get("rkey") != rkey:
                res = None
                try:
                    with st.spinner("Running prediction + explanation..."):
                        t0 = time.perf_counter()
                        res = predict_and_explain(io.BytesIO(img_bytes), model, DEVICE, T, run_occlusion=run_occ)
                        latency = time.perf_counter() - t0
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Analysis failed ({type(e).__name__}). Try a different image.")
                if res is not None:
                    st.session_state.update(rkey=rkey, res=res, latency=latency, llm_key=None, llm_text=None)
                    st.session_state["last"] = {
                        "case_id": case_id, "fact": res["fact_class"], "foil": res["foil_class"],
                        "confidence": res["confidence"], "us": res["usefulness"]["usefulness_score"],
                        "us_norm": res["usefulness"]["usefulness_normalized"]}
            if st.session_state.get("rkey") == rkey:
                res, latency = st.session_state["res"], st.session_state["latency"]
                u = res["usefulness"]

                # --- headline cards -------------------------------------------------
                eyebrow("Result", "02")
                c1, c2, c3 = st.columns([1.15, 1, 1.15])
                with c1:
                    st.markdown(f"""<div class="card dark"><div class="micro">Diagnosis (model prediction)</div>
    <div class="big">{res['fact_class'].capitalize()}</div>
    <div class="sub">Calibrated confidence <b>{res['confidence']*100:.1f}%</b></div>
    <div class="meter"><div style="width:{res['confidence']*100:.1f}%"></div></div></div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="card"><div class="micro">Closest alternative (foil)</div>
    <div class="mid">{res['foil_class'].capitalize()}</div>
    <div class="sub">{res['foil_confidence']*100:.1f}% calibrated probability</div>
    <div class="meter light"><div style="width:{res['foil_confidence']*100:.1f}%"></div></div></div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""<div class="card lime"><div class="micro">Explanation usefulness score</div>
    <div class="mid">{u['usefulness_normalized']:.2f} <span style="font-size:16px;opacity:.7">/ 1 &nbsp;·&nbsp; raw {u['usefulness_score']:+.2f}</span></div>
    <div class="sub">{us_reading(u['usefulness_normalized'])}</div>
    <div class="meter light"><div style="width:{u['usefulness_normalized']*100:.0f}%"></div></div></div>""", unsafe_allow_html=True)

                banners(res)

                # --- images side by side -------------------------------------------
                eyebrow("Original vs heatmap", "03")
                cols = st.columns(3 if res["occlusion_overlay"] is not None else 2)
                cols[0].image(res["original_image"], caption="Original X-ray", width="stretch")
                cols[1].image(res["overlay"], caption="Grad-CAM overlay (model sensitivity)", width="stretch")
                if res["occlusion_overlay"] is not None:
                    cols[2].image(res["occlusion_overlay"], caption="Occlusion baseline", width="stretch")

                with st.expander("How to read this"):
                    st.markdown(
                        "- **Heatmap** = where the model was *sensitive*, not verified pathology. Warm colours mean more influence on the prediction.\n"
                        "- **Confidence** is a temperature-calibrated statistical estimate, not the probability that the patient has the disease.\n"
                        "- **Usefulness score** here is a *prototype*: it depends only on the (predicted class, closest alternative) pair "
                        "and a transparent hand-written concept table (λ = 0.15, US = CS − λ·C). "
                        "It does **not** change with the image or the heatmap, so do not read it as image-specific clinical truth.\n"
                        "- **Accuracy context:** measured test accuracy is 73.69% (below the 85% target). Pneumonia is reliable; "
                        "Normal ↔ Tuberculosis is often confused.\n"
                        "- The clinician keeps final decision authority.")

                # --- probabilities + faithfulness ----------------------------------
                eyebrow("Probabilities & faithfulness", "04")
                p1, p2 = st.columns(2)
                with p1:
                    with st.container(border=True):
                        st.markdown("**Class probabilities** (temperature-calibrated)")
                        prob_bars(res["all_probabilities"], res["fact_class"])
                with p2:
                    with st.container(border=True):
                        st.markdown("**Faithfulness** · probability drop after deleting the top-k% attributed pixels")
                        fdf = pd.DataFrame({"Grad-CAM": res["gradcam_drops"]}, index=["10%", "20%", "30%"])
                        if res["occlusion_drops"] is not None:
                            fdf["Occlusion"] = res["occlusion_drops"]
                        st.dataframe(fdf.style.format("{:.3f}"), width="stretch")
                with st.expander("Usefulness score breakdown"):
                    st.write(f"CS = {u['contrastive_score']} | penalty λ·C = {u['complexity_penalty']} | US = {u['usefulness_score']}")
                    st.dataframe(pd.DataFrame(u["concept_breakdown"]).T, width="stretch")

                # --- explanation ----------------------------------------------------
                eyebrow("Explanation", "05")
                lkey = (rkey, use_llm, hashlib.sha256(eff_key.encode()).hexdigest() if eff_key else "")
                if st.session_state.get("llm_key") != lkey:
                    t1 = time.perf_counter()
                    if use_llm:
                        with st.spinner("Generating explanation..."):
                            text = generate_llm_explanation(res, eff_key or None)
                    else:
                        text = _fallback_summary(res)
                    st.session_state.update(llm_key=lkey, llm_text=text, llm_time=time.perf_counter() - t1)
                text = st.session_state["llm_text"]
                with st.container(border=True):
                    st.markdown(text)

                # --- footer: latency + report --------------------------------------
                total = latency
                ok = "within" if total <= GOOD_LATENCY_S else "above"
                st.caption(f"Prediction + explanation took {latency:.1f} s on {DEVICE.type.upper()} "
                           f"({ok} the 10 s target)" + ("" if run_occ else " · occlusion off") +
                           f" · LLM text {st.session_state.get('llm_time', 0):.1f} s · case ref {case_id}")
                try:
                    pdf = report.build_pdf(res, text, case_id, latency)
                    st.download_button("Download PDF report", pdf, file_name=f"xai_report_{case_id}.pdf",
                                       mime="application/pdf")
                except Exception as e:
                    st.caption(f"PDF report unavailable ({type(e).__name__}).")


# ----------------------------------------------------------------------------
# Clinician validation tab
# ----------------------------------------------------------------------------
with tab_validate:
    eyebrow("Clinician validation", "06")
    st.write("Rate how useful the last explanation was (1 = not useful, 5 = very useful). Ratings are stored in "
             "`clinician_ratings.csv` (no images, no names, no API key) and compared with the automated score.")
    last = st.session_state.get("last")
    if not last:
        st.info("Analyze an X-ray first.")
    else:
        st.write(f"Last case **{last['case_id']}** → {last['fact'].capitalize()} "
                 f"({last['confidence']*100:.1f}%), US = {last['us']:+.2f} (normalised {last['us_norm']:.2f})")
        with st.form("rating"):
            clin = st.text_input("Clinician ID / initials")
            rating = st.slider("Explanation usefulness", 1, 5, 3)
            comment = st.text_area("Comment (optional, no patient information)")
            if st.form_submit_button("Submit rating"):
                row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "clinician": clin or "anon",
                       **last, "rating": rating, "comment": comment}
                pd.DataFrame([row]).to_csv(RATINGS_CSV, mode="a", header=not RATINGS_CSV.exists(), index=False)
                st.success("Saved.")
    if RATINGS_CSV.exists():
        try:
            df = pd.read_csv(RATINGS_CSV)
            st.dataframe(df.tail(20), width="stretch")
            if len(df) >= 3 and {"us", "rating", "confidence"} <= set(df.columns):
                from scipy.stats import spearmanr

                def sp(a, b):
                    if df[a].nunique() < 2 or df[b].nunique() < 2:
                        return "n/a (no variation yet - rate different cases)"
                    r, p = spearmanr(df[a], df[b])
                    return f"**{r:.3f}** (p={p:.3f})"
                st.write(f"Spearman(automated US, clinician rating) = {sp('us', 'rating')}, n={len(df)}")
                st.write(f"Spearman(model confidence, clinician rating) = {sp('confidence', 'rating')}")
                st.caption("The prototype US depends only on the (Fact, Foil) pair, so it takes at most 6 distinct values.")
            else:
                st.caption("Need at least 3 ratings to compute agreement.")
        except Exception:
            st.warning("The ratings file could not be read (old format?). Delete clinician_ratings.csv to start fresh.")

# ----------------------------------------------------------------------------
# Performance tab
# ----------------------------------------------------------------------------
with tab_perf:
    eyebrow("Measured performance", "07")
    a, b, c, d = st.columns(4)
    for col, label, val, sub in [(a, "Test accuracy", "73.69%", "target ≥ 85%"), (b, "Macro-F1", "0.747", ""),
                                 (c, "OvR AUC", "0.921", ""), (d, "Recall N / P / TB", "0.59 / 0.99 / 0.73", "")]:
        col.markdown(f'<div class="card"><div class="micro">{label}</div><div class="mid">{val}</div>'
                     f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)
    st.caption("Figures recorded at hand-off. Pneumonia is reliable; Normal ↔ Tuberculosis is confused. "
               "Weights may be replaced - this page shows metrics.json and plots if present next to app.py.")
    mpath = HERE / "metrics.json"
    if mpath.exists():
        try:
            st.json(json.loads(mpath.read_text()))
        except Exception:
            st.warning("metrics.json could not be parsed.")
    shown = False
    for name in ["training_curves.png", "eval_plots.png", "faithfulness_comparison.png"]:
        if (HERE / name).exists():
            st.image(str(HERE / name), caption=name, width="stretch")
            shown = True
    if not (mpath.exists() or shown):
        st.info("Place metrics.json, training_curves.png, eval_plots.png, faithfulness_comparison.png next to app.py to show them here.")
