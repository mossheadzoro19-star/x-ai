"""
Clinician-facing XAI dashboard (Streamlit) - Team 19, KLE Tech.

Run:  pip install -r requirements.txt   then   streamlit run app.py

Model weights: upload best_densenet121.pth in the sidebar (or place it next to app.py).
Gemini key:    paste it in the sidebar (or set GEMINI_API_KEY / .streamlit/secrets.toml).
Nothing is written to disk except the clinician-ratings CSV (no images, no keys).
Research prototype - not a diagnostic device.
"""
import hashlib
import os
import io
import json
import time
import tempfile
import urllib.request
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
METRICS_PATH = HERE / "metrics.json"
try:
    MODEL_METRICS = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
except Exception:
    MODEL_METRICS = {}
DEFAULT_T = 1.4974132776260376  # clean-data checkpoint calibration; overridden by calibration_params.json
LOW_CONF = 0.70            # below this -> low-confidence warning
CLOSE_GAP = 0.20           # top-2 probability gap below this -> "uncertain" banner
GOOD_LATENCY_S = 10.0      # NFR2

st.set_page_config(page_title="Chest X-ray XAI · Team 19", page_icon="🫁", layout="wide", initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------
# Design (Olea-style: Hanken Grotesk, forest + lime, soft rounded cards)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{
  --forest:#26382F;--forest-2:#344B3D;--lime:#D8E8A8;--lime-2:#EDF4D9;
  --canvas:#F5F7F4;--surface:#FFFFFF;--mist:#F0F3EF;--line:#E2E8E1;
  --muted:#69766D;--ink:#1D2922;--warn:#A64B32;--warnbg:#FFF1EB;
  --amber:#85651C;--amberbg:#FFF7E3;--note:#365B69;--notebg:#EAF3F6;
  --shadow:0 12px 32px rgba(28,45,34,.055);
}
html,body,[class*="css"],.stApp,button,input,textarea{font-family:'DM Sans',system-ui,sans-serif!important}
.stApp{background:var(--canvas);color:var(--ink)}
.block-container{padding:2.1rem 2rem 4rem;max-width:1320px}
h1,h2,h3{font-family:'Manrope','DM Sans',sans-serif!important;font-weight:700!important;letter-spacing:-.035em;color:var(--ink)}
header[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}
.hero{
  position:relative;overflow:hidden;
  background:radial-gradient(ellipse at 90% 0%,rgba(126,157,105,.34),transparent 42%),
             linear-gradient(130deg,#1F3027 0%,#2D4235 58%,#344A3A 100%);
  border:1px solid rgba(255,255,255,.08);border-radius:30px;
  padding:38px 46px 36px;color:#fff;margin:0 0 22px;
  box-shadow:0 18px 42px rgba(29,48,36,.12)
}
.hero .badge{display:flex;align-items:center;gap:10px;font:700 11px/1.2 'DM Sans';letter-spacing:.09em;text-transform:uppercase;color:#E4EBDD}
.dot{position:relative;width:9px;height:9px;border-radius:50%;background:var(--lime);flex:none;box-shadow:0 0 0 5px rgba(216,232,168,.13)}
.dot::after{content:'';position:absolute;inset:-3px;border-radius:50%;border:1px solid var(--lime);animation:pulse 2.4s ease-out infinite}
@keyframes pulse{from{transform:scale(.8);opacity:.75}to{transform:scale(2.1);opacity:0}}
.hero h1{color:#fff;font:700 clamp(32px,4vw,46px)/1.12 'Manrope'!important;letter-spacing:-.045em;margin:22px 0 12px;max-width:850px}
.hero p{color:#DCE6DC;font:400 15px/1.65 'DM Sans';max-width:700px;margin:0}
.eyebrow{display:flex;justify-content:space-between;align-items:center;font:700 10px/1.2 'DM Sans';letter-spacing:.11em;text-transform:uppercase;
  color:#78857B;border-top:1px solid var(--line);padding-top:14px;margin:28px 0 16px}
.card{border:1px solid var(--line);border-radius:22px;padding:24px 26px;background:var(--surface);height:100%;box-shadow:var(--shadow)}
.card.dark{background:linear-gradient(145deg,#24362C,#31483A);border-color:#31483A;color:#fff}
.card.lime{background:linear-gradient(145deg,#E5EFC7,#D5E5A8);border-color:#D5E5A8}
.micro{font:700 10px/1.35 'DM Sans';letter-spacing:.09em;text-transform:uppercase;opacity:.72}
.big{font:700 clamp(34px,3vw,46px)/1.05 'Manrope';letter-spacing:-.045em;margin:13px 0 7px;font-variant-numeric:tabular-nums}
.mid{font:700 clamp(25px,2.4vw,34px)/1.1 'Manrope';letter-spacing:-.035em;margin:13px 0 7px}
.sub{font:400 13px/1.55 'DM Sans';opacity:.88}
.meter{height:7px;border-radius:20px;background:rgba(255,255,255,.18);margin-top:16px;overflow:hidden}
.meter.light{background:rgba(38,56,47,.12)}
.meter>div{height:100%;border-radius:20px;background:var(--lime)}
.meter.light>div{background:var(--forest)}
.banner{border-radius:14px;padding:13px 16px;margin:10px 0;font:400 13px/1.6 'DM Sans';border:1px solid transparent}
.banner b{font-weight:700}
.banner.warn{background:var(--warnbg);color:var(--warn);border-color:#F6D9CF}
.banner.amber{background:var(--amberbg);color:var(--amber);border-color:#F1E4BD}
.banner.note{background:var(--notebg);color:var(--note);border-color:#D7E7EC}
.prow{display:flex;align-items:center;gap:12px;margin:11px 0;font:500 13px 'DM Sans'}
.prow .n{width:108px;color:#526056}.prow .v{width:58px;text-align:right;font-variant-numeric:tabular-nums;color:#344238}
.prow .t{flex:1;height:8px;border-radius:20px;background:#E8EDE7;overflow:hidden}
.prow .t>div{height:100%;border-radius:20px;background:#8EA77F}
.prow.top .t>div{background:#385D43}
section[data-testid="stSidebar"]{background:var(--mist);border-right:1px solid var(--line)}
section[data-testid="stSidebar"] .block-container{padding-top:1.2rem}
div[data-testid="stFileUploader"]{margin:4px 0 12px}
div[data-testid="stFileUploader"] section{border:1.5px dashed #B7C3B8;border-radius:18px;background:#FBFCFA;padding:16px 18px;transition:border-color .2s,background .2s}
div[data-testid="stFileUploader"] section:hover{border-color:#718B70;background:#F4F8F1}
.stButton>button,.stDownloadButton>button{background:var(--forest);color:#fff;border:1px solid var(--forest);border-radius:11px;padding:.58rem 1.15rem;font-weight:700;transition:all .18s ease}
.stButton>button:hover,.stDownloadButton>button:hover{background:#3D5B46;border-color:#3D5B46;color:#fff;transform:translateY(-1px);box-shadow:0 6px 16px rgba(38,56,47,.16)}
.stTabs [data-baseweb="tab-list"]{gap:6px;background:#E9EEE8;padding:5px;border:1px solid #E0E7DF;border-radius:14px;width:fit-content;margin:0 0 8px}
.stTabs [data-baseweb="tab"]{border-radius:10px;padding:10px 18px;height:auto;color:#536057;font:600 13px 'DM Sans';background:transparent}
.stTabs [aria-selected="true"]{background:var(--forest)!important;color:#fff!important;box-shadow:0 2px 6px rgba(25,45,31,.14)}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:18px!important;border-color:var(--line)!important;background:#fff}
div[data-testid="stImage"] img{border-radius:18px;border:1px solid rgba(34,54,40,.08)}
div[data-testid="stExpander"]{border-radius:14px;border:1px solid var(--line);background:#FBFCFA}
div[data-testid="stAlert"]{border-radius:14px}
.pill{display:inline-block;padding:6px 12px;border-radius:10px;font:700 11px 'DM Sans';background:#fff;border:1px solid var(--line);color:#405046}
.pill.ok{background:var(--lime-2);border-color:#D9E8B8;color:#344B35}
.disc{background:#fff;border:1px solid var(--line);border-radius:15px;padding:13px 16px;font:400 12px/1.6 'DM Sans';color:var(--muted);box-shadow:0 3px 12px rgba(28,45,34,.025)}
.disc b{color:var(--ink);font-weight:700}
.stApp,.stApp p,.stApp label,.stApp span,.stApp li,.stApp h1,.stApp h2,.stApp h3,.stApp small,
[data-testid="stWidgetLabel"] *,[data-testid="stCaptionContainer"] *,[data-testid="stExpander"] summary *{color:var(--ink)}
.hero,.hero *,.card.dark,.card.dark *{color:#fff!important}
.hero p{color:#DCE6DC!important}
.card.lime,.card.lime *{color:#243528!important}
.stTabs [aria-selected="true"],.stTabs [aria-selected="true"] *{color:#fff!important}
.stButton>button,.stDownloadButton>button,.stButton>button *,.stDownloadButton>button *{color:#fff!important}
div[data-testid="stFileUploader"] button,div[data-testid="stFileUploader"] button *{color:#fff!important}
html{scroll-behavior:smooth}
@media(max-width:800px){
  .block-container{padding:1rem 1rem 3rem}
  .hero{padding:28px 24px;border-radius:22px}
  .hero h1{font-size:32px!important}
  .stTabs [data-baseweb="tab-list"]{width:100%;overflow-x:auto}
  .stTabs [data-baseweb="tab"]{padding:9px 12px;font-size:12px}
  .card{padding:20px;border-radius:18px}
}
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


@st.cache_resource(show_spinner="Downloading model checkpoint...")
def _download_checkpoint(url: str, expected_sha256: str = "", github_token: str = "") -> str:
    """Download a deployment-hosted checkpoint to temporary storage, verifying it when a hash is supplied."""
    target = Path(tempfile.gettempdir()) / "best_densenet121.pth"
    headers = {"User-Agent": "Chest-Xray-XAI-Streamlit"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
        headers["Accept"] = "application/octet-stream"
    request = urllib.request.Request(url, headers=headers)
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=180) as response, target.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            out.write(chunk)
    actual = digest.hexdigest()
    if expected_sha256 and actual.lower() != expected_sha256.strip().lower():
        target.unlink(missing_ok=True)
        raise ValueError("Downloaded model SHA-256 does not match MODEL_WEIGHTS_SHA256.")
    if target.stat().st_size < 1024 * 1024:
        target.unlink(missing_ok=True)
        raise ValueError("Downloaded model checkpoint is unexpectedly small.")
    return str(target)


def secret_key() -> str:
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "") or st.secrets.get("GOOGLE_API_KEY", ""))
    except Exception:
        return ""


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


# ----------------------------------------------------------------------------
# Setup (collapsed dropdown). Set HIDE_SETUP=1 to show it only with ?admin=1 in the URL.
# ----------------------------------------------------------------------------
def _flag(name: str) -> bool:
    if os.environ.get(name) == "1":
        return True
    try:
        return str(st.secrets.get(name, "")) == "1"
    except Exception:
        return False


show_setup = st.query_params.get("admin") == "1"
w_up = c_up = None
key_in = ""
use_llm = True
run_occ = DEVICE.type == "cuda"
detected_T, T_src, T = DEFAULT_T, "default from training", DEFAULT_T
status_box = None

if show_setup:
    with st.expander("⚙  Setup · model weights & Gemini key (click to open)", expanded=False):
        status_box = st.container()
        sc1, sc2 = st.columns(2)
        with sc1:
            eyebrow("Model", "01")
            w_up = st.file_uploader("Upload trained weights (.pth)", type=["pth", "pt"], key="w_up",
                                    help="best_densenet121.pth from your training notebook. Loaded in memory only.")
            c_up = st.file_uploader("Calibration file (optional)", type=["json"], key="c_up",
                                    help="calibration_params.json - contains the temperature T.")
            T_slot = st.container()
        with sc2:
            eyebrow("Gemini explanation", "02")
            key_in = st.text_input("Gemini API key", type="password", placeholder="Paste key here (AIza...)",
                                   help="Used only for this session, held in memory, never written to disk or logs. "
                                        "The model receives numbers only - never your X-ray.")
            _eff = (key_in or "").strip() or secret_key()
            if st.button("Test key"):
                with st.spinner("Checking key..."):
                    ok, msg = test_gemini_key(_eff)
                st.session_state["key_test"] = (ok, msg)
            kt = st.session_state.get("key_test")
            if kt:
                (st.success if kt[0] else st.error)(kt[1])
            use_llm = st.toggle("Generate Gemini explanation", value=True)
            st.caption("No key → a deterministic safe summary is shown instead.")
            run_occ = st.toggle("Occlusion baseline", value=DEVICE.type == "cuda",
                                help="Slow on CPU (can exceed the 10 s target). Default ON only with a GPU.")
eff_key = (key_in or "").strip() or secret_key()

# ---- resolve model + temperature ----
model, model_src, model_id = None, "", ""
disk_w = HERE / "best_densenet121.pth"
load_err = None
try:
    if w_up is not None:
        wb = w_up.getvalue()
        model_id = sha(wb)
        model = _cached_uploaded_model(wb, model_id)
        model_src = f"uploaded · {w_up.name}"
    else:
        deploy_url = ""
        deploy_sha = ""
        deploy_token = ""
        try:
            deploy_url = str(st.secrets.get("MODEL_WEIGHTS_URL", "") or "").strip()
            deploy_sha = str(st.secrets.get("MODEL_WEIGHTS_SHA256", "") or "").strip()
            deploy_token = str(st.secrets.get("MODEL_GITHUB_TOKEN", "") or "").strip()
        except Exception:
            pass
        if disk_w.exists():
            chosen_w = disk_w
            model_src = "local folder · best_densenet121.pth"
        elif deploy_url:
            chosen_w = Path(_download_checkpoint(deploy_url, deploy_sha, deploy_token))
            model_src = "configured deployment checkpoint"
        else:
            chosen_w = None
        if chosen_w is not None:
            model = _cached_disk_model(str(chosen_w), chosen_w.stat().st_mtime)
            model_id = f"disk{int(chosen_w.stat().st_mtime)}"
except ValueError as e:
    load_err = str(e)
except Exception as e:
    load_err = f"Could not load weights ({type(e).__name__}). Is this a PyTorch state_dict for DenseNet-121?"

if c_up is not None:
    try:
        detected_T = float(json.loads(c_up.getvalue())["temperature"])
        T_src = "uploaded calibration file"
    except Exception:
        pass
elif (HERE / "calibration_params.json").exists():
    detected_T = read_temperature(str(HERE / "calibration_params.json"), DEFAULT_T)
    T_src = "local calibration_params.json"
T = float(detected_T)
if show_setup:
    with T_slot:
        T = st.number_input("Calibration temperature T", 0.1, 10.0, float(detected_T), 0.001, format="%.3f",
                            key=f"T_{T_src}_{detected_T}",
                            help=f"Detected from: {T_src}. Confidence = softmax(logits / T).")
    with status_box:
        if load_err:
            st.error(load_err)
        if model is not None:
            st.markdown(f'<span class="pill ok">● Model ready</span> <span class="pill">{DEVICE.type.upper()}</span> '
                        f'<span class="pill">T = {T:.3f}</span>', unsafe_allow_html=True)
            st.caption(model_src)
        else:
            st.markdown('<span class="pill">○ No weights loaded yet</span>', unsafe_allow_html=True)
elif load_err:
    st.error(load_err)

st.markdown('<div class="disc"><b>Educational research prototype.</b> Not a diagnostic device. Heatmaps show '
            'model sensitivity, not verified pathology. The clinician keeps final decision authority.</div>',
            unsafe_allow_html=True)

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
    per_class = MODEL_METRICS.get("per_class", {})
    recall_fact = per_class.get(fact, {}).get("sensitivity")
    if recall_fact is not None and float(recall_fact) < 0.80:
        recall_pct = float(recall_fact) * 100
        st.markdown(f'<div class="banner note"><b>Performance caution for {fact.capitalize()}.</b> '
                    f'Recorded test recall for this class is {recall_pct:.1f}%. This is a model-level evaluation '
                    f'metric, not a guarantee for this individual prediction.</div>', unsafe_allow_html=True)
    if MODEL_METRICS.get("metadata_shortcut_accuracy") is not None and float(MODEL_METRICS["metadata_shortcut_accuracy"]) >= 0.90:
        st.markdown('<div class="banner warn"><b>Evaluation warning.</b> A metadata-only baseline predicts the test labels '
                    'with very high accuracy. The reported model accuracy may reflect dataset/source shortcuts; '
                    'independent external validation is required.</div>', unsafe_allow_html=True)


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
    eyebrow("Upload X-ray", "01")
    up = st.file_uploader("Chest X-ray (JPG / PNG)", type=["jpg", "jpeg", "png"], key="xray",
                          disabled=model is None, label_visibility="collapsed")
    if model is None:
        st.error("The configured DenseNet-121 checkpoint was not found or could not be loaded. Confirm best_densenet121.pth is beside app.py and restart the app. Admin setup is available by adding ?admin=1 to the URL.")
    elif up is None:
        st.info("Upload a chest X-ray above to begin. Files are processed in memory and never saved.")
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
                    "- **Accuracy context:** see the Model performance tab for metrics from the bundled checkpoint. "
                    "Review the evaluation-integrity warning before interpreting the reported score.\n"
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
    if not MODEL_METRICS:
        st.warning("No metrics.json found beside app.py. Performance figures are intentionally not hard-coded because they must match the loaded checkpoint.")
        st.info("Add the metrics.json generated by the same training run as best_densenet121.pth.")
    else:
        acc = float(MODEL_METRICS.get("accuracy", 0.0))
        macro_f1 = float(MODEL_METRICS.get("macro_f1", 0.0))
        auc = float(MODEL_METRICS.get("ovr_roc_auc", 0.0))
        pc = MODEL_METRICS.get("per_class", {})
        recalls = [float(pc.get(k, {}).get("sensitivity", 0.0)) for k in ("normal", "pneumonia", "tuberculosis")]
        a, b, c, d = st.columns(4)
        for col, label, val, sub in [
            (a, "Test accuracy", f"{acc*100:.2f}%", "target ≥ 85%"),
            (b, "Macro-F1", f"{macro_f1:.3f}", ""),
            (c, "OvR AUC", f"{auc:.3f}", ""),
            (d, "Recall N / P / TB", " / ".join(f"{x:.2f}" for x in recalls), "")
        ]:
            col.markdown(f'<div class="card"><div class="micro">{label}</div><div class="mid">{val}</div>'
                         f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)
        shortcut = MODEL_METRICS.get("metadata_shortcut_accuracy")
        if shortcut is not None and float(shortcut) >= 0.90:
            st.error(f"Evaluation integrity warning: metadata-only accuracy is {float(shortcut)*100:.2f}%. "
                     "This is close to the model's reported score and suggests possible source/metadata shortcuts. "
                     "Do not claim the model generalizes until an independent, source-separated test is run.")
        elif acc < 0.85:
            st.warning("The measured test accuracy is below the project's 85% target.")
        st.caption(f"Checkpoint evaluation · test size {MODEL_METRICS.get('test_size', 'not recorded')} · "
                   f"temperature T = {MODEL_METRICS.get('temperature', 'not recorded')}")
        st.json(MODEL_METRICS)
    shown = False
    for name in ["training_curves.png", "eval_plots.png", "faithfulness_comparison.png"]:
        if (HERE / name).exists():
            st.image(str(HERE / name), caption=name, width="stretch")
            shown = True
    if not shown:
        st.info("Training and evaluation plots will appear here when their matching PNG files are placed beside app.py.")
