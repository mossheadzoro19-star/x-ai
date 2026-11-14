# Chest X-ray XAI Dashboard (Team 19, KLE Tech)

Upload a chest X-ray and get, on one screen: predicted class, calibrated confidence, Grad-CAM heatmap beside the
original, Explanation Usefulness Score (raw + 0-1), and a guard-railed Gemini explanation. Plus a clinician-validation
tab and a model-performance tab. **Educational research prototype, not a diagnostic device.**

## Install & run
    pip install -r requirements.txt
    streamlit run app.py

## The two things you set in the sidebar
1. **Model weights** - drop `best_densenet121.pth` into "Upload trained weights". Optionally also
   `calibration_params.json` (temperature T, about 1.284). Weights load in memory only (`weights_only=True`), nothing is saved.
   Alternative: put the files next to `app.py` and they load automatically.
2. **Gemini key** - paste it into "Gemini API key" and press **Test key**. It stays in memory for the session only.
   Alternatives: env var `GEMINI_API_KEY`, or `.streamlit/secrets.toml` (see the `.example` file).
   Without a key the app shows a deterministic safe summary. Gemini only ever receives numbers, never the image.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (Olea-style design) |
| `xai_core.py` | Model loading, Grad-CAM, occlusion, faithfulness, usefulness score, LLM call (UI-free) |
| `report.py` | PDF report builder (no patient identifiers) |
| `metrics.json`, `*.png` | Optional, shown in the Performance tab if placed next to `app.py` |

## Notes
- Occlusion baseline is OFF by default on CPU (about 30 s on one core) and ON on GPU. Without it, a CPU run takes about 1 s.
- Class order is fixed: normal, pneumonia, tuberculosis. Preprocessing is unchanged from training.
- Measured test accuracy at hand-off: 73.69% (target 85%). Normal vs Tuberculosis is confused. The UI warns accordingly.
- The prototype Usefulness Score depends only on the (Fact, Foil) class pair, not on the image.
- Only `clinician_ratings.csv` is written to disk (case hash, scores, rating; no image, name or key).

## Running in GitHub Codespaces
1. Push this folder to a GitHub repo (the `.devcontainer/` folder is included).
2. Repo -> Code -> Codespaces -> Create codespace on main. Dependencies install automatically (CPU PyTorch) and the app starts on port 8501.
3. When the "XAI Dashboard" port opens, upload `best_densenet121.pth` and paste your Gemini key in the sidebar.
`*.pth` is gitignored on purpose; upload weights through the sidebar instead of committing them.
