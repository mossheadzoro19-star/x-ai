# HANDOFF — Build the Streamlit clinical-XAI dashboard (Team 19 SDP)

You are picking up a university project (KLE Tech, "Clinical-facing confidence and explainability interfaces for AI-assisted decision making"). Training is DONE. Your job: **finish, harden and polish a Streamlit dashboard**. A working first draft already exists (`app.py` + `xai_core.py` in this folder) — build on it, do not start from scratch.

## 1. What the dashboard must do
User uploads a chest X-ray (JPG/PNG) and gets, in one screen:
1. **Diagnosis** — predicted class (Normal / Pneumonia / Tuberculosis) from the trained DenseNet-121.
2. **Confidence score** — temperature-calibrated softmax, shown as 0–100 %.
3. **Heatmap** — Grad-CAM overlay next to the original X-ray (occlusion baseline optional).
4. **Explanation Usefulness Score** — raw US and normalised 0–1, with a one-line interpretation.
5. **LLM explanation** — a short guard-railed plain-language explanation from an LLM called with a user-supplied API key.
Plus a clinician-validation tab (rate usefulness 1–5, compare with automated score) and a model-performance tab.

## 2. Files in this bundle
| File | Status |
|---|---|
| `xai_core.py` | Shared backend: model loading, Grad-CAM, occlusion, deletion-faithfulness, usefulness score, `predict_and_explain()`, `generate_llm_explanation()`, `get_api_key()`. **Tested on CPU** (see §7). Keep UI-free. |
| `app.py` | Streamlit UI (3 tabs). Launches with no errors in the empty state (Streamlit `AppTest`). The **upload → results path has not been exercised inside Streamlit** — test it. |
| `requirements.txt` | deps |
| `HANDOFF.md` | this file |

**Model artifacts are NOT in this bundle** — ask the user for `chest_xray_artifacts.zip` (output of their Kaggle training notebook) and put these next to `app.py`:
`best_densenet121.pth`, `calibration_params.json` (temperature T ≈ 1.284), optionally `metrics.json`, `training_curves.png`, `eval_plots.png`, `faithfulness_comparison.png` (the Performance tab shows them if present).
If the weights are missing, the app should show a clear error, not a stack trace (add this — currently `get_model()` will raise).

## 3. Model facts you must respect
- DenseNet-121, ImageNet-pretrained, head `Linear(1024, 3)`. Input 224×224 RGB, ImageNet mean/std (already in `xai_core.EVAL_TRANSFORM`). **Do not change preprocessing.**
- Class order is fixed: `['normal', 'pneumonia', 'tuberculosis']` (index 0,1,2).
- Calibration: `softmax(logits / T)`, T from `calibration_params.json`.
- Grad-CAM target layer: `model.features.norm5`. Implementation verified bit-identical to the original notebook's.

## 4. IMPORTANT honesty constraints (surface these in the UI)
- **Measured test accuracy is 73.69 %** (macro-F1 0.747, OvR AUC 0.921), below the project's 85 % target. Pneumonia is reliable (recall 0.99); **Normal ↔ Tuberculosis is confused** (Normal recall 0.59, TB recall 0.73). Root cause under investigation (a diagnostic notebook is with the user); weights may be replaced later — so load weights from files, never hard-code.
- Show a visible **low-confidence warning** (< 70 %) and, if the top-2 probabilities are close, an "uncertain" banner. Consider an extra caution note when the prediction is Normal or TB.
- Permanent disclaimer: educational research prototype, not a diagnostic device; heatmaps show model sensitivity, not verified pathology; clinician keeps final authority.
- The prototype Usefulness Score **depends only on the (Fact, Foil) class pair**, not on the image or heatmap (concept table is a transparent hand-written prototype in `xai_core.CONCEPT_ASSOCIATION_TABLE`, λ = 0.15, US = CS − λ·C). Say so in the UI's "how to read this" expander; do not present it as image-specific clinical truth. (A future improvement is making it image-dependent; out of scope unless the user asks.)

## 5. LLM integration (API key)
- Provider: **Google Gemini** via `google-genai`, model `gemini-2.5-flash` (already implemented in `generate_llm_explanation`).
- Key sources, in order: sidebar password field → env `GEMINI_API_KEY` / `GOOGLE_API_KEY` → Kaggle secret. **Never hard-code or log the key; never write it to disk or to the ratings CSV.** Use `st.secrets` as an additional source if you add `.streamlit/secrets.toml` (gitignored).
- The LLM receives ONLY model outputs (class, confidences, usefulness numbers, top concepts, deletion-faithfulness drops) — **never the image**. Guardrails in the prompt: no diagnosis claims, heatmap ≠ pathology, confidence ≠ disease probability, recommend physician review, ≤150 words. Keep these.
- If the key is missing/invalid/quota-hit, the code falls back to a deterministic safe summary and prints a visible note. Keep that behaviour; add a "Test key" button that makes a tiny call and reports OK/invalid.
- Optional nice-to-have: make the provider pluggable (e.g. add Anthropic/OpenAI behind the same `generate_llm_explanation` signature) — only if the user asks.

## 6. Project requirements to satisfy (from the team's PPT)
Functional: FR1 predict via DenseNet-121 · FR2 validate/resize/normalise upload · FR3 calibrated confidence · FR4 Grad-CAM · FR5 usefulness score · FR6 perturbation faithfulness · FR7 interface shows prediction + confidence + heatmap + usefulness · FR8 clinician validation vs automated score.
Non-functional: NFR1 results within ≤3 user interactions · NFR2 prediction + explanation in **<10 s** (occlusion baseline is slow on CPU — default it OFF on CPU, ON on GPU, and show a spinner) · NFR4 confidence shown 0–100 % · NFR6 usefulness normalised 0–1 · NFR7 protect medical images (process in memory, never save uploads; no PHI in logs) · NFR8 handle invalid images gracefully (`load_pil_image` raises `ValueError` with a clear message — keep showing it via `st.error`) · NFR9 modular (UI separate from `xai_core`) · NFR10 reproducibility notes.
UX (PPT slide): simple upload; predicted class and confidence % clearly; original X-ray **alongside** its Grad-CAM; usefulness score with brief interpretation; clear errors for invalid images; clinician retains decision authority.

## 7. What was verified (so you don't redo it)
- `xai_core.py` on CPU with random weights + synthetic image: end-to-end `predict_and_explain` OK; batched occlusion matches the original loop (max diff 8e-5); Grad-CAM identical to original implementation (diff 0.0); works after `torch.no_grad()` forwards; invalid/tiny/non-image inputs raise `ValueError`; LLM fallback path works.
- Clinician-rating CSV logic and Spearman (with NaN guard for no variation) tested via the earlier Gradio version; the same logic is in `app.py`.
- `app.py` loads under `streamlit.testing.AppTest` with no exceptions (empty state).
- NOT verified: real trained weights, a real X-ray, live Gemini call, file-upload flow in Streamlit, GPU, the Performance tab with real artifacts.

## 8. Your task list
1. Get artifacts from the user; run `streamlit run app.py`; test with 3–4 real X-rays (one per class + an invalid file + a tiny image).
2. Fix anything that breaks in the upload→results path. Add the friendly missing-weights error.
3. UX polish per §6: single-screen result layout, clear confidence meter, colour-coded low-confidence/uncertain banners, "how to read this" expander (heatmap = model sensitivity; US limitation in §4), disclaimer pinned in sidebar.
4. LLM: sidebar key box + "Test key" button, spinner, graceful failure; verify with a real key supplied by the user (never paste it into files).
5. Performance: cache the model (`st.cache_resource`, present), measure end-to-end latency, keep <10 s on GPU; on CPU default occlusion off. Add a download button for a PDF/PNG report of the result (original, overlay, scores, LLM text, disclaimer) — no patient identifiers.
6. Clinician-validation tab: keep CSV logging (no images, no key); consider adding the case's filename hash instead of name.
7. Write a short README (install, run, where to put artifacts, how to set the key, screenshots) and, if asked, a `Dockerfile`.

## 9. Acceptance checklist
- [ ] Upload X-ray → class, calibrated confidence %, original + Grad-CAM side by side, raw + normalised US, LLM text — all on one screen.
- [ ] Works with no API key (fallback) and with a valid key (Gemini text shown); invalid key shows a clear message.
- [ ] Invalid / corrupt / tiny image → friendly error, app stays alive.
- [ ] Low-confidence and close-call warnings appear; disclaimer always visible.
- [ ] No uploads or keys written to disk or logs.
- [ ] End-to-end latency measured and reported (GPU and CPU).
- [ ] README + requirements.txt up to date.

## 10. User context
User: Tejas M (Team 19, with Ria Javalagi, Maitri Patil, Varun S Terdal; guide Dr. Suvarna G K). Wants the dashboard "to run" the final project demo. Ask before expanding scope; keep answers short and status-oriented.
