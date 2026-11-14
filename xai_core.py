"""
xai_core.py - shared inference + explainability core for the Chest X-Ray XAI project.

Used by BOTH the Gradio dashboard (inside the Kaggle notebook) and the Streamlit app (local).
Modules (NFR9, modular design): model loading, Grad-CAM, occlusion baseline, deletion
faithfulness, usefulness score, end-to-end pipeline, guard-railed LLM explainer.

Clinical disclaimer: educational research prototype, not a diagnostic device.
"""
import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# ----------------------------------------------------------------------------
# Constants (identical to the training notebook)
# ----------------------------------------------------------------------------
IMAGE_SIZE: Tuple[int, int] = (224, 224)
CLASS_NAMES: List[str] = ['normal', 'pneumonia', 'tuberculosis']
IDX_TO_CLASS: Dict[int, str] = {i: n for i, n in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
LAMBDA_PENALTY = 0.15
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Transparent concept association prototype S(c_i, class) - not an automated knowledge graph.
CONCEPT_ASSOCIATION_TABLE = {
    'airspace_consolidation':   {'normal': -0.9, 'pneumonia': 0.95, 'tuberculosis': 0.30},
    'apical_cavitation':        {'normal': -1.0, 'pneumonia': 0.10, 'tuberculosis': 0.95},
    'clear_lung_fields':        {'normal': 0.95, 'pneumonia': -0.9, 'tuberculosis': -0.9},
    'pleural_effusion':         {'normal': -0.8, 'pneumonia': 0.50, 'tuberculosis': 0.60},
    'bronchovascular_markings': {'normal': 0.80, 'pneumonia': -0.2, 'tuberculosis': -0.3},
}


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_densenet121(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    net = models.densenet121(weights=weights)
    net.classifier = nn.Linear(net.classifier.in_features, num_classes)
    return net


def _extract_state_dict(obj) -> Dict:
    """Accept a plain state_dict or a common checkpoint wrapper; strip 'module.' prefixes."""
    if isinstance(obj, nn.Module):
        raise ValueError("This file holds a full pickled model. Please upload a state_dict "
                         "(torch.save(model.state_dict(), ...)).")
    if isinstance(obj, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    if not isinstance(obj, dict) or not obj:
        raise ValueError("Could not find a state_dict inside the uploaded .pth file.")
    return {(k[7:] if k.startswith("module.") else k): v for k, v in obj.items()}


def model_from_state(state_obj, device: torch.device) -> nn.Module:
    """Build DenseNet-121 (3-class head) and load a state_dict; clear ValueError on mismatch."""
    state = _extract_state_dict(state_obj)
    model = build_densenet121(NUM_CLASSES, pretrained=False)
    try:
        model.load_state_dict(state)
    except Exception as e:
        raise ValueError(
            "These weights do not match DenseNet-121 with a 3-class head "
            f"(classes: {', '.join(CLASS_NAMES)}). Details: {str(e)[:200]}")
    return model.to(device).eval()


def load_model_from_bytes(weights_bytes: bytes, device: torch.device) -> nn.Module:
    """Load weights straight from uploaded bytes - nothing is written to disk.
    weights_only=True blocks arbitrary-code execution from untrusted pickle files."""
    import io
    try:
        obj = torch.load(io.BytesIO(weights_bytes), map_location=device, weights_only=True)
    except Exception as e:
        raise ValueError(f"Could not read the .pth file (is it a PyTorch state_dict?): {str(e)[:200]}")
    return model_from_state(obj, device)


def read_temperature(path: str, default: float = 1.0) -> float:
    try:
        with open(path) as f:
            return float(json.load(f)["temperature"])
    except Exception:
        return default


def load_model(artifact_dir: str, device: torch.device) -> Tuple[nn.Module, float]:
    """Load trained weights + temperature from an artifact folder (FileNotFoundError if absent)."""
    wpath = os.path.join(artifact_dir, 'best_densenet121.pth')
    if not os.path.exists(wpath):
        raise FileNotFoundError(wpath)
    obj = torch.load(wpath, map_location=device, weights_only=True)
    model = model_from_state(obj, device)
    return model, read_temperature(os.path.join(artifact_dir, 'calibration_params.json'))


# ----------------------------------------------------------------------------
# Grad-CAM (target: model.features.norm5, as in the original notebook)
# ----------------------------------------------------------------------------
class DenseNetGradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        # NOTE: no clone on purpose - DenseNet applies an in-place ReLU right after norm5,
        # so these are the post-ReLU feature maps (standard Grad-CAM practice for DenseNet).
        self.activations = out.detach()
        if out.requires_grad:  # skipped automatically inside torch.no_grad()
            out.register_hook(lambda g: setattr(self, 'gradients', g.detach()))

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()
        with torch.enable_grad():
            output = self.model(input_tensor)
            output[0, class_idx].backward()
        pooled = torch.mean(self.gradients, dim=[0, 2, 3])
        acts = self.activations[0].clone()
        acts = acts * pooled[:, None, None]
        hmap = torch.sum(acts, dim=0).cpu().numpy()
        hmap = np.maximum(hmap, 0)
        mx = np.max(hmap)
        hmap = hmap / mx if mx > 0 else np.zeros_like(hmap)
        return cv2.resize(hmap.astype(np.float32), IMAGE_SIZE)

    def remove_hooks(self):
        self.handle.remove()


# ----------------------------------------------------------------------------
# Baseline XAI: occlusion sensitivity (batched; same maths as the original loop)
# ----------------------------------------------------------------------------
@torch.no_grad()
def compute_occlusion_sensitivity(model, input_tensor, class_idx, patch_size=32, stride=16,
                                  batch_size=64) -> np.ndarray:
    model.eval()
    base_prob = torch.softmax(model(input_tensor), dim=1)[0, class_idx].item()
    _, _, H, W = input_tensor.shape
    hmap = np.zeros((H, W))
    counts = np.zeros((H, W))
    coords = [(y, x) for y in range(0, H - patch_size + 1, stride)
              for x in range(0, W - patch_size + 1, stride)]
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        batch = input_tensor.repeat(len(chunk), 1, 1, 1)
        for j, (y, x) in enumerate(chunk):
            batch[j, :, y:y + patch_size, x:x + patch_size] = 0.0
        probs = torch.softmax(model(batch), dim=1)[:, class_idx].cpu().numpy()
        for (y, x), p in zip(chunk, probs):
            hmap[y:y + patch_size, x:x + patch_size] += base_prob - p
            counts[y:y + patch_size, x:x + patch_size] += 1
    counts[counts == 0] = 1.0
    hmap = np.maximum(hmap / counts, 0)
    if hmap.max() > 0:
        hmap /= hmap.max()
    return hmap


# ----------------------------------------------------------------------------
# Perturbation faithfulness: probability drop after deleting the top-k% attributed pixels
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_deletion_faithfulness(model, input_tensor, class_idx, attribution_map,
                                   fractions=(0.1, 0.2, 0.3)) -> List[float]:
    model.eval()
    init_prob = torch.softmax(model(input_tensor), dim=1)[0, class_idx].item()
    flat = attribution_map.flatten()
    order = np.argsort(flat)[::-1]
    drops = []
    for f in fractions:
        mask = np.ones_like(flat)
        mask[order[:int(f * len(flat))]] = 0.0
        m = torch.tensor(mask.reshape(attribution_map.shape), dtype=torch.float,
                         device=input_tensor.device)[None, None]
        new_prob = torch.softmax(model(input_tensor * m), dim=1)[0, class_idx].item()
        drops.append(init_prob - new_prob)
    return drops


# ----------------------------------------------------------------------------
# Explanation Usefulness Score:  US = CS - lambda * C,  CS = sum_i [S(c_i,Fact) - S(c_i,Foil)]
# ----------------------------------------------------------------------------
def calculate_usefulness_score(fact_class: str, foil_class: str,
                               concept_table: Dict = CONCEPT_ASSOCIATION_TABLE,
                               lambda_penalty: float = LAMBDA_PENALTY) -> Dict:
    c_count = len(concept_table)
    breakdown, total_cs = {}, 0.0
    for c, w in concept_table.items():
        s_fact, s_foil = w.get(fact_class.lower(), 0.0), w.get(foil_class.lower(), 0.0)
        breakdown[c] = {'S_fact': s_fact, 'S_foil': s_foil, 'diff': s_fact - s_foil}
        total_cs += s_fact - s_foil
    penalty = lambda_penalty * c_count
    us = total_cs - penalty

    # NFR6: normalised 0-1 score = min-max over every possible (Fact, Foil) pair
    pair_scores = []
    for a in CLASS_NAMES:
        for b in CLASS_NAMES:
            if a != b:
                pair_scores.append(sum(w.get(a, 0.0) - w.get(b, 0.0) for w in concept_table.values())
                                   - lambda_penalty * c_count)
    lo, hi = min(pair_scores), max(pair_scores)
    us_norm = (us - lo) / (hi - lo) if hi > lo else 0.5

    return {
        'fact_class': fact_class, 'foil_class': foil_class,
        'contrastive_score': round(total_cs, 4), 'complexity_penalty': round(penalty, 4),
        'usefulness_score': round(us, 4), 'usefulness_normalized': round(float(us_norm), 4),
        'num_concepts': c_count, 'lambda': lambda_penalty, 'concept_breakdown': breakdown,
    }


# ----------------------------------------------------------------------------
# End-to-end pipeline
# ----------------------------------------------------------------------------
def load_pil_image(image_input) -> Image.Image:
    """Validate + load an image (NFR8: handle invalid images)."""
    try:
        if isinstance(image_input, (str, os.PathLike)):
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input)
        else:
            img = Image.open(image_input)  # file-like (Streamlit upload)
        img = img.convert('RGB')
    except Exception as e:
        raise ValueError(f"Could not read the uploaded file as an image: {e}")
    if min(img.size) < 64:
        raise ValueError(f"Image too small ({img.size[0]}x{img.size[1]}); need at least 64x64 pixels.")
    return img


def overlay_heatmap(img_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_np, 1 - alpha, colored, alpha, 0)


def predict_and_explain(image_input, model: nn.Module, device: torch.device, temperature: float = 1.0,
                        concept_table: Dict = CONCEPT_ASSOCIATION_TABLE,
                        lambda_penalty: float = LAMBDA_PENALTY, run_occlusion: bool = True) -> Dict:
    pil_image = load_pil_image(image_input)
    tensor_in = EVAL_TRANSFORM(pil_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        raw_logits = model(tensor_in)
        raw_probs = torch.softmax(raw_logits, dim=1)[0].cpu().numpy()
        cal_probs = torch.softmax(raw_logits / temperature, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(cal_probs))
    foil_idx = int(np.argsort(cal_probs)[-2])
    fact_class, foil_class = IDX_TO_CLASS[pred_idx], IDX_TO_CLASS[foil_idx]

    cam = DenseNetGradCAM(model, model.features.norm5)
    try:
        heatmap = cam.generate_heatmap(tensor_in, pred_idx)
    finally:
        cam.remove_hooks()

    img_np = np.array(pil_image.resize(IMAGE_SIZE))
    overlay = overlay_heatmap(img_np, heatmap)

    gcam_drops = evaluate_deletion_faithfulness(model, tensor_in, pred_idx, heatmap)
    occ_map, occ_overlay, occ_drops = None, None, None
    if run_occlusion:
        occ_map = compute_occlusion_sensitivity(model, tensor_in, pred_idx)
        occ_overlay = overlay_heatmap(img_np, occ_map)
        occ_drops = evaluate_deletion_faithfulness(model, tensor_in, pred_idx, occ_map)

    return {
        'fact_class': fact_class, 'confidence': float(cal_probs[pred_idx]),
        'raw_confidence': float(raw_probs[pred_idx]),
        'all_probabilities': {IDX_TO_CLASS[i]: float(cal_probs[i]) for i in range(NUM_CLASSES)},
        'foil_class': foil_class, 'foil_confidence': float(cal_probs[foil_idx]),
        'heatmap': heatmap, 'overlay': overlay, 'original_image': img_np,
        'occlusion_map': occ_map, 'occlusion_overlay': occ_overlay,
        'gradcam_drops': gcam_drops, 'occlusion_drops': occ_drops,
        'usefulness': calculate_usefulness_score(fact_class, foil_class, concept_table, lambda_penalty),
    }


# ----------------------------------------------------------------------------
# Modular, guard-railed LLM explainer (Google Gemini) with deterministic fallback
# ----------------------------------------------------------------------------
def get_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Order: explicit argument -> env var -> Kaggle Secret (label GEMINI_API_KEY or GOOGLE_API_KEY)."""
    if explicit and explicit.strip():
        return explicit.strip()
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            try:
                return client.get_secret(name)
            except Exception:
                continue
    except Exception:
        pass
    return None


def _fallback_summary(r: Dict, note: str = "") -> str:
    fact, foil = r['fact_class'].capitalize(), r['foil_class'].capitalize()
    conf, foil_conf = r['confidence'] * 100, r['foil_confidence'] * 100
    us, cs = r['usefulness']['usefulness_score'], r['usefulness']['contrastive_score']
    return (
        f"{note}"
        f"### Model Decision Support Summary (Non-Diagnostic Prototype)\n\n"
        f"- **Primary Hypothesis (Fact)**: **{fact}** (Calibrated Confidence: `{conf:.1f}%`)\n"
        f"- **Closest Competing Differential (Foil)**: **{foil}** (Confidence: `{foil_conf:.1f}%`)\n"
        f"- **Contrastive Concept Differential**: `{cs:+.2f}` | **Usefulness Score**: `{us:+.2f}`\n\n"
        f"**Radiological Context & Attribution Notice**:\n"
        f"The visual overlay indicates anatomical zones in the thoracic cage that drove the DenseNet-121 "
        f"feature activations toward '{fact}'. This attribution reflects model sensitivity and does not "
        f"confirm the presence of active pathology. Statistical confidence scores must not be interpreted as "
        f"calibrated disease likelihoods.\n\n"
        f"⚠️ **Mandatory Clinical Notice**: This output is strictly for research and education. "
        f"Formal diagnosis requires comprehensive clinical correlation and review by a board-certified radiologist."
    )


def generate_llm_explanation(prediction_results: Dict, api_key: Optional[str] = None,
                             model_name: str = "gemini-3.5-flash-lite") -> str:
    key = get_api_key(api_key)
    if not key:
        return _fallback_summary(prediction_results)
    fact = prediction_results['fact_class'].capitalize()
    foil = prediction_results['foil_class'].capitalize()
    conf = prediction_results['confidence'] * 100
    foil_conf = prediction_results['foil_confidence'] * 100
    us_d = prediction_results['usefulness']
    top = sorted(us_d['concept_breakdown'].items(), key=lambda kv: -abs(kv[1]['diff']))[:3]
    concepts = "; ".join(f"{c.replace('_', ' ')} (differential {v['diff']:+.2f})" for c, v in top)
    drops = prediction_results.get('gradcam_drops') or []
    drops_txt = ", ".join(f"{d:.2f}" for d in drops) if drops else "n/a"
    try:
        from google import genai
        from google.genai import types
        # Bound each Gemini request and avoid long SDK retry chains in the UI.
        http_options = types.HttpOptions(
            timeout=45_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        client = genai.Client(api_key=key, http_options=http_options)
        prompt = f"""
You are an AI radiological research assistant acting as an educational decision aid for a clinician.
Summarize ONLY these model outputs (do not invent findings):
- Primary prediction (Fact): {fact} (calibrated confidence {conf:.1f}%)
- Competing differential (Foil): {foil} (confidence {foil_conf:.1f}%)
- Contrastive score: {us_d['contrastive_score']:.2f}; Usefulness score: {us_d['usefulness_score']:.2f}
  (normalised 0-1: {us_d['usefulness_normalized']:.2f})
- Most discriminating concepts between Fact and Foil: {concepts}
- Grad-CAM deletion faithfulness (probability drop after removing top 10/20/30% pixels): {drops_txt}

Guardrails:
1. Never state or imply that this confirms a diagnosis.
2. Explicitly say the heatmap reflects model sensitivity, not verified pathology.
3. Note that confidence scores are statistical estimates, not disease probabilities.
4. If confidence is below 70% or Fact/Foil are close, say the result is uncertain.
5. Recommend mandatory review by a physician/radiologist.
6. Maximum 150 words, plain language, no markdown headings.
"""
        resp = client.models.generate_content(model=model_name, contents=prompt)
        text = (resp.text or "").strip()
        if not text:
            raise RuntimeError("empty response")
        return "### AI-generated explanation (Gemini)\n\n" + text + \
               "\n\n⚠️ *Research prototype - not a diagnosis. Physician review required.*"
    except Exception as e:
        return _fallback_summary(prediction_results,
                                 note=f"> ⚠️ LLM unavailable ({type(e).__name__}: {str(e)[:120]}). "
                                      f"Showing deterministic summary instead.\n\n")


def test_gemini_key(api_key: str, model_name: str = "gemini-3.5-flash-lite") -> Tuple[bool, str]:
    """Tiny call to check a key. Returns (ok, message). The key is never logged or stored."""
    if not api_key or not api_key.strip():
        return False, "No key entered."
    try:
        from google import genai
        from google.genai import types
        # Keep the Streamlit button responsive: one attempt, 15-second timeout.
        http_options = types.HttpOptions(
            timeout=45_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        client = genai.Client(api_key=api_key.strip(), http_options=http_options)
        resp = client.models.generate_content(
            model=model_name,
            contents="Reply with the single word: OK",
            config=types.GenerateContentConfig(max_output_tokens=8),
        )
        return (True, "Key works.") if (resp.text or "").strip() else (False, "Empty response from Gemini.")
    except Exception as e:
        # google-genai APIError exposes an HTTP status in .code and a safe
        # service message in .message. Surface that detail so users can tell
        # authentication, model access, quota, and service/network failures apart.
        code = getattr(e, "code", None)
        detail = str(getattr(e, "message", "") or str(e)).strip()
        detail = detail.replace(api_key.strip(), "[REDACTED]") if api_key and api_key.strip() else detail
        detail = " ".join(detail.split())[:240]
        try:
            code = int(code) if code is not None else None
        except (TypeError, ValueError):
            code = None

        if code == 400 or any(t in detail.lower() for t in ("api_key_invalid", "api key not valid")):
            return False, "Google rejected this API key (HTTP 400). Create or copy a valid Gemini API key in Google AI Studio."
        if code == 401:
            return False, "Gemini authentication failed (HTTP 401). Check that this is a Gemini Developer API key, copied completely, and not a Vertex/Agent Platform credential."
        if code == 403:
            return False, "Gemini denied access (HTTP 403). Check that the key is enabled for the Gemini API and that the project/API restrictions allow generativelanguage.googleapis.com."
        if code == 404:
            return False, f"Gemini model is unavailable to this project (HTTP 404). The app now tests {model_name}; verify model access in Google AI Studio. {detail}"
        if code == 429:
            return False, "Gemini rate limit or quota reached (HTTP 429). Check the project's AI Studio rate limits and billing."
        if code == 503:
            return False, "Gemini is temporarily overloaded (HTTP 503). Retry later; this is not evidence that the API key is invalid."
        if code is not None:
            return False, f"Gemini API returned HTTP {code}: {detail or type(e).__name__}"
        return False, f"Gemini connection failed ({type(e).__name__}): {detail or 'Check Codespaces network access and retry.'}"
