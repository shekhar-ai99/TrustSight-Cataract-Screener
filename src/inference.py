import json
import argparse
import os
from .utils import set_seed
from .model import CataractModel
from .preprocess import load_image_to_tensor
from .iqa import check_image_quality
from .gradcam import generate_gradcam
from .schema import InferenceOutput
import torch
import numpy as np
from pydantic import ValidationError


set_seed()

# OOD rejection threshold (Phase 3 hardening)
OOD_CONF_THRESHOLD = 0.60


def _compute_confidence_from_probs(probs: np.ndarray):
    # probs: np.array of shape (N,)
    mean = float(probs.mean())
    var = float(probs.var(ddof=0))
    # Normalize confidence using max Bernoulli variance 0.25
    confidence = max(0.0, 1.0 - (var / 0.25))
    return mean, var, confidence


def infer(image_path: str, explain: bool = False, n_mc: int = 15, weights_path: str | None = None):
    # IQA gate (runs BEFORE inference)
    status, reason = check_image_quality(image_path)
    if status == "REJECT":
        out = InferenceOutput(status="REJECT", cataract_prob=None, confidence=None, action="REJECT", reason=reason)
        return json.dumps(out.model_dump())

    # Load model (CPU-only)
    model = CataractModel(weights_path=weights_path)
    model.to(torch.device("cpu"))

    # Preprocess
    img_tensor = load_image_to_tensor(image_path)

    # MC Dropout sampling (use model.predict_proba which uses safe_mc_forward)
    probs = model.predict_proba(img_tensor, n_mc=n_mc)
    probs = np.array(probs, dtype=np.float64)

    mean_prob, var, confidence = _compute_confidence_from_probs(probs)

    # OOD confidence-based rejection: if the maximum sampled probability
    # across MC runs is below a conservative threshold, treat as OOD and reject.
    max_prob = float(np.max(probs)) if probs.size > 0 else 0.0
    if max_prob < OOD_CONF_THRESHOLD:
        reason_out = {"code": "low_confidence_ood", "max_prob": float(round(max_prob, 4))}
        err = InferenceOutput(status="REJECT", cataract_prob=None, confidence=float(round(confidence, 4)), action="REJECT", reason=reason_out)
        return json.dumps(err.model_dump())

    # Action policy
    if confidence >= 0.8:
        action = "PREDICT"
    elif 0.5 <= confidence < 0.8:
        action = "REFER_TO_SPECIALIST"
    else:
        action = "REJECT"

    status_out = "PREDICT" if action != "REJECT" else "REJECT"

    reason_out = None
    if action == "REJECT":
        reason_out = {"code": "LOW_CONFIDENCE", "confidence": float(confidence), "variance": float(var)}

    result = InferenceOutput(
        status=status_out,
        cataract_prob=(round(float(mean_prob), 4) if action != "REJECT" else None),
        confidence=(round(float(confidence), 4) if action != "REJECT" else None),
        action=action,
        reason=reason_out,
    )

    # Explainability: generate Grad-CAM deterministically
    if explain:
        try:
            model.eval()
            os.makedirs("outputs", exist_ok=True)
            gradcam_path = os.path.join("outputs", "gradcam_image.jpg")
            generate_gradcam(model, img_tensor, target_class=1, out_path=gradcam_path)
            with open(os.path.join("outputs", "result.json"), "w") as f:
                json.dump(result.model_dump(), f)
        except Exception:
            pass

    # Validate and return JSON
    try:
        # model_dump returns a dict that is JSON serializable
        validated = result.model_dump()
        return json.dumps(validated)
    except ValidationError as e:
        # In the unlikely event validation fails, return explicit rejection
        err = InferenceOutput(status="REJECT", cataract_prob=None, confidence=None, action="REJECT", reason={"code": "SCHEMA_VALIDATION_ERROR", "detail": str(e)})
        return json.dumps(err.model_dump())


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--n-mc", type=int, default=15)
    args = parser.parse_args()
    out = infer(args.image, explain=args.explain, n_mc=args.n_mc, weights_path=args.weights)
    print(out)


if __name__ == "__main__":
    _cli()
