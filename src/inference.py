import json
import argparse
import os
from .utils import set_seed
from .model import CataractModel
from .preprocess import load_image_to_tensor
from .iqa import check_image_quality
from .gradcam import generate_gradcam
from schema.output_schema import InferenceOutputSchema
from .inference.mc_dropout import mc_inference
from .inference.decision import decide_action
from .utils import set_seed
import torch
import numpy as np
from pydantic import ValidationError


set_seed()


def infer(image_path: str, explain: bool = False, n_mc: int = 15, weights_path: str | None = None, seed: int = 42):
    """Main inference pipeline for Phase 2.

    - Runs IQA gate first and returns the exact JSON on IQA failure.
    - Runs MC Dropout for `n_mc` stochastic passes.
    - Computes mean, variance, confidence and decides action.
    - Validates final output against the strict Pydantic schema.
    """
    # Determinism
    set_seed(seed)

    # IQA gate (runs BEFORE inference)
    status, reason = check_image_quality(image_path)
    if status == "REJECT":
        # Per Phase 2 spec, return immediately this exact structure
        return json.dumps({"status": "REJECT", "reason": "LOW_IMAGE_QUALITY"})

    # Load model (CPU-only as required)
    model = CataractModel(weights_path=weights_path)
    model.to(torch.device("cpu"))
    model.eval()

    # Preprocess
    img_tensor = load_image_to_tensor(image_path)

    # MC Dropout sampling and statistics
    probs = mc_inference(model, img_tensor, n_mc=n_mc, seed=seed)
    probs = np.array(probs, dtype=np.float64)

    mean_prob = float(np.mean(probs)) if probs.size > 0 else 0.0
    var = float(np.var(probs, ddof=0)) if probs.size > 0 else 0.0
    # Normalize confidence: 1 - (var / max_bernoulli_var)
    confidence = max(0.0, 1.0 - (var / 0.25))

    # Decide action conservatively
    action = decide_action(mean_prob, var, confidence)

    # Build the strict schema payload
    payload = {
        "cataract_prob": round(float(mean_prob), 4),
        "confidence": round(float(confidence), 4),
        "action": action,
    }

    # Validate output strictly using Pydantic
    try:
        validated = InferenceOutputSchema(**payload)
    except ValidationError as e:
        # Hard failure if schema invalid
        raise

    # Optionally explain
    if explain:
        try:
            os.makedirs("outputs", exist_ok=True)
            gradcam_path = os.path.join("outputs", "gradcam_image.jpg")
            generate_gradcam(model, img_tensor, target_class=1, out_path=gradcam_path)
            with open(os.path.join("outputs", "result.json"), "w") as f:
                json.dump(validated.model_dump(), f)
        except Exception:
            pass

    return json.dumps(validated.model_dump())


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--n-mc", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = infer(args.image, explain=args.explain, n_mc=args.n_mc, weights_path=args.weights, seed=args.seed)
    print(out)


if __name__ == "__main__":
    _cli()
