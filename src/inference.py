import json
import argparse
import os
import math
from .utils import set_seed
from .model import CataractModel
from .preprocess import load_image_to_tensor
from .iqa import check_image_quality
from .gradcam import generate_gradcam
import torch


set_seed()


def _compute_confidence_from_probs(probs):
    # probs: np.array of shape (N,)
    mean = float(probs.mean())
    var = float(probs.var(ddof=0))
    # normalize variance by max possible Bernoulli variance mean*(1-mean)
    denom = mean * (1.0 - mean)
    if denom <= 0:
        normalized_var = 0.0
    else:
        normalized_var = var / (denom + 1e-12)
    normalized_var = max(0.0, min(1.0, normalized_var))
    confidence = 1.0 - normalized_var
    return mean, var, float(normalized_var), float(confidence)


def infer(image_path: str, explain: bool = False, n_mc: int = 20, weights_path: str | None = None):
    # IQA gate
    if not check_image_quality(image_path):
        return json.dumps({
            "status": "REJECT",
            "reason": "LOW_IMAGE_QUALITY"
        })

    # Load model (CPU-only)
    model = CataractModel(weights_path=weights_path)
    model.to(torch.device("cpu"))

    # Preprocess
    img_tensor = load_image_to_tensor(image_path)

    # MC Dropout sampling
    probs = []
    model.eval()
    # Enable dropout layers manually when doing MC
    for i in range(n_mc):
        model.enable_mc_dropout()
        with torch.no_grad():
            out = model(img_tensor, mc=True)
            p = torch.sigmoid(out).squeeze().cpu().item()
            probs.append(p)

    import numpy as np
    probs = np.array(probs, dtype=np.float64)
    mean_prob, var, norm_var, confidence = _compute_confidence_from_probs(probs)

    result = {
        "status": "PREDICT",
        "cataract_prob": round(float(mean_prob), 4),
        "confidence": round(float(confidence), 4),
        "uncertainty": round(float(norm_var), 4),
        "action": "PREDICT"
    }

    # Explainability
    if explain:
        # For Grad-CAM generate using deterministic forward (dropout disabled)
        model.disable_mc_dropout()
        # generate_gradcam expects tensor on cpu
        gradcam_path = os.path.join("outputs", "gradcam_image.jpg")
        os.makedirs("outputs", exist_ok=True)
        try:
            generate_gradcam(model, img_tensor, target_class=1, out_path=gradcam_path)
        except Exception:
            # Non-fatal: continue and return prediction
            pass
        # Save result json
        with open(os.path.join("outputs", "result.json"), "w") as f:
            json.dump(result, f)

    return json.dumps(result)


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()
    out = infer(args.image, explain=args.explain, weights_path=args.weights)
    print(out)


if __name__ == "__main__":
    _cli()
