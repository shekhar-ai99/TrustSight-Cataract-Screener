import cv2
import numpy as np
from typing import Tuple, Optional


def _laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _mean_intensity(gray: np.ndarray) -> float:
    return float(gray.mean())


def _saturation_ratio(img_bgr: np.ndarray) -> float:
    # convert to HSV and compute fraction of pixels with saturation at max (>= 250/255)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    # consider saturated pixels as those with saturation >= 250
    saturated = (sat >= 250).sum()
    total = sat.size
    return float(saturated) / float(total) if total > 0 else 0.0


def check_image_quality(image_path: str) -> Tuple[str, Optional[dict]]:
    """
    Perform Image Quality Assessment (IQA) on the given image path.

    Returns:
      (status, reason)
      status: "PREDICT" or "REJECT"
      reason: None if PREDICT, otherwise dict with 'code' and 'metric'

    Rejection rules (Phase-2):
      - Laplacian variance < 50 -> LOW_BLUR
      - mean intensity < 40 or > 220 -> LOW_EXPOSURE
      - >15% pixels saturated -> HIGH_GLARE
    """
    img = cv2.imread(image_path)
    if img is None:
        return "REJECT", {"code": "FILE_NOT_FOUND", "metric": 0}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lv = _laplacian_variance(gray)
    if lv < 50.0:
        return "REJECT", {"code": "LOW_BLUR", "metric": lv}

    mean_i = _mean_intensity(gray)
    if mean_i < 40.0 or mean_i > 220.0:
        return "REJECT", {"code": "LOW_EXPOSURE", "metric": mean_i}

    sat_ratio = _saturation_ratio(img)
    if sat_ratio > 0.15:
        return "REJECT", {"code": "HIGH_GLARE", "metric": sat_ratio}

    return "PREDICT", None

