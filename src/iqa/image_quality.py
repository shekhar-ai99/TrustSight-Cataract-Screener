"""Image Quality Assessment (IQA) gate.

Implements three deterministic checks:
- Blur: Laplacian variance (<50 fails)
- Exposure: mean intensity (<40 or >220 fails)
- Glare: percent of pixels >240 (>15% fails)

API:
    check_image_quality(image_path: str) -> tuple[str, None|str]
Returns ("REJECT", "LOW_IMAGE_QUALITY") on failure, or ("OK", None) on pass.
"""
from typing import Tuple
import cv2
import numpy as np


def _read_gray(image_path: str) -> np.ndarray:
    # Read as grayscale; raise on failure to load
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return gray


def check_image_quality(image_path: str) -> Tuple[str, str | None]:
    """Run IQA checks. Return tuple (status, reason).

    On any failure return ("REJECT", "LOW_IMAGE_QUALITY").
    Otherwise return ("OK", None).
    """
    gray = _read_gray(image_path)

    # Ensure uint8
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Blur check: Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var_lap = float(lap.var())
    if var_lap < 50.0:
        return "REJECT", "LOW_IMAGE_QUALITY"

    # Exposure: mean pixel intensity
    mean_int = float(gray.mean())
    if mean_int < 40.0 or mean_int > 220.0:
        return "REJECT", "LOW_IMAGE_QUALITY"

    # Glare: percent of saturated pixels (>240)
    sat_pct = float((gray > 240).sum()) / float(gray.size) * 100.0
    if sat_pct > 15.0:
        return "REJECT", "LOW_IMAGE_QUALITY"

    return "OK", None
