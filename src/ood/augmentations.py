"""Deterministic OOD-style augmentations for evaluation only.

Provides functions to apply various realistic corruptions: gaussian blur,
additive noise, jpeg compression, brightness/contrast, partial occlusion.
All functions accept a `seed` for determinism and do not change labels.
"""
from typing import Tuple
import numpy as np
import cv2


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def gaussian_blur(img: np.ndarray, sigma: float = 1.5, seed: int = 0) -> np.ndarray:
    """Apply Gaussian blur with deterministic kernel size derived from sigma."""
    rng = np.random.default_rng(seed)
    k = max(3, int(2 * round(sigma * 3) + 1))
    blurred = cv2.GaussianBlur(_ensure_uint8(img), (k, k), sigmaX=sigma)
    return blurred


def additive_noise(img: np.ndarray, sigma: float = 10.0, seed: int = 0) -> np.ndarray:
    """Add Gaussian noise (mean 0) with given sigma, deterministic by seed."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=img.shape)
    out = img.astype(np.float32) + noise
    return _ensure_uint8(out)


def jpeg_compression(img: np.ndarray, quality: int = 30, seed: int = 0) -> np.ndarray:
    """Simulate JPEG compression artifacts by re-encoding at low quality."""
    enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    _, encimg = cv2.imencode('.jpg', _ensure_uint8(img), enc_param)
    decimg = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)
    return _ensure_uint8(decimg)


def brightness_contrast(img: np.ndarray, brightness: float = 0.0, contrast: float = 1.0, seed: int = 0) -> np.ndarray:
    """Adjust brightness and contrast: out = img*contrast + brightness."""
    out = img.astype(np.float32) * float(contrast) + float(brightness)
    return _ensure_uint8(out)


def partial_occlusion(img: np.ndarray, occlusion_area: float = 0.1, seed: int = 0) -> np.ndarray:
    """Apply a deterministic rectangular occlusion covering occlusion_area fraction of image."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    area = h * w
    occ_pixels = int(area * float(occlusion_area))
    # make rectangle aspect ratio ~1
    side = int(np.sqrt(max(1, occ_pixels)))
    top = rng.integers(0, max(1, h - side))
    left = rng.integers(0, max(1, w - side))
    out = img.copy()
    out[top:top+side, left:left+side] = 0
    return _ensure_uint8(out)


def apply_augmentations(img: np.ndarray, seed: int = 0, config: dict | None = None) -> np.ndarray:
    """Apply a sequence of augmentations deterministically based on config.

    Config keys (all optional): gaussian_sigma, noise_sigma, jpeg_quality,
    brightness, contrast, occlusion_area. Any missing values are skipped.
    """
    cfg = config or {}
    out = img.copy()
    if 'gaussian_sigma' in cfg:
        out = gaussian_blur(out, sigma=cfg['gaussian_sigma'], seed=seed+1)
    if 'noise_sigma' in cfg:
        out = additive_noise(out, sigma=cfg['noise_sigma'], seed=seed+2)
    if 'jpeg_quality' in cfg:
        out = jpeg_compression(out, quality=cfg['jpeg_quality'], seed=seed+3)
    if 'brightness' in cfg or 'contrast' in cfg:
        b = cfg.get('brightness', 0.0)
        c = cfg.get('contrast', 1.0)
        out = brightness_contrast(out, brightness=b, contrast=c, seed=seed+4)
    if 'occlusion_area' in cfg:
        out = partial_occlusion(out, occlusion_area=cfg['occlusion_area'], seed=seed+5)
    return out
