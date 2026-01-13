import os
import sys
import tempfile
import json
import numpy as np
import cv2

# Ensure src is importable when tests run from repo root
ROOT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_SRC not in sys.path:
    sys.path.insert(0, ROOT_SRC)

from inference.mc_dropout import mc_inference
from inference.decision import decide_action
from iqa.image_quality import check_image_quality
from schema.output_schema import InferenceOutputSchema
from inference import infer


def _write_image(arr: np.ndarray, path: str):
    cv2.imwrite(path, arr)


def test_iqa_rejects_bad_image():
    # Create an extremely dark image (mean < 40)
    img = np.zeros((64, 64), dtype=np.uint8)
    fd, p = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        _write_image(img, p)
        out = infer(p, n_mc=3, seed=123)
        parsed = json.loads(out)
        assert parsed.get("status") == "REJECT"
        assert parsed.get("reason") == "LOW_IMAGE_QUALITY"
    finally:
        os.remove(p)


def test_mc_dropout_produces_variance():
    # Create a dummy model-like object with predict_proba that returns varying samples
    class DummyModel:
        def predict_proba(self, x, n_mc=10):
            rng = np.random.default_rng(0)
            return rng.random(n_mc).tolist()

    dummy = DummyModel()
    # img_tensor placeholder (not used by DummyModel)
    samples = mc_inference(dummy, img_tensor=None, n_mc=10, seed=0)
    assert isinstance(samples, list) and len(samples) == 10
    assert np.var(samples) > 0.0


def test_schema_never_breaks():
    payload = {"cataract_prob": 0.5, "confidence": 0.7, "action": "REFER"}
    obj = InferenceOutputSchema(**payload)
    assert obj.cataract_prob == 0.5


def test_deterministic_inference_same_input_same_output():
    # Create a deterministic acceptable image (random noise with fixed seed)
    rng = np.random.default_rng(42)
    img = (rng.integers(60, 180, size=(128, 128), dtype=np.uint8)).astype(np.uint8)
    fd, p = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        _write_image(img, p)
        out1 = infer(p, n_mc=5, seed=999)
        out2 = infer(p, n_mc=5, seed=999)
        assert out1 == out2
        # also validate schema
        parsed = json.loads(out1)
        _ = InferenceOutputSchema(**parsed)
    finally:
        os.remove(p)
