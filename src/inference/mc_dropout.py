"""Monte-Carlo Dropout helper.

Provides `mc_inference(model, img_tensor, n_mc, seed)` which returns a list
of sampled probabilities. Uses model.predict_proba / safe_mc_forward and
ensures deterministic seeding.
"""
from typing import List
import numpy as np
from ..utils import set_seed


def mc_inference(model, img_tensor, n_mc: int = 15, seed: int = 42) -> List[float]:
    """Run MC Dropout sampling and return probability samples.

    Args:
        model: a model exposing `predict_proba(x, n_mc)` or `safe_mc_forward` behaviour.
        img_tensor: preprocessed tensor for model input.
        n_mc: number of stochastic forward passes.
        seed: random seed for determinism.
    """
    # Deterministic behaviour across numpy/torch
    set_seed(seed)

    # Use model.predict_proba which returns a list of floats when n_mc>1
    samples = model.predict_proba(img_tensor, n_mc=n_mc)
    # Ensure we return a flat list of floats
    samples = list(map(float, np.array(samples).reshape(-1).tolist()))
    return samples
