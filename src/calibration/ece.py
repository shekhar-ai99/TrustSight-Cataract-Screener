"""Expected Calibration Error (ECE) computation.

Provides `expected_calibration_error(confidences, labels, n_bins=10)`.
"""
import numpy as np
from typing import Tuple


def expected_calibration_error(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Tuple[float, dict]:
    """Compute ECE and per-bin stats.

    Args:
        confidences: model confidence scores (0-1), shape (N,)
        labels: true binary labels (0/1), shape (N,)
    Returns:
        ece: scalar
        bins: dict with per-bin accuracy, avg_confidence, support
    """
    confidences = np.asarray(confidences)
    labels = np.asarray(labels)
    assert confidences.shape[0] == labels.shape[0]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = confidences.shape[0]
    bins = {}

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i+1]
        mask = (confidences > lo) & (confidences <= hi) if i < n_bins - 1 else (confidences >= lo) & (confidences <= hi)
        support = mask.sum()
        if support == 0:
            bins[f'{lo:.2f}-{hi:.2f}'] = {'support': 0, 'accuracy': None, 'avg_confidence': None}
            continue
        acc = labels[mask].mean()
        avg_conf = confidences[mask].mean()
        ece += (support / n) * abs(avg_conf - acc)
        bins[f'{lo:.2f}-{hi:.2f}'] = {'support': int(support), 'accuracy': float(acc), 'avg_confidence': float(avg_conf)}

    return float(ece), bins
