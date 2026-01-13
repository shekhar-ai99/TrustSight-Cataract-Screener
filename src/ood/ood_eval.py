"""Utilities for OOD evaluation using Phase 2 inference pipeline.

Provides helpers to run inference on augmented datasets and compute conservative
metrics: accuracy, sensitivity, specificity, rejection rate, refer rate.
"""
from typing import Callable, Iterable, Tuple, List, Dict
import os
import numpy as np
import cv2
from ..preprocess import load_image_to_tensor
from ..inference import infer


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # y_true, y_pred are 0/1 arrays
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = float((tp + tn) / max(1, (tp + tn + fp + fn)))
    sensitivity = float(tp / max(1, (tp + fn)))
    specificity = float(tn / max(1, (tn + fp)))
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def evaluate_dataset(image_paths: Iterable[str], labels: Iterable[int], weights_path: str | None = None, n_mc: int = 15, seed: int = 42) -> Dict:
    """Run Phase 2 inference on dataset and compute metrics.

    Returns a dict with per-sample outputs and aggregate metrics.
    """
    image_paths = list(image_paths)
    labels = np.array(list(labels), dtype=int)
    assert len(image_paths) == len(labels)

    preds = []
    actions = []
    rejections = 0
    refers = 0

    for i, p in enumerate(image_paths):
        out_json = infer(p, explain=False, n_mc=n_mc, weights_path=weights_path, seed=seed)
        # infer returns JSON string; parse
        try:
            import json
            parsed = json.loads(out_json)
        except Exception:
            parsed = {}

        action = parsed.get('action')
        # Determine binary prediction: PREDICT with cataract_prob>=0.5 => 1 else 0
        prob = parsed.get('cataract_prob')
        if action == 'REJECT' or prob is None:
            preds.append(0)  # conservative: treat reject as negative prediction for accuracy metrics
            rejections += 1
        else:
            preds.append(1 if prob >= 0.5 else 0)
        if action == 'REFER':
            refers += 1
        actions.append(action)

    preds = np.array(preds, dtype=int)
    metrics = _binary_metrics(labels, preds)
    metrics.update({
        'n': len(labels),
        'rejection_rate': rejections / max(1, len(labels)),
        'refer_rate': refers / max(1, len(labels))
    })

    return {
        'metrics': metrics,
        'actions': actions,
    }
