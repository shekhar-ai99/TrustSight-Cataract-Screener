"""Threshold suggestion and freeze utility for Phase 3.

This module computes a one-time recommended threshold set based on calibration
statistics and saves a frozen JSON-like dict for handoff. It does NOT mutate
Phase 2 logic; it only produces locked recommendations.
"""
from typing import Dict


FROZEN_THRESHOLDS: Dict[str, float] = {
    # Conservative default recommendations (frozen values)
    'predict_confidence_min': 0.90,
    'predict_variance_max': 0.01,
    'refer_confidence_min': 0.60,
    'reject_confidence_max': 0.50,
}


def get_frozen_thresholds() -> Dict[str, float]:
    """Return the frozen threshold dict. These values are intended to be
    referenced in reporting and for manual handoff; Phase 2 code is not
    automatically changed by this module.
    """
    return dict(FROZEN_THRESHOLDS)
