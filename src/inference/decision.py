"""Decision logic for routing based on uncertainty.

Conservative, clinician-first policy:
- High confidence + low variance -> PREDICT
- Medium uncertainty -> REFER
- Very high uncertainty -> REJECT

Only returns one of: "PREDICT", "REFER", "REJECT".
"""
from typing import Literal


def decide_action(mean_prob: float, var: float, confidence: float) -> Literal["PREDICT", "REFER", "REJECT"]:
    # Conservative thresholds chosen to favor specificity and safety.
    # If model is very confident (confidence >= 0.85) and variance low -> PREDICT
    if confidence >= 0.85 and var <= 0.02:
        return "PREDICT"

    # Medium confidence -> REFER for clinician review
    if 0.6 <= confidence < 0.85 or (var > 0.02 and confidence >= 0.5):
        return "REFER"

    # Fallback: REJECT for very uncertain or low confidence
    return "REJECT"
