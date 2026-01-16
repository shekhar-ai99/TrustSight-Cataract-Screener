# Model Card for TrustSight Cataract Screener

## Intended Use
This model is designed as a **screening aid** for cataract detection in mobile eye images, not a diagnostic tool. It supports public health initiatives under the National Health Authority (NHA) by providing conservative predictions that prioritize clinician review.

## Architecture and Trust Mechanisms
- **Backbone**: EfficientNet-B0 with a single output linear head.
- **Reliability (IQA Gate)**: Images are first assessed for quality (blur, exposure, glare). Low-quality images are rejected before inference to ensure reliable inputs.
- **Transparency (MC Dropout)**: Monte Carlo Dropout (10–15 passes) estimates epistemic uncertainty. Confidence is derived from variance, normalized by 0.25.
- **Openness (Alpha-Budget Discipline)**: Limited to Rule of 3 submissions, enforcing restraint and reproducibility.

## Data
- Trained on Phase-1 dataset (frozen). No training changes in subsequent phases.
- OOD augmentations applied during Phase-3 training for robustness.

## Calibration and Decision Policy
- Outputs include prediction (CATARACT_PRESENT/NORMAL), confidence (0-1), uncertainty level (LOW/MEDIUM/HIGH), and action (PREDICT/REFER_TO_SPECIALIST/REJECT).
- Conservative thresholds: PREDICT only for high confidence + low variance; REFER for medium uncertainty; REJECT for high uncertainty.

## Limitations and Failure Modes
- CPU-only, deterministic inference.
- Prefers deferral (REFER/REJECT) over confident errors.
- May reject valid images if quality is poor or uncertainty is high.
- Not suitable for real-time diagnostics without clinician oversight.

## Ethical Considerations
- Aligned with ABDM (Ayushman Bharat Digital Mission) for privacy-preserving health data.
- Promotes equity by enabling screening in resource-constrained settings.
- Emphasizes safety: high specificity to avoid false positives that could delay treatment.
