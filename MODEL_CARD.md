# Model Card

Architecture
- EfficientNet-B0 backbone with a single output linear head producing a scalar logit.

Data
- Trained on Phase-1 dataset (frozen). No training changes were made in Phase-2/Phase-3/Phase-4.

OOD Handling
- Heavy OOD augmentations applied during Phase-3 training stage (training frozen).
- Inference-time OOD rejection: if the maximum MC-sampled probability is below a conservative threshold, the input is rejected with reason `low_confidence_ood`.

Calibration
- MC Dropout is used (N=10–15) to estimate epistemic uncertainty. Confidence is computed from variance and normalized by 0.25.

Limits
- CPU-only, deterministic inference expected. Do not run with network access in production.
