Phase 2: Safety, Trust & Transparency

Notes:
- No model architecture, training, or dataset changes were made.
- Phase-2 adds inference-time safety and transparency layers only: IQA gate and MC Dropout.
- IQA rejects images with excessive blur, poor exposure, or high glare before inference.
- MC Dropout (10–15 passes) estimates epistemic uncertainty; confidence computed from variance.
- Decision policy: PREDICT (confidence ≥ 0.8), REFER_TO_SPECIALIST (0.5 ≤ confidence < 0.8), REJECT (confidence < 0.5).

CLI smoke tests are provided at `scripts/smoke_test.py` — provide a clean image and a bad image to verify behavior.

Commit message for this change: "Phase 2: Safety & Trust layers (IQA + MC Dropout)"
Tag: phase-2-freeze
