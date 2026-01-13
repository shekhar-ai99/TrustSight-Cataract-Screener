# Phase 3 Report — OOD Hardening & Calibration

Author: Phase 3 Automation

Summary
-------
- Implements deterministic OOD augmentations for evaluation.
- Provides cross-dataset evaluation wrappers calling Phase 2 inference.
- Computes Expected Calibration Error (ECE) and produces reliability diagrams.
- Produces a one-time frozen threshold recommendation for conservative behavior.

Artifacts produced by this module (to be generated during evaluation):
- `reports/phase3/reliability_in_distribution.png`
- `reports/phase3/reliability_ood.png`
- `reports/phase3/ece_summary.json`
- `reports/phase3/augmentation_examples/` (sample images)
- `reports/phase3/phase3_thresholds.json`

Final frozen thresholds (recommended, conservative):

```
predict_confidence_min: 0.90
predict_variance_max: 0.01
refer_confidence_min: 0.60
reject_confidence_max: 0.50
```

Rationale
---------
- Thresholds are intentionally conservative to prioritize specificity and
  clinician review. Those working in production may adopt these values
  for manual handoff into Phase 2 policy enforcement.

Next steps for deployment
-------------------------
1. Run cross-dataset evaluation with real dataset paths and save artifacts.
2. Review ECE and reliability diagrams with clinical stakeholders.
3. Freeze thresholds in production config and lock weights.
