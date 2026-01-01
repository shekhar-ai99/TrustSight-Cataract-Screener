# OOD Robustness Report

## 1) OOD Strategy

We apply conservative, inference-time OOD handling without any retraining. Two mechanisms are used:
- Heavy OOD augmentations were applied during Phase 3 (training frozen) to improve robustness offline.
- At inference time we apply a conservative confidence-based rejection gate: if the maximum sampled probability
  (across MC dropout passes) is below a fixed threshold, the input is considered OOD and rejected.

This approach preserves the frozen model and ensures the system refuses to guess on unfamiliar inputs.

## 2) Cross-dataset AUC Comparison

Use the outputs from `src/calibrate.py` to obtain AUC proxies for in-distribution and cross-dataset evaluations. Example CLI:

```
python3 -m src.calibrate in_dist_preds.csv cross_dist_preds.csv
```

Replace the table below with the actual logged metrics from the CLI run.

## 3) Calibration Results Table

| Scenario        | AUC  | ECE  | Reject % |
|-----------------|------|------|----------|
| In-distribution | ...  | ...  | ...      |
| Cross-dataset   | ...  | ...  | ...      |

`Reject %` can be computed by running inference over the dataset and counting samples rejected by the OOD gate.

## 4) Safety Policy

When uncertainty or image quality exceeds safe bounds, the system rejects rather than guessing.

---

Notes:
- This report is intentionally concise and audit-friendly. All thresholds and rules are fixed and inference-only.
- To populate the table, run `src.calibrate` and the existing evaluation scripts; record the outputs here.
