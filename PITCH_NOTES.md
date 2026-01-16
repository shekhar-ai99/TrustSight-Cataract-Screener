# Pitch Notes for TrustSight Cataract Screener (5-7 min presentation)

## Problem Statement
Mobile eye screening for cataracts remains unreliable in public health settings, leading to missed diagnoses or unnecessary referrals. Traditional AI models lack trust mechanisms, resulting in overconfidence and poor calibration in real-world, out-of-distribution scenarios.

## Federated Challenge
The NHA–IITK–ICMR Hackathon introduces federated evaluation with hidden datasets, differential privacy noise, and an alpha-budget constraint (Rule of 3 submissions). Models must perform under uncertainty without overfitting to leaderboard signals.

## Architecture Overview
- **Input Gate (Reliability)**: Image Quality Assessment (IQA) rejects poor-quality images (blur, glare) before inference.
- **Core Model**: EfficientNet-B0 backbone with Monte Carlo Dropout for uncertainty estimation.
- **Decision Layer (Transparency)**: Conservative policy routes outputs to PREDICT, REFER_TO_SPECIALIST, or REJECT based on confidence and variance.
- **Output Schema**: Strict JSON validation ensures deterministic, auditable results.

## Trilemma Resolution
- **Reliability**: IQA gate + conservative thresholds prioritize safety over accuracy.
- **Openness**: Deterministic, CPU-only inference with no internet dependencies; alpha-budget discipline limits submissions.
- **Transparency**: MC Dropout provides uncertainty quantification; all decisions are explainable via GradCAM.

## Alpha-Budget Discipline
Adhering to Rule of 3 enforces restraint: no retries, no debugging prints in competition mode, ensuring stable behavior under DP noise. This promotes reproducible, trust-worthy submissions.

## ABDM Alignment & Public Health Impact
Aligned with Ayushman Bharat Digital Mission for privacy-preserving digital health. Enables scalable screening in rural areas, reducing clinician burden while maintaining high specificity. As a Digital Public Good, it supports equitable access to eye care, potentially preventing vision loss through early intervention.