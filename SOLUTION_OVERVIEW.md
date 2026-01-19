# TrustSight Cataract Screener - Complete Solution Overview

## Project Goal
Build an **AI-powered cataract screening system** for the NHA–IITK–ICMR Federated Intelligence Hackathon. The system classifies eye conditions into 4 categories and provides both predictions and uncertainty estimates with clinical decision support.

---

## Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│     INPUT (Training / Internal Inference): Eye Image (512×512)  │
│     INPUT (Hackathon Evaluation): Flattened image vectors       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  1. IMAGE QUALITY ASSESSMENT (IQA)                              │
│     - Detects blur (Laplacian variance < 50)                    │
│     - Detects bad exposure (intensity < 40 or > 220)            │
│     - Detects glare/saturation (>15% saturated pixels)          │
│     ↓ REJECTS low-quality images immediately                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. PREPROCESSING                                               │
│     - Load image and convert to RGB                             │
│     - Resize to 512×512                                         │
│     - Normalize (ImageNet: mean=[0.485, 0.456, 0.406])          │
│     - Convert to tensor                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. NEURAL NETWORK (EfficientNet-B0)                            │
│     - Backbone: EfficientNet-B0 (pretrained on ImageNet)        │
│     - Output layer: 4 neurons (4-class classification)          │
│     - Classes: [No Cataract, Immature, Mature, IOL Inserted]    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. MONTE CARLO DROPOUT SAMPLING                                │
│     - Run model 15 times with dropout ENABLED                   │
│     - Each pass produces different predictions due to dropout   │
│     - Samples predict = predictions from multiple stochastic    │
│       forward passes through the network                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. UNCERTAINTY ESTIMATION                                      │
│     - Mean probability: average across all 15 samples           │
│     - Variance: how much samples differ from each other         │
│     - Confidence = 1 - (variance / 0.25)                        │
│                                                                 │
│     These metrics tell us how "sure" the model is               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  6. DECISION LOGIC (Conservative Clinical Routing)              │
│     ┌──────────────────────────────────────────────────────┐    │
│     │ IF confidence ≥ 0.85 AND variance ≤ 0.02:           │    │
│     │    ACTION = "PREDICT" (trust the prediction)         │    │
│     ├──────────────────────────────────────────────────────┤    │
│     │ ELSE IF confidence 0.6-0.85 OR medium variance:      │    │
│     │    ACTION = "REFER_TO_SPECIALIST"                    │    │
│     │    (uncertain → clinician review)                    │    │
│     ├──────────────────────────────────────────────────────┤    │
│     │ ELSE:                                                │    │
│     │    ACTION = "REJECT" (too uncertain)                 │    │
│     └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Internal Diagnostic Output (for development & evaluation only) │
│  {                                                              │
│    "prediction": "CATARACT_PRESENT" or "NORMAL",                │
│    "confidence": 0.0-1.0 (model certainty),                     │
│    "uncertainty": "LOW" | "MEDIUM" | "HIGH",                    │
│    "action": "PREDICT" | "REFER_TO_SPECIALIST" | "REJECT"       │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. **Data Pipeline** (`dataset/` + `src/preprocess.py`)
- **Input**: Parquet file with image vectors (512×512×3 flattened to 786,432 values)
- **Storage**: `dataset/cataract-training-dataset.parquet`
- **Content**: 
  - `image_vector`: flattened 512×512×3 RGB array
  - `label`: one of [No_Cataract, Immature_Cataract, Mature_Cataract, IOL_Inserted]
  - `image_quality`: metadata
  - `patient_age_group`: metadata

### 2. **Model Architecture** (`src/model.py`)
- **Backbone**: EfficientNet-B0 (efficient mobile-friendly CNN)
- **Input**: (N, 3, 512, 512) RGB tensors
- **Output**: (N, 4) logits for 4 classes
- **Key Features**:
  - MC Dropout support for uncertainty estimation
  - `safe_mc_forward()`: runs multiple forward passes with dropout active
  - `predict_proba()`: deterministic (n_mc=1) or stochastic (n_mc>1) predictions

### 3. **Training Pipeline** (`src/train.py`)
- **Data Loading**: Reads parquet file → extracts image vectors → reconstructs 512×512 RGB
- **Class Balancing**: Weighted CrossEntropy loss accounts for class imbalance
- **Optimization**: AdamW optimizer with lr=1e-4
- **Training**:
  - Epochs: 5 (configurable)
  - Batch size: 8
  - Device: CPU (to avoid memory issues)
  - Metrics: F1 score (macro), validation loss
- **Output**: Saves trained model to `test_submission/` with timestamp

### 4. **Inference Pipeline** (`src/infer.py`)
The complete prediction flow:
1. **IQA Check** (`src/iqa.py`): Reject blurry/exposed/glare images
2. **Preprocessing** (`src/preprocess.py`): Load and normalize image
3. **Model Load**: Load trained weights
4. **MC Dropout Sampling**: 15 stochastic forward passes
5. **Statistics Computation**: Mean, variance, confidence
6. **Decision Making** (`src/inference/decision.py`): Route to PREDICT/REFER/REJECT
7. **Output Validation**: Pydantic schema validation
8. **Explainability** (optional): GradCAM visualization

### 5. **Image Quality Assessment** (`src/iqa.py`)
Rejects images if:
- **Blur**: Laplacian variance < 50
- **Bad Exposure**: Mean intensity < 40 or > 220
- **Glare/Saturation**: >15% pixels with saturation ≥ 250

### 6. **Uncertainty Estimation** (`src/inference/mc_dropout.py`)
- **MC Dropout**: Multiple forward passes with stochastic dropout
- **Mean Probability**: Average prediction across samples
- **Variance**: Spread of predictions (higher = more uncertain)
- **Confidence**: 1 - (variance / 0.25) → normalized to [0, 1]

### 7. **Decision Logic** (`src/inference/decision.py`)
Conservative thresholds favoring clinician review:
- **PREDICT**: confidence ≥ 0.85 AND variance ≤ 0.02 (high confidence)
- **REFER_TO_SPECIALIST**: 0.6 ≤ confidence < 0.85 (medium uncertainty)
- **REJECT**: confidence < 0.6 (too uncertain to act)

Decision logic is part of an internal clinical decision layer and is not invoked during automated benchmark evaluation, which consumes only class probabilities.

---

## File Structure

```
src/
├── train.py                 # Training loop (loads parquet, trains EfficientNet)
├── infer.py                 # Main inference pipeline
├── model.py                 # CataractModel with MC Dropout
├── preprocess.py            # Image loading and normalization
├── iqa.py                   # Image Quality Assessment
├── gradcam.py               # Explainability (visualization)
├── utils.py                 # Seed setting, logging
├── schema/
│   └── output_schema.py      # Pydantic schema for output validation
├── inference/
│   ├── decision.py           # Action routing logic (PREDICT/REFER/REJECT)
│   └── mc_dropout.py         # MC Dropout sampling
├── calibration/             # Uncertainty calibration
│   ├── ece.py                # Expected Calibration Error
│   ├── reliability_diagram.py
│   └── thresholding.py
├── iqa/                     # Image quality modules
├── ood/                     # Out-of-distribution detection
├── evaluation/              # Cross-dataset evaluation
└── tests/                   # Unit tests

dataset/
└── cataract-training-dataset.parquet  # Training data

utils/
├── extract_from_parquet.py  # Extract images from parquet to folders
└── log_run.py               # Experiment logging
```

---

## Data Flow: Training to Inference

### Training Phase
```
Raw Images (folder structure)
    ↓
preprocess.py: Convert to parquet
    ↓
dataset/cataract-training-dataset.parquet
    ↓
train.py: Read parquet → load image vectors → train EfficientNet-B0
    ↓
Saved Model (test_submission/model_[timestamp]_F1_[score].tar.gz)
```

### Inference Phase
```
Eye Image File
    ↓
infer.py → iqa.py (quality check)
    ↓ (if fails: return REJECT)
    ↓ (if passes)
    ↓
preprocess.py (load + normalize)
    ↓
model.py (EfficientNet-B0 + MC Dropout)
    ↓
inference/mc_dropout.py (15 stochastic passes)
    ↓
Compute: mean, variance, confidence
    ↓
inference/decision.py (routing logic)
    ↓
JSON Output: {prediction, confidence, uncertainty, action}
```

---

## Why This Approach?

### 1. **EfficientNet-B0**
- Lightweight & mobile-friendly
- Good accuracy-to-parameters ratio
- Pretrained on ImageNet for transfer learning

### 2. **Monte Carlo Dropout**
- Quantifies model uncertainty
- Single model → multiple predictions
- Computationally cheap vs ensemble

### 3. **Conservative Decision Logic**
- **Specificity first**: Avoids false positives (misclassifying cataract as normal)
- **Clinician review**: Uncertain cases → human expert review
- **Safety**: High threshold for autonomous prediction

### 4. **Image Quality Assessment**
- Garbage in → garbage out
- Catches poor quality before wasting inference

### 5. **Class Balancing**
- Some cataract types are rare
- Weighted loss ensures fair training

---

## Recent Change: Parquet Data Loading

**What Changed**: `train.py` now loads data directly from `dataset/cataract-training-dataset.parquet` instead of looking for image folders.

**Why**:
- Data is already in parquet format
- No need for folder structure
- Single source of truth for training data
- Easier to manage large datasets

**How**:
1. Read parquet file with pandas
2. Extract `image_vector` column (flattened 512×512×3)
3. Reshape to (512, 512, 3) → PIL Image
4. Apply standard transforms
5. Feed to model

---

## Key Metrics & Thresholds

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| **Confidence** | ≥ 0.85 | High certainty |
| **Variance** | ≤ 0.02 | Low uncertainty |
| **Laplacian Variance** | ≥ 50 | Not blurry |
| **Mean Intensity** | 40-220 | Well exposed |
| **Saturation** | ≤ 15% | No excessive glare |
| **F1 Score** | Macro average | Training success metric |

---

## Output Schema Example

```json
{
  "prediction": "CATARACT_PRESENT",
  "confidence": 0.92,
  "uncertainty": "LOW",
  "action": "PREDICT"
}
```

OR (if image quality issues):
```json
{
  "status": "REJECT",
  "reason": "LOW_IMAGE_QUALITY"
}
```

---

## Next Steps / Future Work

1. **Evaluation** (`src/evaluation/cross_dataset_eval.py`): Test on multiple datasets
2. **Calibration** (`src/calibration/`): Ensure confidence scores are well-calibrated
3. **OOD Detection** (`src/ood/`): Detect out-of-distribution images
4. **Phase 3 Hardening**: Finalize thresholds, create reliability diagrams, ECE analysis
5. **Deployment**: Package as Docker image, integrate with clinical systems

---

## Team Information

- **Team Name**: UpayaAI
- **Institution**: MSCB University, Odisha
- **Primary Contact**: Chandra Shekhar (shekhar.it99@gmail.com)
