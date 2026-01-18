# Model Submission for NHA–IITK–ICMR Federated Intelligence Hackathon

## 5.1 Model Overview
Model Name: TrustSight Cataract Screener
Task Type: Multi-Class Classification (4-class Cataract Screening)
Problem: Cataract Detection (CDIS – Roshni Study)
Framework: PyTorch
Framework Version: 2.0.1

## 5.2 Input / Output Specification
Input:
- Shape: list of strings or numpy.ndarray of shape (N, H, W, 3) or torch.Tensor
- Dtype: str for list, float32 for arrays
- Description: Image paths or RGB images (0-255 range)

Output:
- Shape: list of dicts, each with keys 'predicted_class', 'class_probs', 'confidence', 'uncertainty'
- Meaning: predicted_class (str: "No Cataract", "Immature Cataract", "Mature Cataract", "IOL Inserted"), class_probs (dict of 4 floats), confidence (float), uncertainty (str: "LOW", "MEDIUM", "HIGH")

## 5.3 How to Replicate Predictions (MANDATORY)
```bash
tar -xvzf model.tar.gz
pip install -r requirements.txt
python test.py
```

Python usage:

```python
import sys
sys.path.append("code")
import inference
preds = inference.predict(test_data)  # test_data is list of paths
```

## 5.4 Training Summary (High-Level Only)
Dataset used: Phase-1 frozen dataset
Loss function: Cross-Entropy Loss (4 classes)
Optimizer: AdamW
Key preprocessing steps: Resize to 224x224, normalize with ImageNet stats

====================
TEAM DETAILS
====================
Team Number: [Your Team Number]
Team Name: UpayaAI
Institution / Organization: MSCB University, Odisha

Primary Contact:
- Name: Chandra Shekhar
- Email: shekhar.it99@gmail.com
