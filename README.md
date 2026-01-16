# Model Submission for NHA–IITK–ICMR Federated Intelligence Hackathon

## 5.1 Model Overview
Model Name: TrustSight Cataract Screener
Task Type: Binary Classification (Cataract Detection)
Framework: PyTorch
Framework Version: 2.0.1

## 5.2 Input / Output Specification
Input:
- Shape: list of strings or numpy.ndarray of shape (N, H, W, 3) or torch.Tensor
- Dtype: str for list, float32 for arrays
- Description: Image paths or RGB images (0-255 range)

Output:
- Shape: (N,) where N is batch size
- Meaning of each dimension: Probability of cataract presence (0.0 to 1.0)

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
Loss function: Binary Cross-Entropy
Optimizer: Adam
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
