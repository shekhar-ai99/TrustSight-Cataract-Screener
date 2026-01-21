# Model Submission for NHA–IITK–ICMR Federated Intelligence Hackathon

## 5.1 Model Overview

Model Name: TrustSight Cataract Screener  
Task Type: Multi-class classification (4 classes)  
Framework: PyTorch  
Framework Version: 2.0.1

---

## 5.2 Input / Output Specification

Input:
- Type: pd.DataFrame with `image_vector` column OR np.ndarray
- Shape: (N, 786432)
- Dtype: float32
- Description: Flattened vectors corresponding to 512×512 RGB eye images

Output:
- Type: np.ndarray of strings
- Shape: (N,)
- Meaning: Class labels in the following order:
  - "No Cataract"
  - "Immature Cataract"
  - "Mature Cataract"
  - "IOL Inserted"

---

## 5.3 How to Replicate Predictions

```bash
# Extract submission
tar -xvzf model.tar.gz

# Install dependencies
pip install -r requirements.txt
```

### Usage Examples

**Option 1: With DataFrame (Evaluator format)**
```python
import pandas as pd
import numpy as np
import inference

# DataFrame with image_vector column
df = pd.DataFrame({
    "image_vector": [np.random.rand(786432).astype("float32") for _ in range(2)],
    "other_column": ["sample1", "sample2"]  # Other columns are ignored
})

preds = inference.predict(df)
print(preds)  # Output: array(["Mature Cataract", "No Cataract"])
```

**Option 2: With raw numpy array (Backward compatible)**
```python
import numpy as np
import inference

# Flattened image vectors
data = np.random.rand(2, 786432).astype("float32")

preds = inference.predict(data)
print(preds)  # Output: array(["Mature Cataract", "No Cataract"])
```

### Critical Data Contract

⚠️ **IMPORTANT**: The model assumes `image_vector` is a **flattened raw RGB pixel vector of exactly 786,432 elements** (512×512×3).

If the evaluator provides:
- ✅ Raw flattened RGB pixels (512×512×3) → Works perfectly
- ❌ Embeddings or different resolution → Will raise clear error: `image_vector length X cannot be reshaped to (3, 512, 512)`

---

## 5.4 Training Summary

Datasets Used:  
Public cataract image datasets aligned to the CDIS schema

Loss Function:  
CrossEntropyLoss (class-weighted)

Backbone Model:  
EfficientNet-B0 (ImageNet pretrained)

Key Preprocessing Steps:
- Resize and pad images to 512×512
- Normalize RGB channels
- Flatten images into 786,432-length vectors
- Store data in Parquet format for training

====================
TEAM DETAILS
====================
Team Number: 992942
Team Name: UpayaAI
Institution / Organization: MSCB University, Odisha

Primary Contact:
- Name: Chandra Shekhar
- Email: shekhar.it99@gmail.com
