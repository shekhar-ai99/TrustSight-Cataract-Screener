# Model Submission for NHA–IITK–ICMR Federated Intelligence Hackathon

## 5.1 Model Overview

Model Name: TrustSight Cataract Screener  
Task Type: Multi-class classification (4 classes)  
Framework: PyTorch  
Framework Version: 2.2.0

---

## 5.2 Input / Output Specification

Input:
- Shape: (N, 786432)
- Dtype: float32
- Description: Flattened vectors corresponding to 512×512 RGB eye images

Output:
- Shape: (N, 4)
- Meaning: Class probabilities in the following fixed order:  
	[No Cataract, Immature Cataract, Mature Cataract, IOL Inserted]

---

## 5.3 How to Replicate Predictions

```bash
# Extract submission
tar -xvzf model.tar.gz

# Install dependencies
pip install -r requirements.txt
```

```python
import numpy as np
import inference

# Example input: N flattened image vectors
# Shape: (N, 786432)
data = np.random.rand(2, 786432).astype("float32")

preds = inference.predict(data)
print(preds)  # Output shape: (N, 4)
```

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
