# Model Submission for NHA–IITK–ICMR Federated Intelligence Hackathon

## 5.1 Model Overview

Model Name: TrustSight Cataract Screener
Task: 4-class classification
Framework: PyTorch
Framework Version: 2.2.0

## 5.2 Input / Output Specification
Input:
- Shape: (N, 786432)
- Description: Flattened 512x512 RGB image vectors

Output:
- Shape: (N, 4)
- Meaning: Probabilities for [No Cataract, Immature Cataract, Mature Cataract, IOL Inserted]

## 5.3 How to Replicate Predictions
tar -xvzf model.tar.gz
pip install -r requirements.txt
python -c "import inference; inference.predict(data)"

## 5.4 Training Summary

Datasets: Roshni Study (CDIS)
Loss: CrossEntropy
Backbone: EfficientNet-B0
Preprocessing: resize/pad 512×512

====================
TEAM DETAILS
====================
Team Number: [Your Team Number]
Team Name: UpayaAI
Institution / Organization: MSCB University, Odisha

Primary Contact:
- Name: Chandra Shekhar
- Email: shekhar.it99@gmail.com
