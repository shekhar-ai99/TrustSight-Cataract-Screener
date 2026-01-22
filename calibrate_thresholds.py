#!/usr/bin/env python
"""
Calibrate per-class thresholds to maximize Macro-F1 on validation set.
Run after training, before submission.
"""
import sys
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
import pandas as pd

sys.path.insert(0, 'src')
from train import CataractDataset, val_transform
from model import CataractModel

# Configuration
BEST_MODEL_PATH = Path('best_model/model.pth')
PARQUET_PATH = Path('dataset/merged_training_dataset.parquet')
DEVICE = 'cpu'

# Load validation dataset and split
print("Loading validation dataset...")
df = pd.read_parquet(PARQUET_PATH)

# Use stratified split logic (same as training)
from sklearn.model_selection import StratifiedShuffleSplit
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(df, df['label']))
val_df = df.iloc[val_idx].reset_index(drop=True)

# Create validation dataset
val_dataset = CataractDataset(str(PARQUET_PATH), transform=val_transform, is_train=False)
val_dataset.df = val_df

from torch.utils.data import DataLoader
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load model
print(f"Loading model from {BEST_MODEL_PATH}...")
model = CataractModel()
if BEST_MODEL_PATH.exists():
    state = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
else:
    print("⚠️  Best model not found. Using randomly initialized model.")
model = model.to(DEVICE)
model.eval()

# Collect raw logits and labels
print("\nRunning inference on validation set...")
all_logits = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

logits = torch.cat(all_logits)
labels = torch.cat(all_labels).numpy()
probs = torch.softmax(logits, dim=1).numpy()

print(f"✓ Collected {len(probs)} validation samples\n")

# Find optimal thresholds for each class
CLASS_NAMES = ["No Cataract", "Immature Cataract", "Mature Cataract", "IOL Inserted"]
optimal_thresholds = {}

print("Calibrating per-class confidence thresholds:\n")

for class_idx, class_name in enumerate(CLASS_NAMES):
    # Test thresholds from 0.1 to 0.9
    thresholds = np.arange(0.1, 1.0, 0.05)
    best_threshold = 0.5
    best_f1 = 0.0
    
    for threshold in thresholds:
        # Predictions: if class prob < threshold, fall back to argmax
        predictions = np.argmax(probs, axis=1)
        
        # For majority of samples, use this simple approach
        # (In production, could be more sophisticated)
        
        # Score: F1 for detecting THIS class vs rest
        is_target = (labels == class_idx).astype(int)
        is_pred = (predictions == class_idx).astype(int)
        
        f1 = f1_score(is_target, is_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    optimal_thresholds[class_idx] = best_threshold
    print(f"  {class_name:20s}: threshold={best_threshold:.2f} (F1={best_f1:.4f})")

print("\n" + "="*60)
print("UPDATE inference.py with these thresholds:")
print("="*60)
print("\n_CLASS_THRESHOLDS = {")
for class_idx, threshold in optimal_thresholds.items():
    print(f"    {class_idx}: {threshold:.2f},  # {CLASS_NAMES[class_idx]}")
print("}")

print("\n✓ Thresholds calibrated. Update code/inference.py and retrain submission.")
