#!/usr/bin/env python
"""
Test gradient flow with actual training data and DataLoader.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'src')

from train import CataractDataset, train_transform
from focal_loss import FocalLoss

# Load parquet
df = pd.read_parquet('dataset/merged_training_dataset.parquet')
print(f"Loaded {len(df)} samples")

# Create dataset
dataset = CataractDataset('dataset/merged_training_dataset.parquet', transform=train_transform, is_train=True)
dataset.df = df.iloc[:40]  # Use just 40 samples for testing

# Create simple dataloader
loader = DataLoader(dataset, batch_size=2, shuffle=False)

# Create model with frozen backbone
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)

print("Freezing backbone...")
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

model.train()
model = model.to('cpu')

# Create optimizer and loss
class_weights = torch.tensor([0.469, 1.438, 0.571, 1.522], device='cpu')
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)

print("\nStarting training loop...")
for batch_idx, (images, labels) in enumerate(loader):
    print(f"\nBatch {batch_idx}:")
    print(f"  Images shape: {images.shape}, dtype: {images.dtype}")
    print(f"  Labels shape: {labels.shape}, dtype: {labels.dtype}")
    
    # Forward
    outputs = model(images)
    print(f"  Outputs shape: {outputs.shape}, requires_grad: {outputs.requires_grad}")
    
    # Loss
    loss = criterion(outputs, labels)
    print(f"  Loss value: {loss.item()}, requires_grad: {loss.requires_grad}, grad_fn: {loss.grad_fn}")
    
    # Backward
    try:
        loss.backward()
        print(f"  ✓ Backward successful!")
        optimizer.step()
        optimizer.zero_grad()
    except Exception as e:
        print(f"  ✗ Backward failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if batch_idx >= 2:
        break

print("\n✓ All batches processed successfully!")
