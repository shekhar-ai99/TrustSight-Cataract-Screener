#!/usr/bin/env python
"""
Quick test to debug gradient flow issues.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import sys
sys.path.insert(0, 'src')

from focal_loss import FocalLoss

# Create a simple ResNet18-like model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)

# Freeze backbone except FC
print("Freezing backbone...")
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

# Check parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Trainable: {trainable:,}, Frozen: {frozen:,}")

# Create dummy batch
images = torch.randn(2, 3, 224, 224)
labels = torch.tensor([0, 1], dtype=torch.long)

# Forward pass
model.train()
outputs = model(images)
print(f"\nOutputs shape: {outputs.shape}")
print(f"Outputs requires_grad: {outputs.requires_grad}")
print(f"Outputs grad_fn: {outputs.grad_fn}")

# Create loss with weights (like in actual training)
class_weights = torch.tensor([0.469, 1.438, 0.571, 1.522])
print(f"Class weights: {class_weights}")
criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
loss = criterion(outputs, labels)
print(f"\nLoss value: {loss.item()}")
print(f"Loss requires_grad: {loss.requires_grad}")
print(f"Loss grad_fn: {loss.grad_fn}")

# Try backward
try:
    loss.backward()
    print("\n✓ Backward pass successful!")
    
    # Check if FC layer got gradients
    for name, param in model.named_parameters():
        if 'fc' in name and param.grad is not None:
            print(f"✓ {name} has gradient: {param.grad.shape}")
except Exception as e:
    print(f"\n✗ Backward failed: {e}")
