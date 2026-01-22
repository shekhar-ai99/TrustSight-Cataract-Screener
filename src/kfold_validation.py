"""
Stratified 5-fold cross-validation for robust model evaluation.
Tests if model generalizes across different data splits.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import datetime
import time

# Add src to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from config import (
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    TRAIN_RESOLUTION, VAL_RESOLUTION, DEVICE, PARQUET_DATASET,
    CLASS_WEIGHTS, FREEZE_BACKBONE_UNTIL_EPOCH, UNFREEZE_LR,
    LABEL_SMOOTHING, GRADIENT_ACCUMULATION_STEPS,
    SCHEDULER_FACTOR, SCHEDULER_PATIENCE, CLASS_WEIGHT_MIN, CLASS_WEIGHT_MAX, CLASS_WEIGHT_POWER
)
from model import CataractModel
from train import CataractDataset, get_train_transforms, get_val_transforms, set_seed

set_seed(42)

def run_kfold(n_splits=5):
    """Run stratified K-fold validation."""
    
    # Load full dataset
    parquet_path = os.path.abspath(os.path.join(ROOT, PARQUET_DATASET))
    df_full = pd.read_parquet(parquet_path)
    
    class_to_idx = {
        'No Cataract': 0,
        'Immature Cataract': 1,
        'Mature Cataract': 2,
        'IOL Inserted': 3
    }
    
    # Extract labels
    label_col = 'label'
    if label_col not in df_full.columns:
        for col in ['Cataract Type', 'cataract type', 'Cataract_Type', 'cataract_type']:
            if col in df_full.columns:
                label_col = col
                break
    
    y = df_full[label_col].map(class_to_idx).values
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(df_full)), y), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{n_splits}")
        print(f"{'='*60}")
        
        # Create datasets
        train_dataset = CataractDataset(parquet_path, transform=get_train_transforms())
        val_dataset = CataractDataset(parquet_path, transform=get_val_transforms())
        
        # Create subsets
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)
        
        # DataLoaders
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Model
        model = CataractModel().to(DEVICE)
        
        # Compute class weights with clamping (same as train.py FIX-2)
        train_labels_raw = [df_full.iloc[i][label_col] for i in train_idx]
        # Convert to indices if they're strings
        try:
            train_labels = [class_to_idx[label] if isinstance(label, str) else int(label) for label in train_labels_raw]
        except (KeyError, ValueError):
            train_labels = [int(label) for label in train_labels_raw]
        
        label_counts = np.bincount(train_labels, minlength=4)
        weights = 1.0 / (label_counts + 1e-6)
        weights = weights ** CLASS_WEIGHT_POWER
        weights = weights / weights.sum() * 4
        weights = np.clip(weights, CLASS_WEIGHT_MIN, CLASS_WEIGHT_MAX)
        weights = weights / weights.sum() * 4
        class_weights_computed = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        
        # Use CrossEntropyLoss instead of FocalLoss
        criterion = nn.CrossEntropyLoss(weight=class_weights_computed, label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, verbose=False)
        
        # Training
        best_val_f1 = 0.0
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS}: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}")
            
            scheduler.step(val_loss)
        
        print(f"\nFold {fold} Best Val F1: {best_val_f1:.4f}")
        fold_results.append(best_val_f1)
    
    print(f"\n{'='*60}")
    print("K-FOLD SUMMARY")
    print(f"{'='*60}")
    print(f"Fold F1 Scores: {[f'{r:.4f}' for r in fold_results]}")
    print(f"Mean F1: {np.mean(fold_results):.4f}")
    print(f"Std F1:  {np.std(fold_results):.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_kfold(n_splits=5)
