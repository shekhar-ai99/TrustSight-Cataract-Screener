#!/usr/bin/env python
"""
Load 3 trained ensemble models and average their predictions on validation set.
Saves ensemble predictions for calibration and submission.
"""
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train import CataractDataset, val_transform
from model import CataractModel

# Configuration
ENSEMBLE_DIR = Path('ensemble_models')
PARQUET_PATH = Path('dataset/merged_training_dataset.parquet')
DEVICE = 'cpu'
BATCH_SIZE = 16

CLASS_NAMES = ["No Cataract", "Immature Cataract", "Mature Cataract", "IOL Inserted"]

def load_validation_set():
    """Load validation dataset (same split as training)."""
    print("Loading validation dataset...")
    df = pd.read_parquet(PARQUET_PATH)
    
    # Use same stratified split as training
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df, df['label']))
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    # Create dataset
    val_dataset = CataractDataset(str(PARQUET_PATH), transform=val_transform, is_train=False)
    val_dataset.df = val_df
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return val_loader, val_df

def load_fold_model(fold: int):
    """Load model from fold directory."""
    fold_dir = ENSEMBLE_DIR / f'fold_{fold}'
    model_path = fold_dir / 'model.pth'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"  Loading fold_{fold} model...")
    model = CataractModel()
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    model.eval()
    
    return model

def get_ensemble_predictions():
    """Run all 3 models and average predictions."""
    print("\n" + "="*70)
    print("ENSEMBLE INFERENCE: Averaging 3 Models")
    print("="*70 + "\n")
    
    # Load validation set
    val_loader, val_df = load_validation_set()
    print(f"✓ Validation set: {len(val_df)} samples\n")
    
    # Load all 3 models
    print("Loading models from ensemble_models/:")
    models = []
    for fold in [1, 2, 3]:
        try:
            model = load_fold_model(fold)
            models.append(model)
        except FileNotFoundError as e:
            print(f"  ❌ {e}")
            return None, None
    
    print(f"✓ Loaded {len(models)} models\n")
    
    # Collect predictions from all models
    print("Running inference...")
    all_ensemble_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(DEVICE)
            
            # Run all 3 models
            batch_probs = []
            for model_idx, model in enumerate(models):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                batch_probs.append(probs.cpu())
            
            # Average probabilities
            ensemble_probs = torch.mean(torch.stack(batch_probs), dim=0)
            all_ensemble_probs.append(ensemble_probs)
            all_labels.append(labels)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches")
    
    ensemble_probs = torch.cat(all_ensemble_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    
    print(f"✓ Collected {len(ensemble_probs)} ensemble predictions\n")
    
    return ensemble_probs, labels

def evaluate_ensemble(probs, labels):
    """Evaluate ensemble using argmax (baseline)."""
    from sklearn.metrics import f1_score, confusion_matrix
    
    predictions = np.argmax(probs, axis=1)
    
    # Overall metrics
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    # Per-class metrics
    print("="*70)
    print("ENSEMBLE BASELINE EVALUATION (Argmax)")
    print("="*70)
    print(f"\nOverall Macro-F1: {macro_f1:.4f}\n")
    
    print("Per-Class F1:")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        is_target = (labels == class_idx).astype(int)
        is_pred = (predictions == class_idx).astype(int)
        f1 = f1_score(is_target, is_pred, zero_division=0)
        support = np.sum(is_target)
        print(f"  {class_name:20s}: F1={f1:.4f} (n={support})")
    
    print("\n" + "="*70)
    
    return macro_f1

def save_predictions(probs, labels):
    """Save ensemble predictions for threshold calibration."""
    output_dir = ENSEMBLE_DIR / 'validation_predictions'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'ensemble_probs.npy', probs)
    np.save(output_dir / 'labels.npy', labels)
    
    print(f"✓ Saved predictions to {output_dir}/")
    print(f"  - ensemble_probs.npy: shape {probs.shape}")
    print(f"  - labels.npy: shape {labels.shape}")

def main():
    # Get ensemble predictions
    ensemble_probs, labels = get_ensemble_predictions()
    
    if ensemble_probs is None:
        print("\n❌ Failed to generate ensemble predictions")
        sys.exit(1)
    
    # Evaluate
    baseline_f1 = evaluate_ensemble(ensemble_probs, labels)
    
    # Save for calibration
    save_predictions(ensemble_probs, labels)
    
    print(f"\n✓ Next step: python calibrate_ensemble_thresholds.py")

if __name__ == '__main__':
    main()
