#!/usr/bin/env python
"""
Calibrate class-specific confidence thresholds for ensemble predictions.
Finds optimal threshold for each class to maximize Macro-F1.
"""
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
import json

ENSEMBLE_DIR = Path('ensemble_models')
PRED_DIR = ENSEMBLE_DIR / 'validation_predictions'

CLASS_NAMES = ["No Cataract", "Immature Cataract", "Mature Cataract", "IOL Inserted"]

def load_predictions():
    """Load saved ensemble predictions."""
    probs_path = PRED_DIR / 'ensemble_probs.npy'
    labels_path = PRED_DIR / 'labels.npy'
    
    if not probs_path.exists() or not labels_path.exists():
        print(f"❌ Predictions not found. Run ensemble_inference.py first.")
        sys.exit(1)
    
    probs = np.load(probs_path)
    labels = np.load(labels_path)
    
    print(f"✓ Loaded {len(probs)} predictions")
    return probs, labels

def apply_thresholds(probs, thresholds):
    """
    Apply per-class confidence thresholds.
    If max prob < threshold, use second-best class.
    """
    predictions = np.zeros(len(probs), dtype=int)
    
    for i, prob_dist in enumerate(probs):
        best_class = np.argmax(prob_dist)
        best_prob = prob_dist[best_class]
        
        threshold = thresholds.get(best_class, 0.3)
        
        if best_prob < threshold:
            # Use second-best class
            second_best = np.argsort(prob_dist)[-2]
            predictions[i] = second_best
        else:
            predictions[i] = best_class
    
    return predictions

def find_optimal_thresholds(probs, labels):
    """Find optimal threshold for each class to maximize macro-F1."""
    print("\n" + "="*70)
    print("THRESHOLD CALIBRATION")
    print("="*70 + "\n")
    
    best_threshold_set = None
    best_macro_f1 = 0.0
    
    # Grid search over thresholds
    threshold_range = np.arange(0.2, 0.9, 0.05)
    
    print("Grid search (this may take a moment)...\n")
    
    # Try all combinations of per-class thresholds
    results = []
    
    # For simplicity, use same approach for all classes but optimize for minority classes
    for t in threshold_range:
        thresholds = {
            0: t,      # No Cataract
            1: t - 0.1,  # Immature Cataract (lower, needs more help)
            2: t,      # Mature Cataract
            3: t + 0.1,  # IOL Inserted (higher, rarest)
        }
        
        # Clip to valid range
        for k in thresholds:
            thresholds[k] = max(0.1, min(0.95, thresholds[k]))
        
        predictions = apply_thresholds(probs, thresholds)
        macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        
        results.append({
            'base_threshold': t,
            'thresholds': thresholds.copy(),
            'macro_f1': macro_f1
        })
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold_set = thresholds.copy()
    
    # Sort results by F1
    results.sort(key=lambda x: -x['macro_f1'])
    
    # Print top 5 threshold sets
    print("Top 5 Threshold Configurations:\n")
    for rank, result in enumerate(results[:5], 1):
        thresholds = result['thresholds']
        macro_f1 = result['macro_f1']
        marker = " ← BEST" if rank == 1 else ""
        print(f"{rank}. Macro-F1: {macro_f1:.4f}{marker}")
        for class_idx, threshold in thresholds.items():
            print(f"   {CLASS_NAMES[class_idx]:20s}: {threshold:.2f}")
        print()
    
    return best_threshold_set, best_macro_f1

def evaluate_with_thresholds(probs, labels, thresholds):
    """Evaluate predictions using optimal thresholds."""
    from sklearn.metrics import confusion_matrix
    
    predictions = apply_thresholds(probs, thresholds)
    
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    print("="*70)
    print("FINAL ENSEMBLE EVALUATION (With Optimal Thresholds)")
    print("="*70)
    print(f"\nMacro-F1: {macro_f1:.4f}\n")
    
    print("Per-Class F1:")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        is_target = (labels == class_idx).astype(int)
        is_pred = (predictions == class_idx).astype(int)
        f1 = f1_score(is_target, is_pred, zero_division=0)
        support = np.sum(is_target)
        print(f"  {class_name:20s}: F1={f1:.4f} (n={support})")
    
    print("\n" + "="*70)
    
    return macro_f1

def save_optimal_thresholds(thresholds):
    """Save optimal thresholds to JSON for submission code."""
    output_file = ENSEMBLE_DIR / 'optimal_thresholds.json'
    
    # Convert to serializable format
    thresholds_dict = {str(k): float(v) for k, v in thresholds.items()}
    
    with open(output_file, 'w') as f:
        json.dump(thresholds_dict, f, indent=2)
    
    print(f"✓ Saved optimal thresholds to {output_file}")
    
    # Also print as Python code
    print("\nUse these values in code/inference.py:")
    print("\n_CLASS_THRESHOLDS = {")
    for class_idx in range(4):
        threshold = thresholds[class_idx]
        print(f"    {class_idx}: {threshold:.2f},  # {CLASS_NAMES[class_idx]}")
    print("}")

def main():
    print("\n" + "="*70)
    print("ENSEMBLE THRESHOLD CALIBRATION")
    print("="*70 + "\n")
    
    # Load predictions
    probs, labels = load_predictions()
    
    # Find optimal thresholds
    optimal_thresholds, best_f1 = find_optimal_thresholds(probs, labels)
    
    # Evaluate with optimal thresholds
    final_f1 = evaluate_with_thresholds(probs, labels, optimal_thresholds)
    
    # Save thresholds
    save_optimal_thresholds(optimal_thresholds)
    
    print(f"\n✓ Calibration complete!")
    print(f"  Baseline Macro-F1 (argmax): ~0.XX")
    print(f"  After threshold tuning: {final_f1:.4f}")
    
    if final_f1 > 0.85:
        print(f"\n🎉 ENSEMBLE ACHIEVES > 0.85 MACRO-F1!")
        print(f"  Ready for submission!")
    else:
        print(f"\n📊 Ensemble at {final_f1:.4f}. Consider CV averaging or other tricks.")
    
    print(f"\n✓ Next step: python create_ensemble_submission.py")

if __name__ == '__main__':
    main()
