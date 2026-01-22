#!/usr/bin/env python
"""
Create final submission with ensemble of 3 models.
Packages code, models, and optimal thresholds into submission format.
"""
import sys
import json
import shutil
import zipfile
from pathlib import Path

ENSEMBLE_DIR = Path('ensemble_models')
SUBMISSION_DIR = Path('test_submission')
CODE_DIR = Path('code')

def load_optimal_thresholds():
    """Load optimal thresholds from calibration."""
    threshold_file = ENSEMBLE_DIR / 'optimal_thresholds.json'
    
    if not threshold_file.exists():
        print(f"❌ Thresholds not found. Run calibrate_ensemble_thresholds.py first.")
        sys.exit(1)
    
    with open(threshold_file) as f:
        thresholds = json.load(f)
    
    # Convert keys back to integers
    return {int(k): float(v) for k, v in thresholds.items()}

def update_inference_code(thresholds):
    """Update inference.py with optimal thresholds."""
    inference_path = CODE_DIR / 'inference.py'
    
    if not inference_path.exists():
        print(f"❌ inference.py not found at {inference_path}")
        return False
    
    # Read current code
    with open(inference_path) as f:
        content = f.read()
    
    # Find and replace _CLASS_THRESHOLDS
    import re
    pattern = r'_CLASS_THRESHOLDS = \{[^}]*\}'
    
    replacement = '_CLASS_THRESHOLDS = {\n'
    for class_idx in range(4):
        threshold = thresholds[class_idx]
        class_names = ["No Cataract", "Immature Cataract", "Mature Cataract", "IOL Inserted"]
        replacement += f"    {class_idx}: {threshold:.2f},  # {class_names[class_idx]}\n"
    replacement += '}'
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write updated code
    with open(inference_path, 'w') as f:
        f.write(new_content)
    
    print(f"✓ Updated inference.py with optimal thresholds")
    return True

def copy_ensemble_models():
    """Copy ensemble models to submission directory."""
    submission_models_dir = SUBMISSION_DIR / 'code' / 'ensemble_models'
    submission_models_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy each fold
    for fold in [1, 2, 3]:
        fold_dir = ENSEMBLE_DIR / f'fold_{fold}'
        if fold_dir.exists():
            dst_fold = submission_models_dir / f'fold_{fold}'
            if dst_fold.exists():
                shutil.rmtree(dst_fold)
            shutil.copytree(fold_dir, dst_fold)
            print(f"  ✓ Copied fold_{fold}")
    
    return True

def copy_submission_files():
    """Copy required submission files."""
    # Make sure submission dir exists
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy inference.py
    src_inference = CODE_DIR / 'inference.py'
    dst_inference = SUBMISSION_DIR / 'code' / 'inference.py'
    dst_inference.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_inference, dst_inference)
    print(f"✓ Copied inference.py")
    
    # Copy requirements.txt
    src_req = Path('requirements.txt')
    dst_req = SUBMISSION_DIR / 'requirements.txt'
    if src_req.exists():
        shutil.copy(src_req, dst_req)
        print(f"✓ Copied requirements.txt")
    
    # Create README
    readme_content = """# Ensemble Cataract Classifier - 3-Fold Submission

## Architecture
- **Base Model**: ResNet18 (pretrained on ImageNet)
- **Ensemble**: 3 models trained with different random seeds
- **Strategy**: Soft voting (average softmax probabilities)

## Training
- **Backbone Freeze**: 12 epochs (FIX-6)
- **Class Weighting**: Inverse frequency (FIX-2)
- **Label Smoothing**: 0.1 (FIX-8)
- **Gradient Clipping**: norm=1.0 (FIX-9)
- **Stratified Split**: Ensures balanced class distribution (FIX-1)

## Inference
- **Test-Time Augmentation**: 4-flip averaging
- **Class-Specific Thresholds**: Calibrated on validation set
- **Soft Voting**: Predictions averaged from 3 models

## Model Files
- `ensemble_models/fold_1/model.pth`: Model 1 (seed=42)
- `ensemble_models/fold_2/model.pth`: Model 2 (seed=123)
- `ensemble_models/fold_3/model.pth`: Model 3 (seed=456)

## Key Fixes Applied
1. Stratified split (prevents data leakage)
2. Inverse-frequency class weights (handles imbalance)
3. Resolution alignment (512×512 throughout)
4. IOL in every batch (WeightedRandomSampler)
5. Extended backbone freeze (stable feature learning)
6. Restricted MixUp (skips when IOL present)
7. Label smoothing (reduces overconfidence)
8. Gradient clipping (prevents exploding gradients)
9. Soft voting ensemble (reduces variance)

## Expected Performance
- Single model: ~0.82-0.84 macro-F1
- Ensemble: ~0.85-0.87 macro-F1
"""
    
    readme_path = SUBMISSION_DIR / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"✓ Created README.md")

def create_submission_zip():
    """Create submission ZIP file."""
    zip_name = 'ensemble_submission.zip'
    
    # Remove old zip if exists
    if Path(zip_name).exists():
        Path(zip_name).unlink()
    
    # Create new zip
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in SUBMISSION_DIR.rglob('*'):
            if file.is_file():
                arcname = file.relative_to(SUBMISSION_DIR)
                zf.write(file, arcname)
    
    zip_size = Path(zip_name).stat().st_size / (1024*1024)
    print(f"✓ Created submission ZIP: {zip_name} ({zip_size:.1f} MB)")

def main():
    print("\n" + "="*70)
    print("CREATE ENSEMBLE SUBMISSION")
    print("="*70 + "\n")
    
    # Load optimal thresholds
    print("Loading optimal thresholds...")
    thresholds = load_optimal_thresholds()
    print(f"✓ Loaded thresholds:")
    class_names = ["No Cataract", "Immature Cataract", "Mature Cataract", "IOL Inserted"]
    for class_idx in range(4):
        print(f"   {class_names[class_idx]:20s}: {thresholds[class_idx]:.2f}")
    
    # Update inference code
    print("\nUpdating inference code...")
    if not update_inference_code(thresholds):
        sys.exit(1)
    
    # Copy submission files
    print("\nPreparing submission structure...")
    copy_submission_files()
    
    # Copy ensemble models
    print("\nCopying ensemble models...")
    copy_ensemble_models()
    
    # Create ZIP
    print("\nCreating submission ZIP...")
    create_submission_zip()
    
    print("\n" + "="*70)
    print("SUBMISSION READY!")
    print("="*70)
    print(f"\nFiles prepared in: {SUBMISSION_DIR}/")
    print(f"ZIP file: ensemble_submission.zip")
    print(f"\nDirectory structure:")
    print("  test_submission/")
    print("    ├── code/")
    print("    │   ├── inference.py (with optimal thresholds)")
    print("    │   └── ensemble_models/")
    print("    │       ├── fold_1/model.pth")
    print("    │       ├── fold_2/model.pth")
    print("    │       └── fold_3/model.pth")
    print("    ├── requirements.txt")
    print("    └── README.md")

if __name__ == '__main__':
    main()
