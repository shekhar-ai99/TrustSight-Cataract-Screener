#!/usr/bin/env python
"""
Train 3 models with different random seeds for ensemble.
Automatically manages config changes and tracks all 3 runs.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / 'src'
CONFIG_PATH = SRC_DIR / 'config.py'
ENSEMBLE_DIR = PROJECT_ROOT / 'ensemble_models'

# Seeds for 3 diverse models
SEEDS = [42, 123, 456]

# Save original config
ORIGINAL_CONFIG_BACKUP = CONFIG_PATH.with_suffix('.py.backup')

def backup_config():
    """Backup original config before modifications."""
    if not ORIGINAL_CONFIG_BACKUP.exists():
        shutil.copy(CONFIG_PATH, ORIGINAL_CONFIG_BACKUP)
        print(f"✓ Backed up original config to {ORIGINAL_CONFIG_BACKUP}")

def restore_config():
    """Restore original config."""
    if ORIGINAL_CONFIG_BACKUP.exists():
        shutil.copy(ORIGINAL_CONFIG_BACKUP, CONFIG_PATH)
        print(f"✓ Restored original config")

def set_random_seed(seed: int):
    """Update RANDOM_SEED in config.py if it exists, otherwise add it."""
    with open(CONFIG_PATH, 'r') as f:
        content = f.read()
    
    # Check if RANDOM_SEED already exists
    if 'RANDOM_SEED =' in content:
        # Replace existing
        import re
        content = re.sub(r'RANDOM_SEED\s*=\s*\d+', f'RANDOM_SEED = {seed}', content)
    else:
        # Add after imports/docstring
        lines = content.split('\n')
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.startswith('# '):
                insert_pos = i + 1
        lines.insert(insert_pos, f'RANDOM_SEED = {seed}  # For ensemble diversity')
        content = '\n'.join(lines)
    
    with open(CONFIG_PATH, 'w') as f:
        f.write(content)
    print(f"  ✓ Set RANDOM_SEED = {seed} in config")

def run_training(seed: int, fold: int):
    """Run training with specified seed."""
    print(f"\n{'='*70}")
    print(f"FOLD {fold}/3: Training with seed={seed}")
    print(f"{'='*70}\n")
    
    set_random_seed(seed)
    
    # Run training
    result = subprocess.run(
        [sys.executable, 'train.py'],
        cwd=SRC_DIR,
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"❌ Training failed for fold {fold}")
        return False
    
    return True

def backup_model(seed: int, fold: int):
    """Back up the trained model to ensemble_models/fold_X/"""
    ensemble_dir = ENSEMBLE_DIR
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = ensemble_dir / f'fold_{fold}'
    fold_dir.mkdir(exist_ok=True)
    
    # Copy model
    src_model = SRC_DIR.parent / 'best_model' / 'model.pth'
    dst_model = fold_dir / 'model.pth'
    
    if src_model.exists():
        shutil.copy(src_model, dst_model)
        print(f"  ✓ Saved model to {dst_model}")
        
        # Save metadata
        metadata = {
            'seed': seed,
            'fold': fold,
            'timestamp': datetime.now().isoformat(),
            'model_path': str(dst_model)
        }
        with open(fold_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True
    else:
        print(f"  ❌ Model not found at {src_model}")
        return False

def main():
    print(f"\n{'='*70}")
    print("ENSEMBLE TRAINING PIPELINE: 3-Fold with Different Seeds")
    print(f"{'='*70}\n")
    
    # Setup
    ensemble_dir = ENSEMBLE_DIR
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    backup_config()
    
    # Train 3 models
    successful_folds = []
    for fold, seed in enumerate(SEEDS, 1):
        try:
            success = run_training(seed, fold)
            if success:
                backup_model(seed, fold)
                successful_folds.append(fold)
                print(f"\n✓ Fold {fold} complete!")
            else:
                print(f"\n❌ Fold {fold} failed, continuing...")
        except Exception as e:
            print(f"\n❌ Error in fold {fold}: {e}")
    
    # Restore config
    restore_config()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"ENSEMBLE TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Successfully trained: {len(successful_folds)}/3 folds")
    
    if successful_folds:
        print(f"\nModels saved in: {ensemble_dir}")
        print(f"Next step: python ensemble_inference.py")
    else:
        print("❌ No models trained. Check logs above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
