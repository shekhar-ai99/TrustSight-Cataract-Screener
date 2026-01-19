#!/usr/bin/env python
"""
Clean cache, pycache, and run training.
Useful before full training runs to ensure clean state.
"""
import os
import shutil
import subprocess
import sys

def clean_pycache(path='.'):
    """Remove all __pycache__ directories"""
    print("🧹 Cleaning __pycache__ directories...")
    for root, dirs, files in os.walk(path):
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                print(f"  ✓ Removed {pycache_path}")
            except Exception as e:
                print(f"  ✗ Failed to remove {pycache_path}: {e}")

def clean_pth_cache(path='../dataset'):
    """Remove cached .pth files (model cache)"""
    print("\n🧹 Cleaning .pth cache files...")
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.pth'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"  ✓ Removed {file_path}")
                except Exception as e:
                    print(f"  ✗ Failed to remove {file_path}: {e}")

def clean_egg_info(path='.'):
    """Remove .egg-info directories"""
    print("\n🧹 Cleaning .egg-info directories...")
    for root, dirs, files in os.walk(path):
        for dir_name in dirs[:]:  # Copy to avoid modification during iteration
            if dir_name.endswith('.egg-info'):
                egg_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(egg_path)
                    print(f"  ✓ Removed {egg_path}")
                except Exception as e:
                    print(f"  ✗ Failed to remove {egg_path}: {e}")

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    print("\n" + "="*70)
    print("CLEANING CACHE BEFORE TRAINING")
    print("="*70)
    
    # Clean caches
    clean_pycache('.')
    clean_pth_cache()
    clean_egg_info('.')
    
    # Clear screen
    clear_screen()
    
    # Run training
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70 + "\n")
    result = subprocess.run([sys.executable, 'train.py'], cwd=os.path.dirname(os.path.abspath(__file__)))
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()
