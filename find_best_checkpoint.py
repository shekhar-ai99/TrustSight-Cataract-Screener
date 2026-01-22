#!/usr/bin/env python
"""
Find the best checkpoint (highest macro-F1) from training logs.
Run after training completes.
"""
import os
import re
from pathlib import Path

# Find latest training run
run_dir = Path('run_reports')
latest_run = max(run_dir.glob('run_*'), key=os.path.getctime)
log_file = latest_run / 'training.log'

print(f"Analyzing: {log_file}\n")

# Parse per-epoch F1 scores
pattern = r'Epoch (\d+).*?Macro-F1: ([\d.]+)'
epochs_f1 = []

with open(log_file) as f:
    content = f.read()
    matches = re.findall(pattern, content)
    
    for epoch, macro_f1 in matches:
        epochs_f1.append((int(epoch), float(macro_f1)))

if not epochs_f1:
    print("❌ Could not parse F1 scores from log. Check format.")
    exit(1)

# Find best epoch (highest macro-F1)
best_epoch, best_f1 = max(epochs_f1, key=lambda x: x[1])

print(f"✓ Found {len(epochs_f1)} epochs\n")
print("Top 5 epochs by Macro-F1:")
for epoch, f1 in sorted(epochs_f1, key=lambda x: -x[1])[:5]:
    marker = " ← BEST" if epoch == best_epoch else ""
    print(f"  Epoch {epoch:2d}: {f1:.4f}{marker}")

print(f"\n🏆 BEST CHECKPOINT: Epoch {best_epoch} (Macro-F1: {best_f1:.4f})")
print(f"\nℹ️  If this F1 > 0.85, you're done!")
print(f"ℹ️  If this F1 < 0.85, proceed to threshold calibration.")
