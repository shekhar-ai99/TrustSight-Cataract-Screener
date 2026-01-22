"""Verify submission tarball end-to-end for evaluator compatibility.

Steps performed:
1) Extract the tarball into a fresh temp directory.
2) Validate presence of required files.
3) Load state_dict safely into CataractModel.
4) Run dummy forward (shape check + softmax sum).
5) Run inference.predict on a dummy DataFrame twice (determinism check).
"""
import tarfile
import shutil
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# -------- CONFIG --------
ROOT = Path(__file__).resolve().parents[1]
TAR_PATH = ROOT / "test_submission" / "model_F1_0.6800" / "model.tar.gz"
WORK_DIR = ROOT / "test_submission" / "model_F1_0.6800_verify"
REQUIRED_FILES = ["model.pth", "requirements.txt", "README.md", "code/inference.py"]

# -------- CLEAN / EXTRACT --------
if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)
WORK_DIR.mkdir(parents=True, exist_ok=True)

if not TAR_PATH.exists():
    sys.exit(f"Tarball not found: {TAR_PATH}")

print(f"Extracting {TAR_PATH} -> {WORK_DIR}")
with tarfile.open(TAR_PATH, "r:gz") as tar:
    tar.extractall(WORK_DIR)

# -------- CHECK REQUIRED FILES --------
missing = []
for rel in REQUIRED_FILES:
    if not (WORK_DIR / rel).exists():
        missing.append(rel)
if missing:
    sys.exit(f"Missing required files after extract: {missing}")
print("✓ Required files present")

# Make code importable
sys.path.insert(0, str(WORK_DIR / "code"))
import inference  # noqa: E402

# -------- SAFE LOAD STATE_DICT --------
state = torch.load(WORK_DIR / "model.pth", map_location="cpu")
if not isinstance(state, dict):
    sys.exit(f"Expected state_dict, got {type(state)}")

model = inference.CataractModel()
missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
if missing_keys:
    print(f"⚠️ Missing keys: {missing_keys}")
if unexpected_keys:
    print(f"⚠️ Unexpected keys: {unexpected_keys}")
model.eval()
print("✓ Model state_dict loaded")

# -------- DUMMY FORWARD --------
x = torch.randn(1, 3, 384, 384)
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)

print(f"Logits shape: {logits.shape}")
print(f"Prob sum: {probs.sum(dim=-1)}")

# -------- PREDICT() DET TEST --------
dummy_vector = np.random.rand(512 * 512 * 3).astype("float32")
df = pd.DataFrame({"image_vector": [dummy_vector]})

p1 = inference.predict(df)
p2 = inference.predict(df)
print(f"Predict #1: {p1}")
print(f"Predict #2: {p2}")
if p1.tolist() != p2.tolist():
    print("⚠️ Non-deterministic outputs detected")
else:
    print("✓ Deterministic predict() outputs")

print("\nALL CHECKS COMPLETED")
