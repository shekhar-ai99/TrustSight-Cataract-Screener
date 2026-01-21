"""
Validate all chunks match reference format (quick check).
"""
import pandas as pd
import numpy as np
import glob
import os

REFERENCE = os.path.join("dataset", "cataract-training-dataset.parquet")
CHUNKS_PATTERN = os.path.join("dataset", "output_chunks", "*.parquet")

# Load reference
df_ref = pd.read_parquet(REFERENCE)
ref_sample = df_ref["image_vector"].iloc[0]
ref_vec_len = len(ref_sample.split(','))
ref_pixels = ref_vec_len // 3
ref_h = ref_w = int(np.sqrt(ref_pixels))

print(f"Reference format: {ref_h}x{ref_w}x3 (vector length: {ref_vec_len})")
print(f"Validating {len(glob.glob(CHUNKS_PATTERN))} chunks...\n")

chunk_files = sorted(glob.glob(CHUNKS_PATTERN))
all_match = True
total_samples = 0

for chunk_file in chunk_files:
    chunk_name = os.path.basename(chunk_file)
    df_chunk = pd.read_parquet(chunk_file)
    
    sample = df_chunk["image_vector"].iloc[0]
    vec_len = len(sample.split(','))
    pixels = vec_len // 3
    h = w = int(np.sqrt(pixels))
    
    matches = (vec_len == ref_vec_len and h == ref_h and w == ref_w)
    status = "[OK]" if matches else "[MISMATCH]"
    
    print(f"{status} {chunk_name:20} {h:4}x{w:4}, samples: {len(df_chunk):4}")
    total_samples += len(df_chunk)
    
    if not matches:
        all_match = False
        print(f"     Expected: {ref_h}x{ref_w}, got {h}x{w}")

print(f"\nTotal samples: {total_samples}")
if all_match:
    print("[SUCCESS] All chunks match reference format - READY TO USE!")
else:
    print("[WARNING] Some chunks have format mismatches")
