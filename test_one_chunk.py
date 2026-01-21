"""
Test single chunk: diagnose format, resize, validate, compare.
"""
import pandas as pd
import numpy as np
from PIL import Image
import os

REFERENCE = os.path.join("dataset", "cataract-training-dataset.parquet")
TEST_CHUNK = os.path.join("dataset", "output_chunks", "part_001.parquet")
OUTPUT_DIR = os.path.join("dataset", "output_chunks_resized")

# Load reference
print("[1] Loading reference dataset...")
df_ref = pd.read_parquet(REFERENCE)
ref_sample = df_ref["image_vector"].iloc[0]
ref_vec_len = len(ref_sample.split(','))
ref_pixels = ref_vec_len // 3
ref_h = ref_w = int(np.sqrt(ref_pixels))
print(f"    Reference: {ref_h}x{ref_w}x3, vector length: {ref_vec_len}")

# Load chunk BEFORE resize
print(f"\n[2] Loading {TEST_CHUNK} (BEFORE resize)...")
df_chunk = pd.read_parquet(TEST_CHUNK)
chunk_sample = df_chunk["image_vector"].iloc[0]
chunk_vec_len = len(chunk_sample.split(','))
chunk_pixels = chunk_vec_len // 3
chunk_h = chunk_w = int(np.sqrt(chunk_pixels))
print(f"    Chunk (before): {chunk_h}x{chunk_w}x3, vector length: {chunk_vec_len}")
print(f"    Status: MISMATCH (need to resize {chunk_h}x{chunk_w} -> {ref_h}x{ref_w})")

# Resize function
print(f"\n[3] Resizing chunk to match reference...")
def resize_vector(vec_str, target_res):
    """Resize single image vector."""
    values = np.fromstring(vec_str, sep=',', dtype=np.float32)
    
    if len(values) % 3 == 0:
        pixels = len(values) // 3
        h = w = int(np.sqrt(pixels))
        
        img_array = values.reshape(h, w, 3)
        
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
        
        img = Image.fromarray(img_array)
        img_resized = img.resize(target_res, Image.Resampling.LANCZOS)
        
        resized_array = np.array(img_resized, dtype=np.float32) / 255.0
        resized_vector = ','.join(np.char.mod('%.8f', resized_array.flatten()))
        
        return resized_vector
    return vec_str

# Apply resize to all samples in chunk
df_chunk['image_vector'] = df_chunk['image_vector'].apply(lambda x: resize_vector(x, (ref_h, ref_w)))

# Verify after resize
chunk_sample_after = df_chunk["image_vector"].iloc[0]
chunk_vec_len_after = len(chunk_sample_after.split(','))
print(f"    Chunk (after): vector length: {chunk_vec_len_after}")
if chunk_vec_len_after == ref_vec_len:
    print(f"    [OK] Vector length matches reference!")
else:
    print(f"    [ERROR] Vector length mismatch: {chunk_vec_len_after} vs {ref_vec_len}")

# Save resized chunk
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "part_001_resized.parquet")
df_chunk.to_parquet(output_path, compression='snappy')
print(f"\n[4] Saved resized chunk to {output_path}")

# Validate resized chunk
print(f"\n[5] Validating resized chunk...")
df_resized = pd.read_parquet(output_path)
resized_sample = df_resized["image_vector"].iloc[0]
resized_vec_len = len(resized_sample.split(','))
resized_pixels = resized_vec_len // 3
resized_h = resized_w = int(np.sqrt(resized_pixels))
print(f"    Resized chunk: {resized_h}x{resized_w}x3, vector length: {resized_vec_len}")

# Final comparison
print(f"\n[6] FINAL COMPARISON:")
print(f"    Reference:     {ref_h}x{ref_w}, len={ref_vec_len}")
print(f"    Chunk (before): {chunk_h}x{chunk_w}, len={chunk_vec_len}")
print(f"    Chunk (after):  {resized_h}x{resized_w}, len={resized_vec_len}")
if resized_vec_len == ref_vec_len and resized_h == ref_h and resized_w == ref_w:
    print(f"    [SUCCESS] Formats match!")
else:
    print(f"    [FAILED] Formats do not match")
