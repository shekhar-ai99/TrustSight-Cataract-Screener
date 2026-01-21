"""
Resize chunk images to match reference dataset resolution (FAST OPTIMIZED).
"""
import pandas as pd
import glob
import os
import numpy as np
from PIL import Image

REFERENCE = os.path.join("dataset", "cataract-training-dataset.parquet")
CHUNKS_PATTERN = os.path.join("dataset", "output_chunks", "*.parquet")
OUTPUT_DIR = os.path.join("dataset", "output_chunks_resized")

# Load reference to get target resolution
df_ref = pd.read_parquet(REFERENCE)
sample_vec_str = df_ref["image_vector"].iloc[0]

# Determine reference resolution from vector length
num_values = len(sample_vec_str.split(','))
print(f"Reference vector has {num_values} values")

# Assume 3 channels (RGB)
if num_values % 3 == 0:
    pixels = num_values // 3
    h = w = int(np.sqrt(pixels))
    print(f"Reference resolution: {h}×{w}×3")
    TARGET_RES = (h, w)
else:
    print(f"Could not determine resolution from {num_values} values")
    exit(1)

print(f"Target resolution: {TARGET_RES}\n")

# Find chunks
chunk_files = sorted(glob.glob(CHUNKS_PATTERN))
print(f"Found {len(chunk_files)} chunks to resize\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

total_samples = 0

for chunk_idx, chunk_file in enumerate(chunk_files, 1):
    chunk_name = os.path.basename(chunk_file)
    print(f"[{chunk_idx}/{len(chunk_files)}] {chunk_name}...", end=" ", flush=True)
    
    try:
        df = pd.read_parquet(chunk_file)
        
        # Vectorized resize using numpy (much faster than apply)
        def resize_batch(vectors):
            """Fast batch resize of image vectors."""
            resized = []
            for vec_str in vectors:
                try:
                    # Parse vector
                    values = np.fromstring(vec_str, sep=',', dtype=np.float32)
                    
                    # Get source resolution
                    if len(values) % 3 == 0:
                        pixels = len(values) // 3
                        h = w = int(np.sqrt(pixels))
                        
                        # Reshape to image
                        img_array = values.reshape(h, w, 3)
                        
                        # Convert to uint8 if needed
                        if img_array.max() <= 1.0:
                            img_array = (img_array * 255).astype(np.uint8)
                        else:
                            img_array = img_array.astype(np.uint8)
                        
                        # Create PIL image and resize
                        img = Image.fromarray(img_array)
                        img_resized = img.resize(TARGET_RES, Image.Resampling.LANCZOS)
                        
                        # Convert back to array and flatten
                        resized_array = np.array(img_resized, dtype=np.float32) / 255.0
                        resized_vector = ','.join(np.char.mod('%.8f', resized_array.flatten()))
                        
                        resized.append(resized_vector)
                    else:
                        resized.append(vec_str)
                except Exception as e:
                    print(f"Error resizing: {e}", flush=True)
                    resized.append(vec_str)
            return resized
        
        # Resize all vectors in chunk
        df['image_vector'] = resize_batch(df['image_vector'].tolist())
        
        # Save
        output_path = os.path.join(OUTPUT_DIR, chunk_name)
        df.to_parquet(output_path, compression='snappy')
        
        total_samples += len(df)
        print(f"[OK] ({len(df)} samples)", flush=True)
        
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)

print("")
print("="*60)
print(f"[SUCCESS] Total samples resized: {total_samples}")
print(f"[SUCCESS] Output: {OUTPUT_DIR}")
print("="*60)
