"""
Merge reference dataset with all validated chunks.
"""
import pandas as pd
import glob
import os

REFERENCE = os.path.join("dataset", "cataract-training-dataset.parquet")
CHUNKS_PATTERN = os.path.join("dataset", "output_chunks_resized", "*.parquet")
OUTPUT_FILE = os.path.join("dataset", "merged_training_dataset.parquet")

print("Loading reference dataset...")
df_ref = pd.read_parquet(REFERENCE)
print(f"  Reference: {df_ref.shape[0]} samples")

# Load all chunks
chunk_files = sorted(glob.glob(CHUNKS_PATTERN))
print(f"\nLoading {len(chunk_files)} chunks...")

dfs = [df_ref]
chunk_total = 0

for chunk_file in chunk_files:
    chunk_name = os.path.basename(chunk_file)
    df_chunk = pd.read_parquet(chunk_file)
    dfs.append(df_chunk)
    chunk_total += len(df_chunk)
    print(f"  {chunk_name}: {len(df_chunk)} samples")

print(f"\nMerging datasets...")
df_merged = pd.concat(dfs, ignore_index=True)

print(f"\nMerged dataset:")
print(f"  Total samples: {df_merged.shape[0]}")
print(f"  Columns: {list(df_merged.columns)}")
print(f"  Label distribution:")
print(df_merged['label'].value_counts().to_string())

print(f"\nSaving to {OUTPUT_FILE}...")
df_merged.to_parquet(OUTPUT_FILE, compression='snappy')

print(f"\n[SUCCESS] Merged dataset saved!")
print(f"  Reference: {df_ref.shape[0]} samples")
print(f"  Chunks: {chunk_total} samples")
print(f"  Total: {df_merged.shape[0]} samples")
