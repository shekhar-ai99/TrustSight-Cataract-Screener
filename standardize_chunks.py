"""
Standardize all chunks to match reference dataset schema.
"""
import pandas as pd
import glob
import os

REFERENCE = os.path.join("dataset", "cataract-training-dataset.parquet")
CHUNKS_PATTERN = os.path.join("dataset", "output_chunks", "*.parquet")
OUTPUT_DIR = os.path.join("dataset", "output_chunks")

# Load reference to get target schema
df_ref = pd.read_parquet(REFERENCE)
ref_columns = list(df_ref.columns)
ref_dtypes = dict(df_ref.dtypes)

print(f"Reference Schema:")
print(f"  Columns: {ref_columns}")
print(f"  Dtypes: {ref_dtypes}")
print()

# Find chunks
chunk_files = sorted(glob.glob(CHUNKS_PATTERN))
print(f"Found {len(chunk_files)} chunks\n")

processed = 0
failed = []

for chunk_idx, chunk_file in enumerate(chunk_files, 1):
    chunk_name = os.path.basename(chunk_file)
    print(f"[{chunk_idx}/{len(chunk_files)}] {chunk_name}...", end=" ", flush=True)
    
    try:
        df = pd.read_parquet(chunk_file)
        
        # Check if schema matches
        current_cols = list(df.columns)
        current_dtypes = dict(df.dtypes)
        
        needs_fix = False
        
        # Fix columns if needed
        if current_cols != ref_columns:
            print(f"\n  Fixing columns: {current_cols} -> {ref_columns}", flush=True)
            df = df[ref_columns]
            needs_fix = True
        
        # Ensure dtypes (convert to object if needed)
        for col in ref_columns:
            if col not in df.columns:
                print(f"  WARNING: Missing column {col}", flush=True)
                failed.append((chunk_name, f"Missing column {col}"))
                continue
            
            if df[col].dtype != 'object':
                df[col] = df[col].astype('object')
                needs_fix = True
        
        # Save if changed
        if needs_fix:
            print(f"  Saving standardized version...", end=" ", flush=True)
            df.to_parquet(chunk_file, compression='snappy')
        
        print(f"[OK]", flush=True)
        processed += 1
        
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        failed.append((chunk_name, str(e)))

print()
print("="*60)
print(f"[SUCCESS] Processed: {processed}/{len(chunk_files)}")
if failed:
    print(f"[WARNING] Failed: {len(failed)}")
    for fname, error in failed:
        print(f"  - {fname}: {error}")
else:
    print(f"[SUCCESS] All chunks standardized!")
print("="*60)
