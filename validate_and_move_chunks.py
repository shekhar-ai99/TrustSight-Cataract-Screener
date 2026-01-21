"""
Validate all chunks against reference schema and move matching ones to output_chunks_resized.
"""
import pandas as pd
import glob
import os
import shutil

REFERENCE = os.path.join("dataset", "cataract-training-dataset.parquet")
CHUNKS_PATTERN = os.path.join("dataset", "output_chunks", "*.parquet")
SOURCE_DIR = os.path.join("dataset", "output_chunks")
OUTPUT_DIR = os.path.join("dataset", "output_chunks_resized")

# Load reference to get target schema
df_ref = pd.read_parquet(REFERENCE)
ref_columns = list(df_ref.columns)
ref_dtypes = dict(df_ref.dtypes)
ref_vec_len = len(df_ref["image_vector"].iloc[0].split(','))

print(f"Reference Schema:")
print(f"  Columns: {ref_columns}")
print(f"  Dtypes: {ref_dtypes}")
print(f"  Vector length: {ref_vec_len}")
print()

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find chunks
chunk_files = sorted(glob.glob(CHUNKS_PATTERN))
print(f"Found {len(chunk_files)} chunks\n")

valid = []
invalid = []
total_samples = 0

for chunk_idx, chunk_file in enumerate(chunk_files, 1):
    chunk_name = os.path.basename(chunk_file)
    print(f"[{chunk_idx}/{len(chunk_files)}] {chunk_name}...", end=" ", flush=True)
    
    try:
        df = pd.read_parquet(chunk_file)
        
        # Validate schema
        is_valid = True
        issues = []
        
        # Check columns
        if list(df.columns) != ref_columns:
            is_valid = False
            issues.append(f"Columns mismatch: {list(df.columns)}")
        
        # Check dtypes
        for col in ref_columns:
            if col not in df.columns:
                is_valid = False
                issues.append(f"Missing column: {col}")
            elif df[col].dtype != 'object':
                is_valid = False
                issues.append(f"Dtype mismatch for {col}: {df[col].dtype}")
        
        # Check vector length
        chunk_vec_len = len(df["image_vector"].iloc[0].split(','))
        if chunk_vec_len != ref_vec_len:
            is_valid = False
            issues.append(f"Vector length mismatch: {chunk_vec_len} vs {ref_vec_len}")
        
        # Check labels
        ref_labels = set(df_ref['label'].unique())
        chunk_labels = set(df['label'].unique())
        if not chunk_labels.issubset(ref_labels):
            is_valid = False
            issues.append(f"Unknown labels: {chunk_labels - ref_labels}")
        
        if is_valid:
            # Move to resized folder
            dest_path = os.path.join(OUTPUT_DIR, chunk_name)
            shutil.copy2(chunk_file, dest_path)
            print(f"[VALID -> MOVED] ({len(df)} samples)", flush=True)
            valid.append(chunk_name)
            total_samples += len(df)
        else:
            print(f"[INVALID] {'; '.join(issues)}", flush=True)
            invalid.append((chunk_name, issues))
        
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        invalid.append((chunk_name, [str(e)]))

print()
print("="*60)
print(f"[VALID] {len(valid)}/{len(chunk_files)} chunks")
print(f"[TOTAL SAMPLES] {total_samples}")
print(f"[OUTPUT] {OUTPUT_DIR}")
if invalid:
    print(f"[INVALID] {len(invalid)} chunks:")
    for fname, issues in invalid:
        print(f"  - {fname}: {'; '.join(issues)}")
else:
    print(f"[SUCCESS] All chunks are valid and moved!")
print("="*60)
