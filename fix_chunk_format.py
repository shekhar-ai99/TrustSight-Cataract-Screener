"""
Fix chunk format to match reference dataset schema.
This script transforms all parquet chunks to match the original dataset format.
"""
import pandas as pd
import glob
import os
import sys

# Paths
REFERENCE = os.path.join("dataset", "cataract-training-dataset.parquet")
CHUNKS_PATTERN = os.path.join("dataset", "output_chunks", "*.parquet")
OUTPUT_DIR = os.path.join("dataset", "output_chunks_fixed")

def fix_chunk_format():
    """Fix all chunks to match reference format."""
    
    # Load reference to get expected schema
    print(f"Loading reference dataset: {REFERENCE}")
    df_ref = pd.read_parquet(REFERENCE)
    
    EXPECTED_COLUMNS = list(df_ref.columns)
    EXPECTED_DTYPES = df_ref.dtypes.to_dict()
    
    print("\n" + "="*60)
    print("EXPECTED SCHEMA (from reference)")
    print("="*60)
    print(f"Columns: {EXPECTED_COLUMNS}")
    print(f"Dtypes: {EXPECTED_DTYPES}")
    print("="*60 + "\n")
    
    # Find all chunks
    chunk_files = sorted(glob.glob(CHUNKS_PATTERN))
    
    if not chunk_files:
        print(f"❌ No chunks found at: {CHUNKS_PATTERN}")
        return False
    
    print(f"Found {len(chunk_files)} chunks to fix\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check first chunk to understand the mismatch
    print("Analyzing first chunk to detect format issues...")
    first_chunk = pd.read_parquet(chunk_files[0])
    
    print(f"\nChunk columns: {list(first_chunk.columns)}")
    print(f"Expected columns: {EXPECTED_COLUMNS}")
    
    # Build column mapping
    column_mapping = {}
    
    # Common column name variations
    possible_mappings = {
        'id': ['id', 'ID', 'image_id', 'filename', 'file_name'],
        'image_vector': ['image_vector', 'vector', 'image', 'pixels'],
        'label': ['label', 'Label', 'Cataract Type', 'cataract_type', 'class', 'target']
    }
    
    for expected_col in EXPECTED_COLUMNS:
        found = False
        for chunk_col in first_chunk.columns:
            if chunk_col == expected_col:
                column_mapping[chunk_col] = expected_col
                found = True
                break
            elif chunk_col in possible_mappings.get(expected_col, []):
                column_mapping[chunk_col] = expected_col
                found = True
                print(f"  Mapping: '{chunk_col}' → '{expected_col}'")
                break
        
        if not found:
            print(f"\n❌ Could not find mapping for required column: '{expected_col}'")
            print(f"   Available columns in chunk: {list(first_chunk.columns)}")
            return False
    
    print(f"\nColumn mapping established: {column_mapping}")
    
    # Process all chunks
    print(f"\n{'='*60}")
    print("FIXING CHUNKS")
    print(f"{'='*60}\n")
    
    fixed_count = 0
    total_samples = 0
    
    for chunk_file in chunk_files:
        chunk_name = os.path.basename(chunk_file)
        print(f"Processing: {chunk_name}...", end=" ")
        
        try:
            # Read chunk
            df = pd.read_parquet(chunk_file)
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Select only expected columns in correct order
            df = df[EXPECTED_COLUMNS]
            
            # Ensure correct dtypes (if needed)
            for col in EXPECTED_COLUMNS:
                if col in df.columns and df[col].dtype != EXPECTED_DTYPES[col]:
                    # Try to cast if safe
                    try:
                        df[col] = df[col].astype(EXPECTED_DTYPES[col])
                    except:
                        print(f"⚠️  Could not cast {col} to {EXPECTED_DTYPES[col]}")
            
            # Save fixed chunk
            output_path = os.path.join(OUTPUT_DIR, chunk_name)
            df.to_parquet(output_path, compression='snappy')
            
            fixed_count += 1
            total_samples += len(df)
            print(f"✅ Fixed ({len(df)} samples)")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return False
    
    print(f"\n{'='*60}")
    print("FIX SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully fixed {fixed_count}/{len(chunk_files)} chunks")
    print(f"✅ Total samples: {total_samples}")
    print(f"✅ Fixed chunks saved to: {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Run validation again: python validate_datasets.py")
    print(f"  2. If validation passes, merge and train")
    print(f"{'='*60}\n")
    
    return True

if __name__ == "__main__":
    success = fix_chunk_format()
    sys.exit(0 if success else 1)
