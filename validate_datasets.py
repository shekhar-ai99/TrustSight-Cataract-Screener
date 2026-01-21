"""
Dataset Format Validation Script
Validates that all new parquet chunks match the original dataset format exactly.
Run this BEFORE training with new data.
"""
import pandas as pd
import glob
import os
import sys

# Paths
REFERENCE = os.path.join("dataset", "cataract-training-dataset.parquet")
NEW_DATASETS_PATTERN = os.path.join("dataset", "output_chunks_fixed", "*.parquet")

def validate_datasets():
    """Validate all new datasets against reference format."""
    
    # Check if reference exists
    if not os.path.exists(REFERENCE):
        print(f"❌ Reference dataset not found: {REFERENCE}")
        return False
    
    # Load reference
    print(f"Loading reference dataset: {REFERENCE}")
    df_ref = pd.read_parquet(REFERENCE)
    
    EXPECTED_COLUMNS = list(df_ref.columns)
    EXPECTED_DTYPES = df_ref.dtypes.to_dict()
    EXPECTED_LABELS = set(df_ref["label"].unique())
    
    # Get vector length (handle both list and array formats)
    sample_vec = df_ref["image_vector"].iloc[0]
    if isinstance(sample_vec, list):
        EXPECTED_VEC_LEN = len(sample_vec)
    else:
        EXPECTED_VEC_LEN = len(sample_vec) if hasattr(sample_vec, '__len__') else sample_vec.size
    
    print("\n" + "="*60)
    print("REFERENCE DATASET SCHEMA")
    print("="*60)
    print(f"Columns: {EXPECTED_COLUMNS}")
    print(f"Labels: {sorted(EXPECTED_LABELS)}")
    print(f"Image vector length: {EXPECTED_VEC_LEN}")
    print(f"Total samples: {len(df_ref)}")
    print("="*60 + "\n")
    
    # Find new datasets
    new_files = sorted(glob.glob(NEW_DATASETS_PATTERN))
    
    if not new_files:
        print(f"⚠️  No new datasets found at: {NEW_DATASETS_PATTERN}")
        return False
    
    print(f"Found {len(new_files)} new dataset files to validate:\n")
    
    errors = []
    warnings = []
    total_new_samples = 0
    
    for file in new_files:
        print(f"Validating: {os.path.basename(file)}...", end=" ")
        
        try:
            df = pd.read_parquet(file)
            file_errors = []
            
            # Column check
            if list(df.columns) != EXPECTED_COLUMNS:
                file_errors.append("COLUMN_MISMATCH")
                file_errors.append(f"  Expected: {EXPECTED_COLUMNS}")
                file_errors.append(f"  Found: {list(df.columns)}")
            
            # Data type check
            for col in EXPECTED_COLUMNS:
                if col in df.columns and df[col].dtype != EXPECTED_DTYPES[col]:
                    file_errors.append(f"DTYPE_MISMATCH in '{col}'")
                    file_errors.append(f"  Expected: {EXPECTED_DTYPES[col]}")
                    file_errors.append(f"  Found: {df[col].dtype}")
            
            # Label check
            if "label" in df.columns:
                new_labels = set(df["label"].unique())
                if not new_labels.issubset(EXPECTED_LABELS):
                    unknown = new_labels - EXPECTED_LABELS
                    file_errors.append(f"LABEL_MISMATCH: Unknown labels {unknown}")
            
            # Vector length check
            if "image_vector" in df.columns:
                sample_vec = df["image_vector"].iloc[0]
                if isinstance(sample_vec, list):
                    vec_len = len(sample_vec)
                else:
                    vec_len = len(sample_vec) if hasattr(sample_vec, '__len__') else sample_vec.size
                
                if vec_len != EXPECTED_VEC_LEN:
                    file_errors.append(f"VECTOR_LENGTH_MISMATCH")
                    file_errors.append(f"  Expected: {EXPECTED_VEC_LEN}")
                    file_errors.append(f"  Found: {vec_len}")
            
            # Null check
            null_cols = df.columns[df.isnull().any()].tolist()
            if null_cols:
                file_errors.append(f"NULL_VALUES_FOUND in columns: {null_cols}")
            
            # Empty check
            if len(df) == 0:
                warnings.append((file, "EMPTY_FILE"))
            
            if file_errors:
                print("❌ FAILED")
                print("\n" + "="*60)
                print("VALIDATION FAILED - STOPPING EARLY")
                print("="*60)
                print(f"\nFirst failing file: {os.path.basename(file)}")
                print("\nERROR DETAILS:")
                for error in file_errors:
                    print(f"  {error}")
                print(f"\n🔴 DO NOT TRAIN — FIX DATASET FORMAT FIRST")
                print("  All files in output_chunks likely have the same format issue.")
                print("="*60 + "\n")
                return False
            else:
                total_new_samples += len(df)
                print(f"✅ OK ({len(df)} samples)")
        
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            print("\n" + "="*60)
            print("VALIDATION FAILED - STOPPING EARLY")
            print("="*60)
            print(f"\nFirst failing file: {os.path.basename(file)}")
            print(f"\nREAD_ERROR: {str(e)}")
            print(f"\n🔴 DO NOT TRAIN — FIX DATASET FIRST")
            print("="*60 + "\n")
            return False
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for file, warning in warnings:
            print(f"  {os.path.basename(file)}: {warning}")
    
    print(f"\n✅ FORMAT VALIDATION PASSED")
    print(f"\n  Reference dataset: {len(df_ref)} samples")
    print(f"  New datasets: {total_new_samples} samples")
    print(f"  Total after merge: {len(df_ref) + total_new_samples} samples")
    print(f"\n🟢 SAFE TO PROCEED WITH TRAINING")
    print("="*60 + "\n")
    return True

if __name__ == "__main__":
    success = validate_datasets()
    sys.exit(0 if success else 1)
