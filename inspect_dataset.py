"""Quick script to inspect parquet dataset schema and content"""
import pandas as pd

parquet_path = "dataset/cataract-training-dataset.parquet"

df = pd.read_parquet(parquet_path, engine="pyarrow")

print("="*70)
print("DATASET SCHEMA INSPECTION")
print("="*70)

print("\n📋 COLUMNS:")
print(df.columns.tolist())

print("\n📊 DATA TYPES:")
print(df.dtypes)

print("\n📈 SHAPE:")
print(f"  Rows: {len(df)}")
print(f"  Columns: {len(df.columns)}")

print("\n🔍 FIRST 5 ROWS:")
print(df.head(5))

print("\n📌 SAMPLE DATA:")
for idx, row in df.head(3).iterrows():
    print(f"\n--- Row {idx} ---")
    for col in df.columns:
        val = row[col]
        if isinstance(val, str) and len(str(val)) > 100:
            print(f"  {col}: {str(val)[:100]}... (truncated)")
        else:
            print(f"  {col}: {val}")

print("\n✅ LABEL VALUE COUNTS:")
if 'label' in df.columns:
    print(df['label'].value_counts())
elif 'Cataract Type' in df.columns:
    print(df['Cataract Type'].value_counts())
else:
    print("No 'label' or 'Cataract Type' column found")

print("\n" + "="*70)
