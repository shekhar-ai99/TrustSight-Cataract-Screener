import os
import argparse
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

# Class mapping
CLASS_TO_IDX = {
    'No_Cataract': 0,
    'Immature_Cataract': 1,
    'Mature_Cataract': 2,
    'IOL_Inserted': 3
}

def preprocess_image(img_path):
    """Load, resize, and flatten image to 512x512x3 RGB vector."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((512, 512), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
    img_vector = img_array.flatten().tolist()
    return img_vector

def main(input_dir, output_parquet):
    data = []
    for class_name, label in CLASS_TO_IDX.items():
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: {class_dir} not found, skipping.")
            continue
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img_vector = preprocess_image(img_path)
                    data.append({
                        "ID": f"{class_name}_{img_name}",
                        "image_vector": img_vector,
                        "Cataract Type": class_name.replace("_", " "),
                        "Image Quality": "Good",
                        "Patient Age Group": ">=50"
                    })
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Write to Parquet with compression
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_parquet, compression='snappy')
    print(f"Parquet file saved to {output_parquet}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images to Parquet for CDIS upload.")
    parser.add_argument('--input_dir', required=True, help='Input directory with class subfolders (e.g., data/train)')
    parser.add_argument('--output_parquet', required=True, help='Output Parquet file path')
    args = parser.parse_args()
    main(args.input_dir, args.output_parquet)