import os
import pandas as pd
import numpy as np
from PIL import Image
import ast

# Class names
CLASS_NAMES = ["No_Cataract", "Immature_Cataract", "Mature_Cataract", "IOL_Inserted"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def main(parquet_path, output_dir):
    # Read Parquet
    df = pd.read_parquet(parquet_path)
    
    # Create output directories
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    
    for idx, row in df.iterrows():
        image_vector = row['image_vector']
        label = row['label']
        if isinstance(label, str):
            label = CLASS_TO_IDX.get(label, 0)  # default to 0 if not found
        class_name = CLASS_NAMES[label]
        
        # Handle image_vector
        if isinstance(image_vector, str):
            try:
                image_vector = ast.literal_eval(image_vector)  # if it's a string list
            except:
                raise ValueError(f"Cannot parse image_vector: {type(image_vector)}")
        
        img_array = np.array(image_vector).reshape(512, 512, 3)
        # Scale to 0-255 if needed
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Save as PNG
        img_path = os.path.join(output_dir, class_name, f"{idx}.png")
        img.save(img_path)
    
    print(f"Extracted {len(df)} images to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', default='test/cataract-sample.parquet')
    parser.add_argument('--output', default='data/train')
    args = parser.parse_args()
    main(args.parquet, args.output)