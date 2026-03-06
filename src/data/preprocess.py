import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import argparse

def resize_dataset(input_csv, input_root, output_root, new_size=320):
    """
    Resizes all images in the dataset and creates a new CSV.
    """
    df = pd.read_csv(input_csv)
    
    # Create output directory structure
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        
    new_paths = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        rel_path = row['Path'] # e.g., CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
        
        # Absolute path for source
        src_path = os.path.join(input_root, rel_path)
        
        # New relative path - preserve directory structure but change root folder name
        # e.g. CheXpert-v1.0-small/train/patient... -> resized/train/patient...
        parts = rel_path.split(os.sep)
        if len(parts) > 1:
            # Reconstruct path without the top-level folder 'CheXpert-v1.0-small'
            # Or just keep structure inside output_root
            new_rel_path = rel_path
        else:
            new_rel_path = rel_path
            
        dst_path = os.path.join(output_root, new_rel_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        try:
            with Image.open(src_path) as img:
                img = img.resize((new_size, new_size), Image.Resampling.LANCZOS)
                img.save(dst_path)
            new_paths.append(new_rel_path)
        except Exception as e:
            # print(f"Error processing {src_path}: {e}")
            new_paths.append(rel_path) # Keep original if failed
            
    # Save new CSV
    df['Path'] = new_paths
    output_csv = os.path.join(output_root, 'train.csv')
    df.to_csv(output_csv, index=False)
    print(f"Saved resized dataset info to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--input_root', type=str, default='.') # Root where CheXpert folder is located
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--size', type=int, default=320)
    args = parser.parse_args()
    
    resize_dataset(args.input_csv, args.input_root, args.output_root, args.size)
