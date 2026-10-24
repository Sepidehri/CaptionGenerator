"""
Download Flickr8k dataset from Kaggle
"""
import os
import subprocess
import zipfile
from pathlib import Path

def download_flickr8k():
    """Download Flickr8k dataset using Kaggle API"""
    
    # Create data directory
    data_dir = Path('data/flickr8k')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading Flickr8k dataset from Kaggle...")
    print("Make sure you have kaggle.json in ~/.kaggle/")
    print("Get your API token from: https://www.kaggle.com/settings/account\n")
    
    try:
        # Download using Kaggle API
        subprocess.run([
            'kaggle', 'datasets', 'download', 
            '-d', 'adityajn105/flickr8k',
            '-p', str(data_dir)
        ], check=True)
        
        # Extract zip file
        zip_path = data_dir / 'flickr8k.zip'
        if zip_path.exists():
            print("\nExtracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Remove zip file
            zip_path.unlink()
            print("✓ Dataset downloaded and extracted successfully!")
            
            # Verify structure
            images_dir = data_dir / 'Images'
            captions_file = data_dir / 'captions.txt'
            
            if images_dir.exists() and captions_file.exists():
                num_images = len(list(images_dir.glob('*.jpg')))
                print(f"\n✓ Found {num_images} images")
                print(f"✓ Found captions file")
                print(f"\nDataset ready at: {data_dir}")
            else:
                print("\n⚠ Warning: Expected files not found. Check dataset structure.")
                
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nAlternative: Download manually from:")
        print("https://www.kaggle.com/datasets/adityajn105/flickr8k")
        print(f"And extract to: {data_dir}")
        
    except FileNotFoundError:
        print("\n✗ Kaggle CLI not found. Install it with:")
        print("pip install kaggle")
        print("\nOr download manually from:")
        print("https://www.kaggle.com/datasets/adityajn105/flickr8k")

if __name__ == '__main__':
    download_flickr8k()
