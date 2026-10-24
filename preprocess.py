"""
Preprocess Flickr8k dataset - build vocabulary and clean captions
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from collections import Counter
from data.vocab import Vocabulary
import re
from tqdm import tqdm


def clean_caption(caption):
    """Clean and normalize caption text"""
    # Convert to lowercase
    caption = caption.lower()
    
    # Remove special characters except apostrophes
    caption = re.sub(r"[^a-z' ]+", '', caption)
    
    # Remove extra whitespace
    caption = re.sub(r'\s+', ' ', caption).strip()
    
    return caption

def build_vocabulary(captions_file, min_freq=5):
    """Build vocabulary from captions"""
    
    print("Loading captions...")
    df = pd.read_csv(captions_file)
    
    print(f"Total captions: {len(df)}")
    
    # Clean captions
    print("Cleaning captions...")
    df['caption'] = df['caption'].apply(clean_caption)
    
    # Count word frequencies
    print("Counting word frequencies...")
    word_freq = Counter()
    for caption in tqdm(df['caption']):
        tokens = caption.split()
        word_freq.update(tokens)
    
    print(f"\nTotal unique words: {len(word_freq)}")
    print(f"Words appearing >= {min_freq} times: {sum(1 for count in word_freq.values() if count >= min_freq)}")
    
    # Build vocabulary
    vocab = Vocabulary()
    
    # Add special tokens
    vocab.add_word('<PAD>')
    vocab.add_word('<START>')
    vocab.add_word('<END>')
    vocab.add_word('<UNK>')
    
    # Add words meeting frequency threshold
    for word, count in word_freq.items():
        if count >= min_freq:
            vocab.add_word(word)
    
    print(f"\nVocabulary size: {len(vocab)}")
    
    # Calculate coverage
    total_words = sum(word_freq.values())
    covered_words = sum(count for word, count in word_freq.items() if word in vocab.word2idx)
    coverage = covered_words / total_words * 100
    print(f"Vocabulary coverage: {coverage:.2f}%")
    
    # Save cleaned captions
    output_dir = Path('data/flickr8k/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / 'captions_clean.csv', index=False)
    print(f"\n✓ Saved cleaned captions to {output_dir}/captions_clean.csv")
    
    # Save vocabulary
    with open(output_dir / 'vocabulary.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print(f"✓ Saved vocabulary to {output_dir}/vocabulary.pkl")
    
    # Print statistics
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Total images: {df['image'].nunique()}")
    print(f"Total captions: {len(df)}")
    print(f"Captions per image: {len(df) / df['image'].nunique():.1f}")
    
    # Caption length statistics
    lengths = df['caption'].apply(lambda x: len(x.split()))
    print(f"\nCaption length statistics:")
    print(f"  Mean: {lengths.mean():.2f} words")
    print(f"  Median: {lengths.median():.0f} words")
    print(f"  Min: {lengths.min()} words")
    print(f"  Max: {lengths.max()} words")
    print(f"  Std: {lengths.std():.2f} words")
    
    # Most common words
    print(f"\nTop 20 most common words:")
    for word, count in word_freq.most_common(20):
        print(f"  {word}: {count}")
    
    return vocab, df

def split_dataset(df, train_split=0.8, val_split=0.1):
    """Split dataset into train, validation, and test sets"""
    
    # Get unique images
    unique_images = np.array(df['image'].unique())  # writable copy, same order
    np.random.seed(42)
    np.random.shuffle(unique_images)
    
    # Calculate split points
    n_train = int(len(unique_images) * train_split)
    n_val = int(len(unique_images) * val_split)
    
    # Split images
    train_images = unique_images[:n_train]
    val_images = unique_images[n_train:n_train + n_val]
    test_images = unique_images[n_train + n_val:]
    
    # Create splits
    train_df = df[df['image'].isin(train_images)]
    val_df = df[df['image'].isin(val_images)]
    test_df = df[df['image'].isin(test_images)]
    
    print(f"\n" + "="*50)
    print("DATASET SPLITS")
    print("="*50)
    print(f"Train: {len(train_images)} images, {len(train_df)} captions")
    print(f"Val:   {len(val_images)} images, {len(val_df)} captions")
    print(f"Test:  {len(test_images)} images, {len(test_df)} captions")
    
    # Save splits
    output_dir = Path('data/flickr8k/processed')
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    
    print(f"\n✓ Saved splits to {output_dir}/")
    
    return train_df, val_df, test_df

def main():
    """Main preprocessing pipeline"""
    
    captions_file = Path('data/flickr8k/captions.txt')
    
    if not captions_file.exists():
        print(f"✗ Captions file not found: {captions_file}")
        print("Please download the dataset first: python download_dataset.py")
        return
    
    # Build vocabulary
    vocab, df = build_vocabulary(captions_file, min_freq=5)
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(df)
    
    print("\n" + "="*50)
    print("✓ Preprocessing complete!")
    print("="*50)
    print("\nNext steps:")
    print("1. Train: python train.py --epochs 40")
    print("2. Evaluate: python evaluate.py --checkpoint checkpoints/cnn_lstm_best.pth")

if __name__ == '__main__':
    main()
