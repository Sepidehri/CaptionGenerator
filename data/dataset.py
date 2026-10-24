"""
PyTorch Dataset for Flickr8k image captioning
"""
import torch
from torch.utils.data import Dataset, DataLoader
from data.vocab import Vocabulary  # needed for pickle.load
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import pickle

class Flickr8kDataset(Dataset):
    """Flickr8k dataset for image captioning"""
    
    def __init__(self, csv_file, images_dir, vocab, transform=None, max_length=40):
        """
        Args:
            csv_file: Path to CSV with image names and captions
            images_dir: Directory with images
            vocab: Vocabulary object
            transform: Optional image transform
            max_length: Maximum caption length
        """
        self.df = pd.read_csv(csv_file)
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image and caption
        row = self.df.iloc[idx]
        image_name = row['image']
        caption = row['caption']
        
        # Load image
        image_path = self.images_dir / image_name
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert caption to token indices
        tokens = caption.split()
        caption_indices = [self.vocab.word2idx['<START>']]
        caption_indices.extend([self.vocab(token) for token in tokens])
        caption_indices.append(self.vocab.word2idx['<END>'])
        
        # Pad or truncate
        if len(caption_indices) < self.max_length:
            caption_indices.extend([self.vocab.word2idx['<PAD>']] * (self.max_length - len(caption_indices)))
        else:
            caption_indices = caption_indices[:self.max_length]
        
        caption_tensor = torch.LongTensor(caption_indices)
        
        return image, caption_tensor, len(tokens) + 2  # +2 for START and END

def get_dataloaders(data_dir, vocab_path, batch_size=64, num_workers=4):
    """Create train, validation, and test dataloaders"""
    
    data_dir = Path(data_dir)
    
    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Image transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = Flickr8kDataset(
        csv_file=data_dir / 'processed/train.csv',
        images_dir=data_dir / 'Images',
        vocab=vocab,
        transform=train_transform
    )
    
    val_dataset = Flickr8kDataset(
        csv_file=data_dir / 'processed/val.csv',
        images_dir=data_dir / 'Images',
        vocab=vocab,
        transform=val_transform
    )
    
    test_dataset = Flickr8kDataset(
        csv_file=data_dir / 'processed/test.csv',
        images_dir=data_dir / 'Images',
        vocab=vocab,
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, vocab
