"""
Training script for image captioning models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from pathlib import Path
import sys
from torch.nn.utils.rnn import pack_padded_sequence


# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.dataset import get_dataloaders
from models.cnn_lstm import CNNLSTMModel

def captioning_loss(outputs, captions, lengths, criterion):
    """
    Computes cross-entropy only over the valid (non-padded) time steps.

    captions: (B, T_cap) including <START> ... <END> ... <PAD>
    outputs:  either (B, T_out, V) or already packed as (N, V)
    lengths:  list/1D tensor of original caption lengths (including <START>/<END>)
    """
    # Targets are next-token labels (drop the <START>)
    targets = captions[:, 1:]  # (B, T_cap-1)

    # Convert lengths to a CPU LongTensor for packing
    lengths = torch.as_tensor(lengths, dtype=torch.long)

    # We predict next tokens, so valid target length is (length-1)
    tgt_lengths = lengths - 1

    if outputs.dim() == 3:
        # outputs: (B, T_out, V). Align time dimension with targets
        T_out = outputs.size(1)
        T_tgt = targets.size(1)
        T = min(T_out, T_tgt)

        outputs = outputs[:, :T, :]
        targets = targets[:, :T]

        # Clamp lengths so they don't exceed T and aren't <= 0
        tgt_lengths = torch.clamp(tgt_lengths, min=1, max=T)

        # Pack both using the same lengths
        packed_outputs = pack_padded_sequence(
            outputs, tgt_lengths.cpu(), batch_first=True, enforce_sorted=False
        ).data  # (N, V)

        packed_targets = pack_padded_sequence(
            targets, tgt_lengths.cpu(), batch_first=True, enforce_sorted=False
        ).data  # (N,)

        return criterion(packed_outputs, packed_targets)

    elif outputs.dim() == 2:
        # outputs already looks like packed: (N, V)
        T_tgt = targets.size(1)
        tgt_lengths = torch.clamp(tgt_lengths, min=1, max=T_tgt)

        packed_targets = pack_padded_sequence(
            targets, tgt_lengths.cpu(), batch_first=True, enforce_sorted=False
        ).data  # (N,)

        if outputs.size(0) != packed_targets.size(0):
            raise ValueError(
                f"Packed size mismatch: outputs N={outputs.size(0)} vs targets N={packed_targets.size(0)}"
            )

        return criterion(outputs, packed_targets)

    else:
        raise ValueError(f"Unexpected outputs.dim()={outputs.dim()}")


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for images, captions, lengths in progress_bar:
        images = images.to(device)
        captions = captions.to(device)

        outputs = model(images, captions, lengths)

        # IMPORTANT: loss is computed on packed tokens only
        loss = captioning_loss(outputs, captions, lengths, criterion)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, captions, lengths in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions, lengths)
            loss = captioning_loss(outputs, captions, lengths, criterion)

            total_loss += loss.item()

    return total_loss / len(val_loader)

def select_device():
    """NVIDIA GPU (CUDA) if available, then Apple Silicon (MPS), otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Metal"
    return torch.device("cpu"), "cpu"


def main(args):
    """Main training function"""
    
    # Setup device
    device, dev_name = select_device()
    print(f"Using device: {device} ({dev_name})")
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        data_dir=args.data_dir,
        vocab_path=args.vocab_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create model
    print(f"\nCreating {args.model} model...")
    if args.model == 'cnn_lstm':
        model = CNNLSTMModel(
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            vocab_size=len(vocab),
            num_layers=args.num_layers,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    print("Model device check:", next(model.parameters()).device)

    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    
    # Tensorboard
    writer = SummaryWriter(log_dir=f'runs/{args.model}_{args.experiment_name}')
    
    # Training loop
    best_val_loss = float('inf')
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*70)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        print("="*70)
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f'{args.model}_epoch_{epoch}.pth'
            model.save_checkpoint(checkpoint_path, vocab, optimizer, epoch, val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = checkpoint_dir / f'{args.model}_best.pth'
            model.save_checkpoint(best_checkpoint_path, vocab, optimizer, epoch, val_loss)
            print(f"✓ New best model saved! Val Loss: {val_loss:.4f}")
    
    writer.close()
    print("\n" + "="*70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_checkpoint_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train image captioning model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/flickr8k',
                       help='Path to dataset directory')
    parser.add_argument('--vocab_path', type=str, default='data/flickr8k/processed/vocabulary.pkl',
                       help='Path to vocabulary file')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='cnn_lstm',
                       choices=['cnn_lstm'],
                       help='Model architecture')
    parser.add_argument('--embed_size', type=int, default=512,
                       help='Embedding dimension')
    parser.add_argument('--hidden_size', type=int, default=512,
                       help='Hidden size for LSTM/Transformer')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--experiment_name', type=str, default='exp1',
                       help='Experiment name for logging')
    
    args = parser.parse_args()
    
    main(args)
