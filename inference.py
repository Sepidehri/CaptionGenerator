"""
Simple inference script for generating captions
"""
import argparse
from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import transforms

# Ensure project root (folder containing models/, data/, etc.) is on sys.path
sys.path.append(str(Path(__file__).parent))

from models.cnn_lstm import CNNLSTMModel


def select_device():
    """NVIDIA GPU if available, then Apple Silicon GPU, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def generate_caption(model, image_path: Path, vocab, device, beam_size=3, max_length=40):
    """Generate caption for a single image"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = get_transform()(image).to(device)  # (C, H, W); generate_caption adds the batch dim
    caption = model.generate_caption(image_tensor, vocab, max_length=max_length, beam_size=beam_size)
    return caption


def main(args):
    # Setup device
    device = select_device()
    print(f"Using device: {device}\n")

    # Resolve checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    if args.model == "cnn_lstm":
        model, vocab, epoch, loss = CNNLSTMModel.load_from_checkpoint(str(checkpoint_path), device=device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.eval()
    print(f"✓ Loaded {args.model} model")
    print(f"  Epoch: {epoch}, Loss: {loss:.4f}")
    print(f"  Vocabulary size: {len(vocab)}\n")

    # Collect images
    image_arg = Path(args.image)

    if image_arg.is_file():
        images = [image_arg]
    elif image_arg.is_dir():
        images = sorted(list(image_arg.glob("*.jpg")) + list(image_arg.glob("*.png")))

        if args.max_images is not None:
            images = images[: args.max_images]
    else:
        raise ValueError(f"Invalid image path: {image_arg}")

    if len(images) == 0:
        raise ValueError(f"No .jpg/.png images found in: {image_arg}")

    print(f"Processing {len(images)} image(s)...\n")
    print("=" * 70)

    # Generate captions
    with torch.no_grad():
        for image_path in images:
            caption = generate_caption(
                model=model,
                image_path=image_path,
                vocab=vocab,
                device=device,
                beam_size=args.beam_size,
                max_length=args.max_length,
            )
            print(f"\nImage: {image_path.name}")
            print(f"Caption: {caption}")
            print("-" * 70)

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for images")

    parser.add_argument("--image", type=str, required=True,
                        help="Path to image file or directory")
    parser.add_argument("--model", type=str, default="cnn_lstm",
                        choices=["cnn_lstm"],
                        help="Model architecture")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cnn_lstm_best.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--beam_size", type=int, default=3,
                        help="Beam size for caption generation")
    parser.add_argument("--max_length", type=int, default=40,
                        help="Maximum caption length")
    parser.add_argument("--max_images", type=int, default=10,
                        help="If --image is a directory, cap the number of images processed")

    args = parser.parse_args()
    main(args)
