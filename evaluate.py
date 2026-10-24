"""
Evaluate a trained CNN-LSTM checkpoint on the Flickr8k test split.

Follows the standard image-captioning protocol: one generated caption per image,
scored against all 5 human reference captions of that image (corpus-level BLEU-1..4
and METEOR).

    python evaluate.py --checkpoint checkpoints/cnn_lstm_best.pth
"""
import argparse
import json
import sys
import time
from pathlib import Path

import nltk
import pandas as pd
import torch
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from inference import get_transform, select_device  # noqa: E402
from models.cnn_lstm import CNNLSTMModel  # noqa: E402

BLEU_WEIGHTS = {
    "bleu1": (1, 0, 0, 0),
    "bleu2": (0.5, 0.5, 0, 0),
    "bleu3": (1 / 3, 1 / 3, 1 / 3, 0),
    "bleu4": (0.25, 0.25, 0.25, 0.25),
}


def load_references(csv_path):
    """Return {image_name: [reference token lists]} from a processed split CSV."""
    df = pd.read_csv(csv_path)
    return {image: [c.split() for c in group["caption"]] for image, group in df.groupby("image", sort=True)}


@torch.no_grad()
def generate_all(model, vocab, image_names, images_dir, device, beam_size, batch_size=32):
    """Generate one caption per image. The ResNet encoder runs in batches; decoding runs per image."""
    transform = get_transform()
    captions = {}
    for start in tqdm(range(0, len(image_names), batch_size), desc="Captioning"):
        names = image_names[start : start + batch_size]
        batch = torch.stack([transform(Image.open(images_dir / n).convert("RGB")) for n in names]).to(device)
        features = model.encoder(batch)
        for i, name in enumerate(names):
            captions[name] = model.decoder.generate_caption(features[i : i + 1], vocab, max_length=40,
                                                            beam_size=beam_size)
    return captions


def score(references, hypotheses):
    """Corpus BLEU-1..4 and mean METEOR, each hypothesis against all its references."""
    names = sorted(hypotheses)
    refs = [references[n] for n in names]
    hyps = [hypotheses[n].split() for n in names]
    metrics = {key: corpus_bleu(refs, hyps, weights=w) for key, w in BLEU_WEIGHTS.items()}
    metrics["meteor"] = sum(meteor_score(r, h) for r, h in zip(refs, hyps)) / len(hyps)
    return metrics


def main(args):
    for resource in ("wordnet", "omw-1.4"):
        nltk.download(resource, quiet=True)

    device = select_device()
    model, vocab, epoch, loss = CNNLSTMModel.load_from_checkpoint(args.checkpoint, device=device)
    print(f"Loaded checkpoint (epoch {epoch}, val loss {loss:.4f}), vocabulary {len(vocab)} words, device {device}")

    data_dir = Path(args.data_dir)
    references = load_references(data_dir / "processed" / f"{args.split}.csv")
    image_names = sorted(references)[: args.limit]
    print(f"{args.split} split: {len(image_names)} images, {sum(len(references[n]) for n in image_names)} references")

    start = time.perf_counter()
    hypotheses = generate_all(model, vocab, image_names, data_dir / "Images", device, args.beam_size)
    elapsed = time.perf_counter() - start

    metrics = score(references, hypotheses)
    metrics.update({
        "split": args.split,
        "images": len(image_names),
        "beam_size": args.beam_size,
        "checkpoint_epoch": epoch,
        "seconds_per_image": round(elapsed / len(image_names), 3),
        "unique_captions": len(set(hypotheses.values())),
    })

    print("\n" + "=" * 50)
    for key in ("bleu1", "bleu2", "bleu3", "bleu4", "meteor"):
        print(f"{key.upper():>7}: {metrics[key]:.4f}")
    print(f"Unique captions: {metrics['unique_captions']} / {len(image_names)}")
    print("=" * 50)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}_beam{args.beam_size}"
    (out_dir / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({
        "image": image_names,
        "generated": [hypotheses[n] for n in image_names],
        "references": [" | ".join(" ".join(r) for r in references[n]) for n in image_names],
    }).to_csv(out_dir / f"predictions_{tag}.csv", index=False)
    print(f"Saved metrics and predictions to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a captioning checkpoint (BLEU-1..4, METEOR)")
    parser.add_argument("--checkpoint", default="checkpoints/cnn_lstm_best.pth")
    parser.add_argument("--data_dir", default="data/flickr8k")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N images (quick check)")
    parser.add_argument("--output_dir", default="results")
    main(parser.parse_args())
