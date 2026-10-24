"""
Shrink a training checkpoint for distribution.

Training checkpoints contain the Adam optimizer state and float32 weights (~194 MB),
which is above GitHub's 100 MB file limit. For inference only the weights are needed,
and storing them in float16 halves the size again. `load_from_checkpoint` casts them
back to float32; `evaluate.py` gives the same scores with both files (see README).

    python slim_checkpoint.py checkpoints/cnn_lstm_best_full.pth checkpoints/cnn_lstm_best.pth
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))

from data.vocab import Vocabulary  # noqa: E402,F401  (needed to unpickle the vocabulary)


def half(state_dict):
    return {k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()}


def main(args):
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    slim = {key: value for key, value in checkpoint.items() if key != "optimizer_state_dict"}
    slim["encoder_state_dict"] = half(checkpoint["encoder_state_dict"])
    slim["decoder_state_dict"] = half(checkpoint["decoder_state_dict"])
    torch.save(slim, args.target)
    size_mb = Path(args.target).stat().st_size / 2**20
    print(f"Saved {args.target} ({size_mb:.1f} MB, epoch {slim.get('epoch')})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drop optimizer state and store weights in float16")
    parser.add_argument("source")
    parser.add_argument("target")
    main(parser.parse_args())
