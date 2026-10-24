"""
Plot randomly chosen test images with the generated caption and one human reference.

Uses the predictions CSV written by evaluate.py, so the captions shown are exactly the
ones that were scored. Images are sampled with a fixed seed, not hand-picked.

    python show_samples.py --predictions results/predictions_test_beam3.csv
"""
import argparse
import random
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402


def main(args):
    df = pd.read_csv(args.predictions)
    rows = df.iloc[sorted(random.Random(args.seed).sample(range(len(df)), args.count))]

    cols = 3
    n_rows = (len(rows) + cols - 1) // cols
    fig, axes = plt.subplots(n_rows, cols, figsize=(4.6 * cols, 4.9 * n_rows))
    for ax, (_, row) in zip(axes.ravel(), rows.iterrows()):
        image = Image.open(Path(args.images_dir) / row["image"]).convert("RGB")
        image.thumbnail((400, 400))
        ax.imshow(image)
        reference = row["references"].split(" | ")[0]
        ax.set_title(
            "\n".join(textwrap.wrap(f"model: {row['generated']}", 42))
            + "\n" + "\n".join(textwrap.wrap(f"human: {reference}", 42)),
            fontsize=9, loc="left",
        )
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output, dpi=100, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot random test predictions")
    parser.add_argument("--predictions", default="results/predictions_test_beam3.csv")
    parser.add_argument("--images_dir", default="data/flickr8k/Images")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/samples.png")
    main(parser.parse_args())
