"""
Export the train/validation loss from a TensorBoard run to CSV and plot it.

    python plot_training.py runs/cnn_lstm_exp1
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402


def read_run(run_dir):
    """Return a DataFrame with epoch, train_loss and val_loss from the longest event file in run_dir."""
    best = None
    for path in sorted(Path(run_dir).glob("events.out.tfevents.*")):
        acc = EventAccumulator(str(path))
        acc.Reload()
        if "Loss/val" not in acc.Tags().get("scalars", []):
            continue
        df = pd.DataFrame({
            "epoch": [s.step for s in acc.Scalars("Loss/train")],
            "train_loss": [s.value for s in acc.Scalars("Loss/train")],
            "val_loss": [s.value for s in acc.Scalars("Loss/val")],
        })
        if best is None or len(df) > len(best):
            best = df
    if best is None:
        raise FileNotFoundError(f"No event file with Loss/val in {run_dir}")
    return best


def main(args):
    df = read_run(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "training_log.csv", index=False, float_format="%.4f")

    best = df.loc[df["val_loss"].idxmin()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["train_loss"], label="train loss")
    ax.plot(df["epoch"], df["val_loss"], label="validation loss")
    ax.axvline(best["epoch"], color="gray", linestyle="--", linewidth=1)
    ax.annotate(f"best: epoch {int(best['epoch'])}, val {best['val_loss']:.3f}",
                xy=(best["epoch"], best["val_loss"]), xytext=(best["epoch"] + 3, best["val_loss"] + 0.6),
                arrowprops={"arrowstyle": "->", "color": "gray"}, fontsize=9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("CNN-LSTM training on Flickr8k")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=120)
    print(f"{len(df)} epochs, best validation loss {best['val_loss']:.4f} at epoch {int(best['epoch'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot train/val loss from a TensorBoard run")
    parser.add_argument("run_dir")
    parser.add_argument("--output_dir", default="results")
    main(parser.parse_args())
