"""
Grad-CAM for the CNN-LSTM captioner: which image regions does each generated word rely on?

For every word of a greedily decoded caption, the word's score is back-propagated to the
last convolutional block of ResNet50 (layer4, 7x7 feature maps). The feature maps are
weighted by their average gradient and summed, giving a coarse heatmap that is upsampled
and overlaid on the image.

    python gradcam.py --image data/flickr8k/Images/1007129816_e794419615.jpg
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.append(str(Path(__file__).parent))

from inference import get_transform, select_device  # noqa: E402
from models.cnn_lstm import CNNLSTMModel  # noqa: E402


class GradCAM:
    """Records layer4 activations and their gradients through hooks."""

    def __init__(self, model):
        self.activations = None
        self.gradients = None
        target_layer = model.encoder.resnet[-2]  # layer4, output (1, 2048, 7, 7)
        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def heatmap(self, score):
        """Back-propagate one word score and return a normalised 7x7 heatmap."""
        self.gradients = None
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, 2048, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1))[0]  # (7, 7)
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


def explain(model, vocab, image_path, device, max_words=12):
    """Greedy-decode a caption and return [(word, heatmap)] plus the resized image."""
    image = Image.open(image_path).convert("RGB")
    tensor = get_transform()(image).unsqueeze(0).to(device).requires_grad_(True)

    gradcam = GradCAM(model)
    features = model.encoder(tensor, track_grad=True)

    end_idx = vocab.word2idx["<END>"]
    inputs, states, results = features.unsqueeze(1), None, []
    for _ in range(max_words):
        hiddens, states = model.decoder.lstm(inputs, states)
        scores = model.decoder.fc(hiddens.squeeze(1))
        word_idx = int(scores.argmax(dim=1))
        if word_idx == end_idx:
            break
        model.zero_grad()
        results.append((vocab.idx2word[word_idx], gradcam.heatmap(scores[0, word_idx])))
        inputs = model.decoder.embed(torch.tensor([word_idx], device=device)).unsqueeze(1)
    return results, image.resize((224, 224))


def plot(results, image, out_path, title):
    cols = min(4, len(results) + 1)
    rows = int(np.ceil((len(results) + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    axes[0].imshow(image)
    axes[0].set_title("input", fontsize=11)
    for ax, (word, cam) in zip(axes[1:], results):
        cam_up = np.array(Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.BILINEAR)) / 255.0
        ax.imshow(image)
        ax.imshow(cam_up, cmap="jet", alpha=0.45)
        ax.set_title(f'"{word}"', fontsize=11)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(args):
    device = select_device()
    model, vocab, _, _ = CNNLSTMModel.load_from_checkpoint(args.checkpoint, device=device)

    results, image = explain(model, vocab, args.image, device)
    caption = " ".join(word for word, _ in results)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.image).stem}_gradcam.png"
    plot(results, image, out_path, f"Generated (greedy): {caption}")
    print(f"Caption: {caption}\nSaved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM heatmaps per generated word")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/cnn_lstm_best.pth")
    parser.add_argument("--output_dir", default="outputs/gradcam")
    main(parser.parse_args())
