import argparse
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from .model import CataractModel
from .preprocess import load_image_to_tensor
from .iqa import check_image_quality
from .utils import set_seed

set_seed()


def evaluate(val_csv: str, weights_path: str | None = None, n_mc: int = 15, out_dir: str = "outputs"):
    # Read validation CSV: expected columns image,label (0 or 1)
    samples = []
    with open(val_csv, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            img, label = row[0], row[1]
            samples.append((img, int(label)))

    model = CataractModel(weights_path=weights_path)

    thresholds = np.linspace(0.0, 1.0, 101)
    rejection_rates = []
    accuracies = []

    # Precompute per-sample results
    records = []  # list of dicts: {image,label,iqa_status,iqa_reason,mean_prob,confidence}
    for img_path, label in samples:
        iqa_status, iqa_reason = check_image_quality(img_path)
        if iqa_status == "REJECT":
            records.append({"image": img_path, "label": label, "iqa_status": "REJECT", "iqa_reason": iqa_reason, "mean_prob": None, "confidence": None})
            continue
        # preprocess and run MC
        tensor = load_image_to_tensor(img_path)
        probs = model.predict_proba(tensor, n_mc=n_mc)
        probs = np.array(probs, dtype=np.float64)
        mean_prob = float(probs.mean())
        var = float(probs.var(ddof=0))
        confidence = max(0.0, 1.0 - (var / 0.25))
        records.append({"image": img_path, "label": label, "iqa_status": "PREDICT", "iqa_reason": None, "mean_prob": mean_prob, "confidence": confidence})

    for thr in thresholds:
        rejected = 0
        accepted_correct = 0
        accepted_total = 0
        for r in records:
            if r["iqa_status"] == "REJECT":
                rejected += 1
                continue
            if r["confidence"] is None or r["confidence"] < thr:
                rejected += 1
            else:
                accepted_total += 1
                pred = 1 if r["mean_prob"] >= 0.5 else 0
                if pred == r["label"]:
                    accepted_correct += 1

        rejection_rate = rejected / len(records) if records else 0.0
        accuracy = (accepted_correct / accepted_total) if accepted_total > 0 else 0.0
        rejection_rates.append(rejection_rate)
        accuracies.append(accuracy)

    os.makedirs(out_dir, exist_ok=True)
    # Save plot
    plt.figure(figsize=(6, 4))
    plt.plot(rejection_rates, accuracies, marker=".")
    plt.xlabel("Rejection Rate")
    plt.ylabel("Accuracy on Accepted Samples")
    plt.title("Rejection vs Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rejection_vs_acc.png"))
    plt.close()

    # Save metrics
    np.savez(os.path.join(out_dir, "rejection_metrics.npz"), thresholds=thresholds, rejection_rates=np.array(rejection_rates), accuracies=np.array(accuracies))


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("val_csv", help="CSV file with image,label per line")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--n-mc", type=int, default=15)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()
    evaluate(args.val_csv, weights_path=args.weights, n_mc=args.n_mc, out_dir=args.out_dir)


if __name__ == "__main__":
    _cli()
