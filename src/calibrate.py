import csv
import argparse
import numpy as np


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).

    probs: predicted probabilities (0..1)
    labels: ground-truth labels (0 or 1)
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if not np.any(mask):
            continue
        prop = mask.sum() / probs.size
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += prop * abs(avg_conf - avg_acc)
    return float(ece)


def _load_csv_preds(csv_path: str):
    # Expect CSV with rows: image, label, prob
    labels = []
    probs = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Accept either image,label,prob or label,prob
            if len(row) >= 3:
                _, label, prob = row[0], row[1], row[2]
            elif len(row) == 2:
                label, prob = row[0], row[1]
            else:
                continue
            try:
                labels.append(int(label))
                probs.append(float(prob))
            except Exception:
                continue
    return np.array(probs, dtype=np.float64), np.array(labels, dtype=np.int32)


def evaluate(in_csv: str, cross_csv: str, n_bins: int = 10):
    probs_in, labels_in = _load_csv_preds(in_csv)
    probs_cross, labels_cross = _load_csv_preds(cross_csv)

    ece_in = compute_ece(probs_in, labels_in, n_bins=n_bins)
    ece_cross = compute_ece(probs_cross, labels_cross, n_bins=n_bins)

    print(f"ECE (in-distribution): {ece_in:.6f}")
    print(f"ECE (cross-dataset): {ece_cross:.6f}")

    # Also print simple metrics that may be useful in the robustness report
    def auc_like(probs, labels):
        # A very small AUC proxy using Mann-Whitney U equivalent
        try:
            from scipy.stats import rankdata
        except Exception:
            # If scipy unavailable, fall back to simple separation metric
            pos = probs[labels == 1]
            neg = probs[labels == 0]
            if pos.size == 0 or neg.size == 0:
                return 0.0
            return float(np.mean(pos) - np.mean(neg))
        ranks = rankdata(probs)
        n1 = (labels == 1).sum()
        n0 = (labels == 0).sum()
        if n1 == 0 or n0 == 0:
            return 0.0
        rank_sum = ranks[labels == 1].sum()
        auc = (rank_sum - n1 * (n1 + 1) / 2) / (n0 * n1)
        return float(auc)

    auc_in = auc_like(probs_in, labels_in)
    auc_cross = auc_like(probs_cross, labels_cross)

    print(f"AUC proxy (in-distribution): {auc_in:.6f}")
    print(f"AUC proxy (cross-dataset): {auc_cross:.6f}")


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_csv", help="CSV with image,label,prob for in-distribution validation")
    parser.add_argument("cross_csv", help="CSV with image,label,prob for cross-dataset validation")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    evaluate(args.in_csv, args.cross_csv, n_bins=args.bins)


if __name__ == "__main__":
    _cli()
