"""Produce reliability diagram plots for calibration analysis.

Uses matplotlib to save a simple confidence vs accuracy plot and a bar
showing per-bin support. Deterministic and suitable for reports.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict


def plot_reliability(bins: Dict[str, dict], out_path: str):
    # bins: dict keyed by bin range with 'accuracy' and 'avg_confidence' keys
    labels = []
    accuracies = []
    confs = []
    supports = []
    for k, v in bins.items():
        labels.append(k)
        supports.append(v['support'])
        accuracies.append(v['accuracy'] if v['accuracy'] is not None else 0.0)
        confs.append(v['avg_confidence'] if v['avg_confidence'] is not None else 0.0)

    x = range(len(labels))
    plt.figure(figsize=(8, 6))
    plt.plot(x, confs, label='Avg Confidence', marker='o')
    plt.plot(x, accuracies, label='Accuracy', marker='s')
    plt.xticks(x, labels, rotation=45, fontsize=8)
    plt.ylim(0.0, 1.0)
    plt.xlabel('Confidence Bin')
    plt.ylabel('Value')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
