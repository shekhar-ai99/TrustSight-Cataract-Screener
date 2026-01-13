import time
import resource
import argparse
import os
from .utils import set_seed
from .model import CataractModel
from .preprocess import load_image_to_tensor


def _current_mem_mb():
    # ru_maxrss is kilobytes on Linux
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024.0


def benchmark(image_path: str, weights_path: str | None = None, runs: int = 5, n_mc: int | None = None):
    set_seed()
    # Load model once to avoid measuring cold-start repeatedly
    model = CataractModel(weights_path=weights_path)

    # Move to CPU (explicit)
    import torch
    model.to(torch.device("cpu"))
    model.eval()

    # Prepare input tensor
    img_tensor = load_image_to_tensor(image_path)

    # Determine MC samples from env if not provided
    if n_mc is None:
        try:
            n_mc = int(os.environ.get("MC_DROPOUT_SAMPLES", "15"))
        except Exception:
            n_mc = 15

    times = []
    mems = []

    # Warmup
    model.predict_proba(img_tensor, n_mc=1)

    for i in range(runs):
        t0 = time.perf_counter()
        # run MC sampling
        probs = model.predict_proba(img_tensor, n_mc=n_mc)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        mems.append(_current_mem_mb())

    avg_latency = sum(times) / len(times)
    peak_mem = max(mems)

    log_line = f"avg_latency_s={avg_latency:.6f},peak_mem_mb={peak_mem:.2f},runs={runs},n_mc={n_mc}\n"
    with open("benchmark_logs.txt", "a") as f:
        f.write(log_line)

    print("Benchmark results:")
    print(log_line)


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--weights", default=None)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--n-mc", type=int, default=None)
    args = p.parse_args()
    benchmark(args.image, weights_path=args.weights, runs=args.runs, n_mc=args.n_mc)


if __name__ == "__main__":
    _cli()
