"""Single-flow inference latency benchmark (what an edge gateway would see)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from nids.config import Config, report_dir
from nids.data.io import load_split
from nids.models.snn import SNNClassifier
from nids.models.vae import VAEDetector
from nids.utils import get_logger, save_json

log = get_logger("nids.benchmark")


def benchmark(cfg: Config, snn_tag: str = "snn", n: int = 300, threads: int = 1) -> dict:
    """Measure CPU latency for batch size 1 (SNN alone, and SNN + VAE) using `threads` threads."""
    torch.set_num_threads(threads)
    device = torch.device("cpu")
    root = Path(cfg.paths.artifacts)
    X, _ = load_split(cfg, "test")
    X = X[np.random.default_rng(cfg.seed).choice(len(X), size=min(n, len(X)), replace=False)]
    snn = SNNClassifier.load(root / snn_tag, device)
    vae = VAEDetector.load(root / "vae", device)

    def timeit(fn) -> dict:
        for x in X[:10]:  # warm-up
            fn(x[None, :])
        times = []
        for x in X:
            t0 = time.perf_counter()
            fn(x[None, :])
            times.append((time.perf_counter() - t0) * 1000)
        t = np.array(times)
        return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
                "p99_ms": float(np.percentile(t, 99)), "throughput_flows_per_s": float(1000 / t.mean())}

    result = {
        "device": "cpu", "threads": threads, "n_flows": int(len(X)), "num_steps": snn.model.num_steps,
        "snn_only": timeit(lambda x: snn.predict_proba(x)),
        "snn_plus_vae": timeit(lambda x: (snn.predict_proba(x), vae.score(x))),
        "note": "PyTorch simulation of the SNN on a general-purpose CPU. Real neuromorphic hardware "
                "(Loihi, SpiNNaker, Akida) executes spikes event-driven; this measures the simulation only.",
    }
    out = report_dir(cfg) / snn_tag
    out.mkdir(parents=True, exist_ok=True)
    save_json(result, out / "latency.json")
    log.info("Latency (batch=1, %d thread): SNN p50 %.2f ms | SNN+VAE p50 %.2f ms",
             threads, result["snn_only"]["p50_ms"], result["snn_plus_vae"]["p50_ms"])
    return result
