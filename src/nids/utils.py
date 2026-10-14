"""Small shared helpers: seeding, device selection, JSON IO, logging."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        get_logger("nids").warning("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(preference)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:  # noqa: D102
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, cls=_NumpyEncoder)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def class_weights(counts: np.ndarray, mode: str) -> np.ndarray:
    """Per-class loss weights, normalised to mean 1 over classes that have samples."""
    counts = counts.astype(np.float64)
    if mode == "none":
        return np.ones_like(counts, dtype=np.float32)
    if mode == "sqrt_inv":
        w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    else:
        raise ValueError(f"Unknown class weighting mode: {mode}")
    w = w / w[counts > 0].mean()
    return w.astype(np.float32)
