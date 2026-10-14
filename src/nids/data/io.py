"""Loading of processed splits and metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nids.config import Config
from nids.utils import load_json


def data_dir(cfg: Config) -> Path:
    return Path(cfg.paths.artifacts) / "data"


def load_meta(cfg: Config) -> dict:
    path = data_dir(cfg) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `nids prepare` first")
    return load_json(path)


def load_split(cfg: Config, name: str) -> tuple[np.ndarray, np.ndarray]:
    path = data_dir(cfg) / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `nids prepare` first")
    with np.load(path) as blob:
        return blob["X"].astype(np.float32), blob["y"].astype(np.int64)
