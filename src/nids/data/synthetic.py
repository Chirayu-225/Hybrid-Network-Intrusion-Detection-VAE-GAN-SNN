"""Synthetic CICIDS2017-schema generator.

Purpose: let CI, unit tests and `make smoke` exercise the *exact* production code path (CSV parsing,
header whitespace, mangled labels, inf/NaN, duplicates, constant and duplicated columns) without the
2.8M-row dataset. The data is heavy-tailed and class-imbalanced like the real thing, but it is NOT a
benchmark: never report metrics measured on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nids.data.schema import CICIDS_FEATURES, DUPLICATE_COLUMNS, map_label
from nids.utils import get_logger

log = get_logger("nids.synthetic")

# (raw label as it appears in CICIDS2017 CSVs, fraction of rows)
CLASS_MIX: list[tuple[str, float]] = [
    ("BENIGN", 0.704),
    ("DoS Hulk", 0.05),
    ("DoS slowloris", 0.03),
    ("DDoS", 0.07),
    ("PortScan", 0.08),
    ("FTP-Patator", 0.015),
    ("SSH-Patator", 0.015),
    ("Bot", 0.015),
    ("Web Attack \x96 Brute Force", 0.008),
    ("Web Attack \x96 XSS", 0.004),
    ("Infiltration", 0.006),
    ("Heartbleed", 0.003),
]

_FLAG_COLUMNS = [
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Fwd PSH Flags",
]
_CONSTANT_ZERO = [
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]
_SENTINEL_COLUMNS = ["Init_Win_bytes_forward", "Init_Win_bytes_backward"]
# How hard each family is to tell apart from benign: (number of shifted features, shift magnitude in sigmas)
_DIFFICULTY = {
    "DoS": (16, 1.6), "DDoS": (16, 1.5), "PortScan": (14, 1.7), "BruteForce": (14, 1.4),
    "WebAttack": (10, 1.0), "Bot": (10, 1.0), "Infiltration": (9, 0.9), "Heartbleed": (9, 1.0),
}


def generate(n_rows: int, seed: int = 7) -> pd.DataFrame:
    """Return a DataFrame with CICIDS2017 columns (header whitespace included) and a 'Label' column."""
    rng = np.random.default_rng(seed)
    n_feat = len(CICIDS_FEATURES)
    base_cols = [c for c in CICIDS_FEATURES if c not in DUPLICATE_COLUMNS]
    d = len(base_cols)

    mu0 = rng.uniform(1.0, 8.0, d)
    sigma = rng.uniform(0.5, 1.3, d)
    mixing = rng.normal(0.0, 0.5, (d, 6))  # shared latent factors -> realistic feature correlation

    def shifted(k: int, mag: float, base: np.ndarray, local_rng: np.random.Generator) -> np.ndarray:
        idx = local_rng.choice(d, size=k, replace=False)
        out = base.copy()
        out[idx] += local_rng.choice([-1.0, 1.0], size=k) * mag * sigma[idx] * local_rng.uniform(0.7, 1.3, k)
        return out

    # benign has three behavioural modes (browsing / DNS-like / bulk transfer)
    benign_modes = [mu0, shifted(10, 1.0, mu0, rng), shifted(10, 1.2, mu0, rng)]
    benign_weights = np.array([0.6, 0.3, 0.1])

    family_mu: dict[str, np.ndarray] = {}
    for family, (k, mag) in _DIFFICULTY.items():
        family_mu[family] = shifted(k, mag, mu0, rng)

    labels = [lab for lab, _ in CLASS_MIX]
    fracs = np.array([f for _, f in CLASS_MIX], dtype=float)
    fracs /= fracs.sum()
    counts = np.maximum((fracs * n_rows).round().astype(int), 12)

    frames = []
    for lab, n in zip(labels, counts):
        family = map_label(lab, "family")
        if family == "Benign":
            mode = rng.choice(3, size=n, p=benign_weights)
            mu = np.stack([benign_modes[m] for m in mode])
        else:
            mu = np.tile(family_mu[family] + rng.normal(0, 0.15, d), (n, 1))
        z = rng.normal(size=(n, 6))
        log_x = mu + sigma * (z @ mixing.T) + sigma * 0.55 * rng.normal(size=(n, d))
        x = np.exp(np.clip(log_x, -5, 22))
        df = pd.DataFrame(x, columns=base_cols)

        flag_p = {c: (rng.uniform(0.05, 0.6) if family == "Benign" else rng.uniform(0.0, 0.9)) for c in _FLAG_COLUMNS}
        for c in _FLAG_COLUMNS:
            df[c] = (rng.random(n) < flag_p[c]).astype(float)
        for c in _SENTINEL_COLUMNS:
            sentinel = rng.random(n) < (0.12 if family == "Benign" else 0.4)
            df.loc[sentinel, c] = -1.0
        for c in _CONSTANT_ZERO:
            df[c] = 0.0
        df["Label"] = lab
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    # Real-world blemishes the loader must cope with
    for col in ("Flow Bytes/s", "Flow Packets/s"):
        inf_rows = rng.random(len(data)) < 0.003
        nan_rows = rng.random(len(data)) < 0.001
        data.loc[inf_rows, col] = np.inf
        data.loc[nan_rows, col] = np.nan
    dup = data.sample(frac=0.02, random_state=seed)
    data = pd.concat([data, dup], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # duplicate CICFlowMeter column + original column order
    data["Fwd Header Length.1"] = data["Fwd Header Length"]
    ordered = [c for c in CICIDS_FEATURES if c in data.columns] + ["Label"]
    assert len(ordered) == n_feat + 1
    return data[ordered]


def write_csvs(df: pd.DataFrame, out_dir: str | Path, n_files: int = 2) -> list[Path]:
    """Write like the original dataset: leading space in headers, latin-1 encoding, one file per 'day'."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spaced = df.copy()
    spaced.columns = [f" {c}" for c in df.columns]
    paths = []
    bounds = np.linspace(0, len(spaced), max(1, n_files) + 1).astype(int)
    for i in range(len(bounds) - 1):
        path = out / f"Synthetic-Day{i + 1}.pcap_ISCX.csv"
        spaced.iloc[bounds[i]:bounds[i + 1]].to_csv(path, index=False, encoding="latin-1")
        paths.append(path)
    log.info("Wrote %d synthetic CSVs (%d rows) to %s", len(paths), len(df), out)
    return paths
