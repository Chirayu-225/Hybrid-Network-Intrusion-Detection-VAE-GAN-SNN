"""CICIDS2017 loading, cleaning, leak-free splitting and scaling.

Design decisions (each one is a classic way to inflate IDS results if done the other way round):

* Exact duplicate flows are removed *before* splitting, otherwise the same flow lands in train and test.
* Splits are stratified and taken at the dataset's natural prevalence. Only the TRAIN split is capped /
  down-sampled, so validation and test false-positive rates and alert volumes are realistic.
* The scaler, the constant-column filter and the NaN fill values are fitted on the (capped) train split only.
* Features are signed-log1p transformed (flow features span ~10 orders of magnitude, with -1 sentinels) and
  then min-max scaled to [0, 1] using train statistics. Out-of-range values at test/serve time are NOT
  clipped here: for the VAE an out-of-range value is precisely the anomaly signal. The GAN and the SNN
  clip to [0, 1] where their math needs it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from nids.config import Config, artifact_dir
from nids.data.schema import (
    BENIGN,
    DUPLICATE_COLUMNS,
    LABEL_RAW,
    map_label,
    order_classes,
)
from nids.utils import get_logger, save_json, set_seed

log = get_logger("nids.preprocess")

_BIG = 1e18  # stand-in for +/-inf at inference time; log-scaled it is still finite but far out of range


class MissingFeaturesError(ValueError):
    """Raised when an input frame lacks columns the preprocessor was fitted on."""


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class FlowPreprocessor:
    """Raw CICFlowMeter features -> float32 matrix in (approximately) [0, 1]. Pure numpy, JSON-serialisable."""

    def __init__(self, feature_names: list[str], fill: np.ndarray, lo: np.ndarray, scale: np.ndarray):
        self.feature_names = list(feature_names)
        self.fill = np.asarray(fill, dtype=np.float64)
        self.lo = np.asarray(lo, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, df: pd.DataFrame) -> FlowPreprocessor:
        names = list(df.columns)
        raw = df.to_numpy(dtype=np.float64)
        fill = np.nanmedian(raw, axis=0)
        logged = signed_log1p(raw)
        lo = logged.min(axis=0)
        hi = logged.max(axis=0)
        scale = np.where(hi - lo < 1e-12, 1.0, hi - lo)
        return cls(names, fill, lo, scale)

    # ------------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        frame = df.copy()
        frame.columns = [str(c).strip() for c in frame.columns]
        missing = [c for c in self.feature_names if c not in frame.columns]
        if missing:
            raise MissingFeaturesError(f"{len(missing)} required feature(s) missing, e.g. {missing[:5]}")
        raw = frame[self.feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        raw = np.where(np.isnan(raw), self.fill, raw)
        raw = np.clip(raw, -_BIG, _BIG)  # +/-inf -> huge finite
        return ((signed_log1p(raw) - self.lo) / self.scale).astype(np.float32)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        save_json(
            {
                "feature_names": self.feature_names,
                "fill": self.fill.tolist(),
                "lo": self.lo.tolist(),
                "scale": self.scale.tolist(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> FlowPreprocessor:
        import json

        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        return cls(blob["feature_names"], blob["fill"], blob["lo"], blob["scale"])


# ---------------------------------------------------------------------- loading / cleaning
def load_raw(raw_dir: str | Path, label_col: str, drop_columns: list[str]) -> pd.DataFrame:
    """Read every CSV in `raw_dir` into one float32 frame plus a `label_raw` string column."""
    files = sorted(Path(raw_dir).rglob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found under '{raw_dir}'. Download CICIDS2017 (see README, 'Real data (CICIDS2017)') "
            "or run `nids synth-data` to create a synthetic stand-in for smoke tests."
        )
    frames: list[pd.DataFrame] = []
    for path in files:
        part = pd.read_csv(path, encoding="latin-1", low_memory=False)
        part.columns = [str(c).strip() for c in part.columns]
        if label_col not in part.columns:
            log.warning("Skipping %s: no '%s' column", path.name, label_col)
            continue
        labels = part[label_col].astype(str).str.strip().to_numpy()
        feats = part.drop(columns=[c for c in [label_col, *drop_columns] if c in part.columns])
        feats = feats.apply(pd.to_numeric, errors="coerce").astype("float32")
        feats[LABEL_RAW] = labels
        frames.append(feats)
        log.info("Loaded %-55s %8d rows", path.name, len(feats))
    cols = [set(f.columns) for f in frames]
    if any(c != cols[0] for c in cols):
        common = sorted(set.intersection(*cols))
        log.warning("CSV files have differing columns; keeping the %d common ones", len(common))
        frames = [f[common] for f in frames]
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop duplicate/identifier columns, non-finite rows and (optionally) exact duplicate flows."""
    stats = {"rows_raw": len(df)}
    drop = list(DUPLICATE_COLUMNS)
    if cfg.data.drop_destination_port:
        drop.append("Destination Port")
    df = df.drop(columns=[c for c in drop if c in df.columns])
    feature_cols = [c for c in df.columns if c != LABEL_RAW]
    finite = np.isfinite(df[feature_cols].to_numpy()).all(axis=1)
    df = df[finite]
    stats["rows_finite"] = len(df)
    if cfg.data.drop_duplicates:
        df = df.drop_duplicates()
    stats["rows_deduplicated"] = len(df)
    return df.reset_index(drop=True), stats


# ---------------------------------------------------------------------- main entry point
def _stratified_splits(y: np.ndarray, val_size: float, test_size: float, seed: int):
    idx = np.arange(len(y))
    train_idx, rest_idx = train_test_split(idx, test_size=val_size + test_size, stratify=y, random_state=seed)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=test_size / (val_size + test_size), stratify=y[rest_idx], random_state=seed
    )
    return train_idx, val_idx, test_idx


def _cap_train(idx: np.ndarray, y: np.ndarray, benign_id: int, cap_benign: int, cap_attack: int, rng) -> np.ndarray:
    keep = []
    for c in np.unique(y[idx]):
        members = idx[y[idx] == c]
        cap = cap_benign if c == benign_id else cap_attack
        if len(members) > cap:
            members = rng.choice(members, size=cap, replace=False)
        keep.append(members)
    return np.sort(np.concatenate(keep))


def _class_counts(y: np.ndarray, n_classes: int) -> list[int]:
    return np.bincount(y, minlength=n_classes).astype(int).tolist()


def _demo_sample(raw_test: pd.DataFrame, names: np.ndarray, size: int, rng) -> pd.DataFrame:
    """Class-balanced-ish sample for the dashboard: half benign, half spread over attack classes."""
    benign_idx = np.flatnonzero(names == BENIGN)
    attack_classes = [c for c in np.unique(names) if c != BENIGN]
    picks = [rng.choice(benign_idx, size=min(len(benign_idx), size // 2), replace=False)]
    per_class = max(1, (size // 2) // max(1, len(attack_classes)))
    for c in attack_classes:
        members = np.flatnonzero(names == c)
        picks.append(rng.choice(members, size=min(len(members), per_class), replace=False))
    sel = rng.permutation(np.concatenate(picks))
    demo = raw_test.iloc[sel].copy()
    demo["Label"] = names[sel]
    return demo.reset_index(drop=True)


def prepare(cfg: Config) -> dict[str, Any]:
    """Run the full preprocessing pipeline and write processed arrays + preprocessor + metadata."""
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    out_dir = artifact_dir(cfg, "data")

    df = load_raw(cfg.paths.raw_dir, cfg.data.label_column, list(cfg.data.drop_columns))
    df, clean_stats = clean(df, cfg)
    log.info("Cleaning: %s", clean_stats)

    # ---- labels
    raw_unique = df[LABEL_RAW].unique()
    mapping = {r: map_label(r, cfg.data.label_grouping) for r in raw_unique}
    names = df[LABEL_RAW].map(mapping).to_numpy()
    uniq, counts = np.unique(names, return_counts=True)
    dropped = {str(u): int(c) for u, c in zip(uniq, counts) if c < cfg.data.min_class_samples}
    if dropped:
        log.warning("Dropping classes with < %d flows: %s", cfg.data.min_class_samples, dropped)
        keep_rows = ~np.isin(names, list(dropped))
        df, names = df[keep_rows].reset_index(drop=True), names[keep_rows]
    classes = order_classes([str(u) for u in np.unique(names)])
    if classes[0] != BENIGN:
        raise ValueError("No benign traffic found; cannot train the anomaly detector")
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_id[n] for n in names], dtype=np.int64)
    feature_cols = [c for c in df.columns if c != LABEL_RAW]
    full_counts = _class_counts(y, len(classes))
    log.info("Class counts (natural): %s", dict(zip(classes, full_counts)))

    # ---- split at natural prevalence, then cap TRAIN only
    train_idx, val_idx, test_idx = _stratified_splits(y, cfg.data.val_size, cfg.data.test_size, cfg.seed)
    train_idx = _cap_train(train_idx, y, 0, cfg.data.max_train_benign, cfg.data.max_train_per_attack_class, rng)

    # ---- fit preprocessing on train only
    train_feats = df.iloc[train_idx][feature_cols]
    varying = [c for c in feature_cols if train_feats[c].nunique() > 1]
    constant = sorted(set(feature_cols) - set(varying))
    if constant:
        log.info("Removing %d constant column(s): %s", len(constant), constant)
    prep = FlowPreprocessor.fit(train_feats[varying])
    prep.save(out_dir.parent / "preprocessor.json")

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        X = prep.transform(df.iloc[idx][varying])
        arrays[split] = (X, y[idx])
        np.savez_compressed(out_dir / f"{split}.npz", X=X, y=y[idx])

    # ---- metadata
    train_counts = _class_counts(y[train_idx], len(classes))
    rare = [
        classes[i]
        for i in range(1, len(classes))
        if train_counts[i] < cfg.data.rare_class_max_train
    ]
    meta = {
        "classes": classes,
        "benign_index": 0,
        "feature_names": varying,
        "n_features": len(varying),
        "label_grouping": cfg.data.label_grouping,
        "cleaning": clean_stats,
        "dropped_classes": dropped,
        "removed_constant_columns": constant,
        "counts": {
            "natural": full_counts,
            "train": train_counts,
            "val": _class_counts(y[val_idx], len(classes)),
            "test": _class_counts(y[test_idx], len(classes)),
        },
        "rare_classes": rare,
        "seed": cfg.seed,
    }
    save_json(meta, out_dir / "meta.json")

    # ---- dashboard replay sample (raw feature space, so it exercises the full serving path)
    demo = _demo_sample(df.iloc[test_idx][varying], names[test_idx], cfg.data.demo_sample_size, rng)
    demo.to_csv(out_dir / "demo_flows.csv", index=False)

    _log_table(classes, meta["counts"], rare)
    return meta


def _log_table(classes: list[str], counts: dict[str, list[int]], rare: list[str]) -> None:
    header = f"{'class':<14}{'natural':>10}{'train':>10}{'val':>10}{'test':>10}  rare"
    lines = [header]
    for i, c in enumerate(classes):
        lines.append(
            f"{c:<14}{counts['natural'][i]:>10}{counts['train'][i]:>10}{counts['val'][i]:>10}"
            f"{counts['test'][i]:>10}  {'*' if c in rare else ''}"
        )
    log.info("Split summary\n%s", "\n".join(lines))
