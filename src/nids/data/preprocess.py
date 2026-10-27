"""CICIDS2017 loading, cleaning, leak-free splitting and scaling.

Design decisions (each one is a classic way to inflate IDS results if done the other way round):

* Exact duplicate flows are removed *before* splitting, otherwise the same flow lands in train and test.
* Splits are stratified and taken at the dataset's natural prevalence. Only the TRAIN split is capped /
  down-sampled, so validation and test false-positive rates and alert volumes are realistic.
* The scaler, the constant-column filter and the NaN fill values are fitted on the (capped) train split only.
* Rows with inf/NaN are KEPT. In CICIDS2017 they are mostly zero-duration flows (Flow Bytes/s = x/0), a real
  and label-correlated traffic pattern. Dropping them in training while imputing them at serve time is
  train/serve skew: the models never see such flows, so every one of them looks maximally anomalous in
  production. Instead, each column that is non-finite somewhere in TRAIN gets a 0/1 "[non-finite]"
  indicator feature, and its non-finite values are imputed in range (+inf -> train finite max,
  -inf -> train finite min, NaN -> train median). A column that was always finite in train has no
  indicator; a non-finite value there at serve time is genuinely novel and is mapped far out of range,
  where the VAE flags it.
* Features are signed-log1p transformed (flow features span ~10 orders of magnitude, with -1 sentinels) and
  then min-max scaled to [0, 1] using train statistics. Out-of-range values at test/serve time are NOT
  clipped here: for the VAE an out-of-range value is precisely the anomaly signal. The GAN and the SNN
  clip to [0, 1] where their math needs it.
"""

from __future__ import annotations

import warnings
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

_BIG = 1e18  # stand-in for +/-inf in columns never non-finite in train: log-scaled it is finite but far out of range
FLAG_SUFFIX = " [non-finite]"


class MissingFeaturesError(ValueError):
    """Raised when an input frame lacks columns the preprocessor was fitted on."""


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class FlowPreprocessor:
    """Raw CICFlowMeter features -> float32 matrix in (approximately) [0, 1]. Pure numpy, JSON-serialisable.

    Output columns: the scaled raw features (`feature_names`, the API's input contract), followed by one 0/1
    indicator per column in `flag_columns` (see `output_names`).
    """

    def __init__(
        self,
        feature_names: list[str],
        fill: np.ndarray,
        lo: np.ndarray,
        scale: np.ndarray,
        flag_columns: list[str] | None = None,
        pos_fill: np.ndarray | None = None,
        neg_fill: np.ndarray | None = None,
    ):
        self.feature_names = list(feature_names)
        n = len(self.feature_names)
        self.fill = np.asarray(fill, dtype=np.float64)
        self.lo = np.asarray(lo, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        # Defaults reproduce the pre-indicator behaviour, so older preprocessor.json files still load.
        self.flag_columns = list(flag_columns or [])
        self.pos_fill = np.full(n, _BIG) if pos_fill is None else np.asarray(pos_fill, dtype=np.float64)
        self.neg_fill = np.full(n, -_BIG) if neg_fill is None else np.asarray(neg_fill, dtype=np.float64)
        self._flag_idx = [self.feature_names.index(c) for c in self.flag_columns]

    @property
    def output_names(self) -> list[str]:
        return self.feature_names + [c + FLAG_SUFFIX for c in self.flag_columns]

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, df: pd.DataFrame) -> FlowPreprocessor:
        names = list(df.columns)
        raw = df.to_numpy(dtype=np.float64)
        finite = np.isfinite(raw)
        masked = np.where(finite, raw, np.nan)
        with warnings.catch_warnings():  # an all-non-finite column yields NaN stats; replaced below
            warnings.simplefilter("ignore", RuntimeWarning)
            fill = np.nanmedian(masked, axis=0)
            fmax = np.nanmax(masked, axis=0)
            fmin = np.nanmin(masked, axis=0)
        fill, fmax, fmin = (np.nan_to_num(v, nan=0.0) for v in (fill, fmax, fmin))
        flagged = ~finite.all(axis=0)
        flag_columns = [n for n, f in zip(names, flagged) if f]
        prep = cls(names, fill, np.zeros(len(names)), np.ones(len(names)), flag_columns,
                   pos_fill=np.where(flagged, fmax, _BIG), neg_fill=np.where(flagged, fmin, -_BIG))
        logged = signed_log1p(prep._impute(raw)[0])
        prep.lo = logged.min(axis=0)
        hi = logged.max(axis=0)
        prep.scale = np.where(hi - prep.lo < 1e-12, 1.0, hi - prep.lo)
        return prep

    # ------------------------------------------------------------------ transform
    def _impute(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Replace NaN / +inf / -inf column-wise; return the imputed matrix and the non-finite mask."""
        nan, pos, neg = np.isnan(raw), np.isposinf(raw), np.isneginf(raw)
        out = np.where(nan, self.fill, raw)
        out = np.where(pos, self.pos_fill, out)
        out = np.where(neg, self.neg_fill, out)
        return np.clip(out, -_BIG, _BIG), nan | pos | neg

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        frame = df.copy()
        frame.columns = [str(c).strip() for c in frame.columns]
        missing = [c for c in self.feature_names if c not in frame.columns]
        if missing:
            raise MissingFeaturesError(f"{len(missing)} required feature(s) missing, e.g. {missing[:5]}")
        raw = frame[self.feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        imputed, nonfinite = self._impute(raw)
        scaled = (signed_log1p(imputed) - self.lo) / self.scale
        if self._flag_idx:
            scaled = np.hstack([scaled, nonfinite[:, self._flag_idx].astype(np.float64)])
        return scaled.astype(np.float32)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        save_json(
            {
                "feature_names": self.feature_names,
                "fill": self.fill.tolist(),
                "lo": self.lo.tolist(),
                "scale": self.scale.tolist(),
                "flag_columns": self.flag_columns,
                "pos_fill": self.pos_fill.tolist(),
                "neg_fill": self.neg_fill.tolist(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> FlowPreprocessor:
        import json

        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        return cls(blob["feature_names"], blob["fill"], blob["lo"], blob["scale"],
                   blob.get("flag_columns"), blob.get("pos_fill"), blob.get("neg_fill"))


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
    """Drop duplicate/identifier columns and (optionally) exact duplicate flows. Non-finite rows are kept."""
    stats = {"rows_raw": len(df)}
    drop = list(DUPLICATE_COLUMNS)
    if cfg.data.drop_destination_port:
        drop.append("Destination Port")
    df = df.drop(columns=[c for c in drop if c in df.columns])
    feature_cols = [c for c in df.columns if c != LABEL_RAW]
    stats["rows_with_nonfinite"] = int((~np.isfinite(df[feature_cols].to_numpy())).any(axis=1).sum())
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
    if prep.flag_columns:
        log.info("Non-finite values kept with indicator features for %d column(s): %s",
                 len(prep.flag_columns), prep.flag_columns)
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
        "feature_names": varying,  # raw input columns (the serving contract)
        "model_feature_names": prep.output_names,  # columns of the processed arrays
        "n_features": len(prep.output_names),
        "nonfinite_flag_columns": prep.flag_columns,
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
