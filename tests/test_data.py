import numpy as np
import pandas as pd
import pytest

from nids.data.preprocess import FlowPreprocessor, MissingFeaturesError, signed_log1p
from nids.data.schema import CICIDS_FEATURES, map_label, normalise_label, order_classes
from nids.data.synthetic import generate


def test_schema_has_78_features():
    assert len(CICIDS_FEATURES) == 78
    assert len(set(CICIDS_FEATURES)) == 78


@pytest.mark.parametrize(
    "raw,family",
    [
        ("BENIGN", "Benign"), ("DoS Hulk", "DoS"), ("DoS slowloris", "DoS"), ("DDoS", "DDoS"),
        ("FTP-Patator", "BruteForce"), ("SSH-Patator", "BruteForce"), ("PortScan", "PortScan"),
        ("Web Attack \x96 XSS", "WebAttack"), ("Web Attack � Brute Force", "WebAttack"),
        ("Web Attack – Sql Injection", "WebAttack"), ("Infiltration", "Infiltration"), ("Heartbleed", "Heartbleed"),
    ],
)
def test_label_mapping_survives_mangled_encodings(raw, family):
    assert map_label(raw, "family") == family


def test_raw_grouping_keeps_variants_apart():
    assert map_label("DoS Hulk", "raw") != map_label("DoS GoldenEye", "raw")
    assert normalise_label("Web Attack \x96 XSS") == "web attack xss"


def test_class_order_puts_benign_first():
    assert order_classes(["Bot", "Benign", "DoS", "Zeta"])[0] == "Benign"
    assert order_classes(["DoS HULK", "Benign"])[0] == "Benign"


def test_signed_log1p_is_odd_and_monotonic():
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    y = signed_log1p(x)
    assert np.allclose(y, -y[::-1])
    assert np.all(np.diff(y) > 0)


def test_preprocessor_roundtrip_and_robustness(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.lognormal(3, 1, (500, 5)), columns=list("abcde"))
    prep = FlowPreprocessor.fit(df)
    X = prep.transform(df)
    assert X.dtype == np.float32 and X.min() >= -1e-6 and X.max() <= 1 + 1e-6

    prep.save(tmp_path / "p.json")
    again = FlowPreprocessor.load(tmp_path / "p.json")
    assert np.allclose(again.transform(df), X)

    dirty = df.head(3).copy()
    dirty.iloc[0, 0] = np.nan
    dirty.iloc[1, 1] = np.inf
    out = again.transform(dirty)
    assert np.isfinite(out).all()
    assert out[1, 1] > 1.0, "an infinite value must read as far out-of-range (anomalous), not be clipped"

    with pytest.raises(MissingFeaturesError):
        again.transform(df.drop(columns=["a"]))


def test_nonfinite_seen_in_train_gets_indicator_and_stays_in_range(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.lognormal(3, 1, (500, 3)), columns=["dur", "bytes_s", "pkts"])
    df.loc[:19, "bytes_s"] = np.inf  # zero-duration flows: x / 0
    df.loc[20:29, "bytes_s"] = np.nan  # 0 / 0
    prep = FlowPreprocessor.fit(df)
    assert prep.flag_columns == ["bytes_s"]
    assert prep.output_names == ["dur", "bytes_s", "pkts", "bytes_s [non-finite]"]

    X = prep.transform(df)
    assert X.shape == (500, 4) and np.isfinite(X).all()
    assert X.min() >= -1e-6 and X.max() <= 1 + 1e-6, "train-seen non-finite values must be imputed in range"
    assert X[:30, 3].tolist() == [1.0] * 30 and X[30:, 3].sum() == 0
    assert np.isclose(X[:20, 1], 1.0).all()  # +inf -> train finite max

    prep.save(tmp_path / "p.json")
    again = FlowPreprocessor.load(tmp_path / "p.json")
    serve = pd.DataFrame({"dur": [5.0], "bytes_s": [np.inf], "pkts": [np.inf]})
    out = again.transform(serve)
    assert out[0, 3] == 1.0 and out[0, 1] <= 1 + 1e-6
    assert out[0, 2] > 1.0, "non-finite in a column that was always finite in train is genuinely novel"


def test_legacy_preprocessor_json_still_loads(tmp_path):
    import json

    path = tmp_path / "old.json"
    path.write_text(json.dumps({"feature_names": ["a"], "fill": [1.0], "lo": [0.0], "scale": [5.0]}))
    prep = FlowPreprocessor.load(path)
    assert prep.flag_columns == [] and prep.transform(pd.DataFrame({"a": [np.inf]}))[0, 0] > 1.0


def test_clean_keeps_nonfinite_rows():
    from nids.config import load_config
    from nids.data.preprocess import clean
    from nids.data.schema import LABEL_RAW

    df = generate(3000, seed=1).rename(columns={"Label": LABEL_RAW})
    n_nonfinite = int((~np.isfinite(df.drop(columns=[LABEL_RAW]).to_numpy(dtype=float))).any(axis=1).sum())
    out, stats = clean(df, load_config("configs/smoke.yaml"))
    assert n_nonfinite > 0 and stats["rows_with_nonfinite"] > 0
    assert (~np.isfinite(out.drop(columns=[LABEL_RAW]).to_numpy(dtype=float))).any(axis=1).sum() > 0


def test_preprocessor_tolerates_header_whitespace():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    prep = FlowPreprocessor.fit(df)
    spaced = df.rename(columns={"a": " a", "b": " b"})
    assert np.allclose(prep.transform(spaced), prep.transform(df))


def test_synthetic_generator_mimics_cicids_blemishes():
    df = generate(3000, seed=1)
    assert list(df.columns) == CICIDS_FEATURES + ["Label"]
    assert np.isinf(df["Flow Bytes/s"]).any() and df["Flow Bytes/s"].isna().any()
    assert df.duplicated().any()
    assert (df["Fwd Header Length"] == df["Fwd Header Length.1"]).all()


def test_prepare_has_no_leakage(trained):
    from nids.data.io import load_meta, load_split

    meta = load_meta(trained)
    tr, va, te = (load_split(trained, s) for s in ("train", "val", "test"))
    # duplicates were removed before splitting -> no identical rows across splits
    train_rows = {r.tobytes() for r in tr[0]}
    assert not any(r.tobytes() in train_rows for r in te[0])
    assert not any(r.tobytes() in train_rows for r in va[0])
    # val/test keep natural prevalence: benign share is similar in val and test, and train is the only capped split
    assert abs((va[1] == 0).mean() - (te[1] == 0).mean()) < 0.03
    assert meta["classes"][0] == "Benign"
    # scaler was fitted on train: train features lie inside [0, 1]
    assert tr[0].min() >= -1e-6 and tr[0].max() <= 1 + 1e-6
    # non-finite rows are kept and carried by indicator features (no train/serve skew)
    assert meta["cleaning"]["rows_with_nonfinite"] > 0 and meta["nonfinite_flag_columns"]
    assert tr[0].shape[1] == meta["n_features"] == len(meta["feature_names"]) + len(meta["nonfinite_flag_columns"])
