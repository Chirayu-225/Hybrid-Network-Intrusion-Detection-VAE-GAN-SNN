"""End-to-end tests on the tiny synthetic pipeline: artifacts, inference, evaluation, API, CLI."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nids.inference import HybridNIDS


@pytest.fixture(scope="module")
def pipe(trained):
    return HybridNIDS.load(trained)


def test_expected_artifacts_exist(trained):
    root = Path(trained.paths.artifacts)
    for rel in ["preprocessor.json", "data/meta.json", "data/demo_flows.csv", "vae/vae.pt", "vae/scorer.json",
                "gan/synthetic.npz", "gan/fidelity.json", "snn/snn.pt", "snn/calibration.json", "snn/thresholds.json"]:
        assert (root / rel).exists(), rel


def test_gan_only_touches_train_and_respects_caps(trained):
    root = Path(trained.paths.artifacts)
    meta = json.loads((root / "data/meta.json").read_text())
    syn = np.load(root / "gan/synthetic.npz")
    counts = meta["counts"]["train"]
    for cid in np.unique(syn["y"]):
        n_syn = int((syn["y"] == cid).sum())
        assert n_syn <= trained.gan.max_synth_per_real * counts[cid]
        assert counts[cid] >= trained.gan.min_real_samples
    assert syn["X"].min() >= 0 and syn["X"].max() <= 1


def test_predict_frame_on_raw_flows(trained, pipe):
    demo = pd.read_csv(Path(trained.paths.artifacts) / "data/demo_flows.csv")
    feats = demo.drop(columns=["Label"])
    out = pipe.predict_frame(feats)
    n = len(demo)
    assert out["risk"].shape == (n,) and out["probs"].shape == (n, len(pipe.classes))
    assert np.allclose(out["probs"].sum(1), 1, atol=1e-5)
    assert set(out["action"]) <= {"allow", "monitor", "block"}
    assert ((out["risk"] >= 0) & (out["risk"] <= 1)).all()
    # deterministic: same flows -> same verdicts (rate-coding RNG is seeded)
    again = pipe.predict_frame(feats)
    assert np.array_equal(out["action"], again["action"]) and np.allclose(out["risk"], again["risk"])


def test_attacks_score_higher_than_benign(trained, pipe):
    demo = pd.read_csv(Path(trained.paths.artifacts) / "data/demo_flows.csv")
    out = pipe.predict_frame(demo.drop(columns=["Label"]))
    attack = demo["Label"].to_numpy() != "Benign"
    assert out["risk"][attack].mean() > out["risk"][~attack].mean()


def test_evaluate_writes_metrics_and_figures(trained):
    from nids.evaluation import evaluate

    metrics = evaluate(trained)
    out = Path(trained.paths.reports) / "snn"
    for name in ["metrics.json", "summary.md", "confusion_matrix.png", "vae_roc.png", "risk_distribution.png",
                 "per_class_f1.png", "action_by_class.png"]:
        assert (out / name).exists(), name
    assert 0 <= metrics["classification"]["macro_f1"] <= 1
    assert metrics["vae"]["auroc"] > 0.5
    assert metrics["fused"]["risk_auroc"] > 0.5
    assert "snn_only (predicted != Benign)" in metrics["fused"]["comparators"]


def test_baseline_tag_trains_without_synthetic_data(trained):
    from nids.calibrate import calibrate
    from nids.training.train_snn import train_snn

    m = train_snn(trained, augment=False, tag="snn_baseline")
    assert m["n_train_synthetic"] == 0
    calibrate(trained, "snn_baseline")
    assert (Path(trained.paths.artifacts) / "snn_baseline" / "thresholds.json").exists()


# ------------------------------------------------------------------ API
@pytest.fixture(scope="module")
def client(pipe):
    from fastapi.testclient import TestClient

    from nids.serving.api import create_app

    with TestClient(create_app(pipe)) as c:
        yield c


def test_api_health_and_info(client, pipe):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True, "detail": None}
    info = client.get("/model-info").json()
    assert info["classes"][0] == "Benign" and info["n_features"] == len(pipe.preprocessor.feature_names)


def test_api_predict(client, trained):
    demo = pd.read_csv(Path(trained.paths.artifacts) / "data/demo_flows.csv").drop(columns=["Label"]).head(5)
    resp = client.post("/predict", json={"flows": demo.to_dict(orient="records")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_flows"] == 5 and len(body["results"]) == 5
    first = body["results"][0]
    assert first["action"] in ("allow", "monitor", "block")
    assert {"class", "probability"} <= set(first["top_classes"][0])


def test_api_rejects_bad_input(client):
    assert client.post("/predict", json={"flows": []}).status_code == 422
    resp = client.post("/predict", json={"flows": [{"Flow Duration": 1.0}]})
    assert resp.status_code == 422 and "missing" in resp.json()["detail"]
    assert client.post("/predict", json={"flows": [{"Flow Duration": "abc"}]}).status_code == 422


def test_api_degraded_without_artifacts(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from nids.serving.api import create_app

    cfg_file = tmp_path / "empty.yaml"
    cfg_file.write_text(f"_base_: {Path('configs/default.yaml').resolve()}\npaths:\n  artifacts: {tmp_path / 'nothing'}\n")
    monkeypatch.setenv("NIDS_CONFIG", str(cfg_file))
    with TestClient(create_app(None)) as c:
        health = c.get("/health").json()
        assert health["status"] == "degraded" and not health["model_loaded"]
        assert "Missing artifacts" in health["detail"]
        assert c.post("/predict", json={"flows": [{"a": 1}]}).status_code == 503


def test_cli_parses_all_commands():
    from nids.cli import build_parser

    p = build_parser()
    for cmd in ["synth-data", "prepare", "train-vae", "train-gan", "train-snn --no-augment --tag x", "calibrate",
                "evaluate", "benchmark --threads 2", "run-all --synthetic", "show-config"]:
        p.parse_args(cmd.split())
