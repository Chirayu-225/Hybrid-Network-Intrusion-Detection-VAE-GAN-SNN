"""Shared fixtures. `trained` runs the whole pipeline once on tiny synthetic data (~1 min on CPU)."""

from __future__ import annotations

import copy

import pytest

from nids.config import Config, load_config


@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory) -> Config:
    root = tmp_path_factory.mktemp("nids")
    cfg = load_config("configs/smoke.yaml")
    cfg = Config(copy.deepcopy(cfg.to_dict()))
    cfg["paths"]["raw_dir"] = str(root / "raw")
    cfg["paths"]["artifacts"] = str(root / "artifacts")
    cfg["paths"]["reports"] = str(root / "reports")
    cfg["data"]["synthetic"]["n_rows"] = 12000
    cfg["vae"]["epochs"] = 4
    cfg["gan"]["iterations"] = 150
    cfg["snn"]["epochs"] = 3
    cfg["snn"]["hidden"] = [48, 24]
    cfg["snn"]["num_steps"] = 6
    return cfg


@pytest.fixture(scope="session")
def trained(tiny_cfg) -> Config:
    from nids.calibrate import calibrate
    from nids.data.preprocess import prepare
    from nids.data.synthetic import generate, write_csvs
    from nids.training.train_gan import train_gan
    from nids.training.train_snn import train_snn
    from nids.training.train_vae import train_vae

    cfg = tiny_cfg
    write_csvs(generate(cfg.data.synthetic.n_rows, cfg.data.synthetic.seed), cfg.paths.raw_dir, 2)
    prepare(cfg)
    train_vae(cfg)
    train_gan(cfg)
    train_snn(cfg)
    calibrate(cfg)
    return cfg
