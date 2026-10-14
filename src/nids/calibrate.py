"""Calibrate fusion thresholds on the validation split (never on test)."""

from __future__ import annotations

from pathlib import Path

from nids.config import Config
from nids.data.io import load_meta, load_split
from nids.fusion import RiskEngine
from nids.models.snn import SNNClassifier
from nids.models.vae import VAEDetector
from nids.utils import get_device, get_logger

log = get_logger("nids.calibrate")


def calibrate(cfg: Config, snn_tag: str = "snn") -> dict:
    device = get_device(cfg.device)
    root = Path(cfg.paths.artifacts)
    meta = load_meta(cfg)
    X_val, y_val = load_split(cfg, "val")
    snn = SNNClassifier.load(root / snn_tag, device)
    vae = VAEDetector.load(root / "vae", device)

    probs = snn.predict_proba(X_val)
    anomaly = vae.score(X_val)["score"]
    engine = RiskEngine(meta["classes"], cfg.fusion)
    result = engine.calibrate(probs, anomaly, y_val, cfg.fusion)
    engine.save(root / snn_tag / "thresholds.json", extra={"calibration": result["realised_on_validation"],
                                                         "fusion_config": cfg.fusion.to_dict()})
    log.info("Thresholds: %s | realised on validation: %s", result["thresholds"], result["realised_on_validation"])
    return result
