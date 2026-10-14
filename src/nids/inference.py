"""End-to-end inference: raw flow features -> risk score, class and action."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from nids.config import Config
from nids.data.preprocess import FlowPreprocessor
from nids.fusion import RiskEngine
from nids.models.snn import SNNClassifier
from nids.models.vae import VAEDetector
from nids.utils import get_device, load_json


class HybridNIDS:
    def __init__(
        self,
        preprocessor: FlowPreprocessor,
        vae: VAEDetector,
        snn: SNNClassifier,
        engine: RiskEngine,
        meta: dict[str, Any],
    ):
        self.preprocessor, self.vae, self.snn, self.engine, self.meta = preprocessor, vae, snn, engine, meta
        self.classes: list[str] = snn.classes

    @classmethod
    def load(cls, cfg: Config, snn_tag: str = "snn", device: torch.device | None = None) -> HybridNIDS:
        root = Path(cfg.paths.artifacts)
        device = device or get_device(cfg.device)
        needed = [root / "preprocessor.json", root / "vae" / "vae.pt", root / snn_tag / "snn.pt",
                  root / snn_tag / "thresholds.json", root / "data" / "meta.json"]
        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing artifacts (run the training pipeline first): {missing}")
        snn = SNNClassifier.load(root / snn_tag, device)
        engine = RiskEngine(snn.classes, cfg.fusion, RiskEngine.load_thresholds(root / snn_tag / "thresholds.json"))
        return cls(
            FlowPreprocessor.load(root / "preprocessor.json"),
            VAEDetector.load(root / "vae", device),
            snn,
            engine,
            load_json(root / "data" / "meta.json"),
        )

    # ------------------------------------------------------------------ prediction
    def predict_scaled(self, X: np.ndarray) -> dict[str, np.ndarray]:
        probs = self.snn.predict_proba(X)
        vae_out = self.vae.score(X)
        out = self.engine.fuse(probs, vae_out["score"])
        out["probs"] = probs
        out["anomaly_raw"] = vae_out["raw"]
        return out

    def predict_frame(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        return self.predict_scaled(self.preprocessor.transform(df))

    def to_records(self, out: dict[str, np.ndarray], top_k: int = 3) -> list[dict[str, Any]]:
        records = []
        for i in range(len(out["risk"])):
            order = np.argsort(-out["probs"][i])[:top_k]
            records.append({
                "risk_score": round(float(out["risk"][i]), 4),
                "anomaly_score": round(float(out["anomaly"][i]), 4),
                "attack_probability": round(float(out["p_attack"][i]), 4),
                "predicted_class": str(out["label"][i]),
                "confidence": round(float(out["confidence"][i]), 4),
                "action": str(out["action"][i]),
                "top_classes": [{"class": self.classes[j], "probability": round(float(out["probs"][i][j]), 4)}
                                for j in order],
            })
        return records

    def info(self) -> dict[str, Any]:
        return {
            "classes": self.classes,
            "n_features": len(self.preprocessor.feature_names),
            "feature_names": self.preprocessor.feature_names,
            "thresholds": {"monitor": self.engine.thresholds.monitor, "block": self.engine.thresholds.block},
            "snn": {"num_steps": self.snn.model.num_steps, "hidden": list(self.snn.model.hidden),
                    "encoding": self.snn.model.encoding, "temperature": self.snn.temperature},
            "vae": {"latent_dim": self.vae.model.latent_dim},
        }
