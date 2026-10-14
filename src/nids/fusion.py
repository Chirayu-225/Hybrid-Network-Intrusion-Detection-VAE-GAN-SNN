"""Fusion of the supervised (SNN) and unsupervised (VAE) signals into one risk score and an action.

    p_attack = 1 - P(Benign | flow)                       (SNN, temperature-calibrated)
    anomaly  = benign-percentile score in [0, 1]          (VAE)
    risk     = 1 - (1 - p_attack) * (1 - w * anomaly)     (noisy-OR: either detector can raise the alarm)

Why noisy-OR rather than a weighted average: the two detectors have complementary blind spots. The SNN
only knows the attack families it was trained on; the VAE only knows what "normal" looks like. An average
would let a confident "benign" from the SNN drown out a strong anomaly signal, which is exactly the
zero-day case this system exists for. `w < 1` stops the (noisier) anomaly signal from reaching risk = 1 alone.

Actions come from two thresholds calibrated on the natural-prevalence VALIDATION split so that the benign
false-positive rates at "block" and at "monitor" hit configured targets. Nothing is tuned on test data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nids.utils import load_json, save_json

UNKNOWN = "Unknown-Anomaly"
ACTIONS = ("allow", "monitor", "block")


@dataclass
class Thresholds:
    monitor: float
    block: float


class RiskEngine:
    def __init__(self, classes: list[str], fusion_cfg: dict[str, Any], thresholds: Thresholds | None = None):
        self.classes = list(classes)
        self.benign_idx = self.classes.index(fusion_cfg.get("benign_label", "Benign"))
        self.w = float(fusion_cfg["anomaly_weight"])
        self.unknown_thr = float(fusion_cfg["unknown_anomaly_threshold"])
        self.thresholds = thresholds or Thresholds(monitor=0.3, block=0.7)

    def risk(self, probs: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
        p_attack = 1.0 - probs[:, self.benign_idx]
        return 1.0 - (1.0 - p_attack) * (1.0 - self.w * anomaly)

    def fuse(self, probs: np.ndarray, anomaly: np.ndarray) -> dict[str, np.ndarray]:
        risk = self.risk(probs, anomaly)
        pred_idx = probs.argmax(axis=1)
        labels = np.array(self.classes, dtype=object)[pred_idx]
        unknown = (pred_idx == self.benign_idx) & (anomaly >= self.unknown_thr)
        labels = np.where(unknown, UNKNOWN, labels)

        action = np.full(len(risk), "allow", dtype=object)
        action[risk >= self.thresholds.monitor] = "monitor"
        action[risk >= self.thresholds.block] = "block"
        action[unknown & (action == "allow")] = "monitor"  # never silently allow something we call an anomaly
        return {
            "risk": risk,
            "p_attack": 1.0 - probs[:, self.benign_idx],
            "anomaly": anomaly,
            "pred_idx": pred_idx,
            "label": labels,
            "action": action,
            "confidence": probs.max(axis=1),
        }

    # ---- calibration
    def calibrate(self, probs_val: np.ndarray, anomaly_val: np.ndarray, y_val: np.ndarray, fusion_cfg: dict[str, Any]) -> dict:
        """Pick thresholds from benign validation flows and store them on the engine."""
        risk = self.risk(probs_val, anomaly_val)
        benign_risk = risk[y_val == self.benign_idx]
        block = max(float(fusion_cfg["min_block_threshold"]), float(np.quantile(benign_risk, 1.0 - fusion_cfg["target_fpr_block"])))
        monitor = max(float(fusion_cfg["min_monitor_threshold"]), float(np.quantile(benign_risk, 1.0 - fusion_cfg["target_fpr_monitor"])))
        self.thresholds = Thresholds(monitor=min(monitor, block), block=block)
        realised = {
            "benign_flows_val": int(len(benign_risk)),
            "benign_fpr_at_block": float((benign_risk >= self.thresholds.block).mean()),
            "benign_fpr_at_monitor_or_block": float((benign_risk >= self.thresholds.monitor).mean()),
        }
        return {"thresholds": asdict(self.thresholds), "realised_on_validation": realised}

    # ---- persistence
    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        save_json({"thresholds": asdict(self.thresholds), **(extra or {})}, path)

    @staticmethod
    def load_thresholds(path: str | Path) -> Thresholds:
        return Thresholds(**load_json(path)["thresholds"])


def alert_funnel(is_attack: np.ndarray, action: np.ndarray) -> dict[str, float]:
    """What an analyst actually sees. `review` = flows needing a human (monitor + block)."""
    n = len(action)
    review = np.isin(action, ["monitor", "block"])
    blocked = action == "block"
    n_attack, n_benign = int(is_attack.sum()), int((~is_attack).sum())
    return {
        "flows": n,
        "allow_pct": float((action == "allow").mean() * 100),
        "monitor_pct": float((action == "monitor").mean() * 100),
        "block_pct": float(blocked.mean() * 100),
        "attack_recall_at_block": float(blocked[is_attack].mean()) if n_attack else float("nan"),
        "attack_recall_at_review": float(review[is_attack].mean()) if n_attack else float("nan"),
        "block_precision": float(is_attack[blocked].mean()) if blocked.any() else float("nan"),
        "review_precision": float(is_attack[review].mean()) if review.any() else float("nan"),
        "benign_fpr_at_block": float(blocked[~is_attack].mean()) if n_benign else float("nan"),
        "benign_fpr_at_review": float(review[~is_attack].mean()) if n_benign else float("nan"),
        "false_alerts_per_1k_flows": float((review & ~is_attack).sum() / n * 1000),
    }
