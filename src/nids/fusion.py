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

The monitor target is calibrated on the FINAL action, not on the risk score alone. A flow the classifier calls
Benign but the VAE scores >= `unknown_anomaly_threshold` is always sent to monitor, whatever its risk; those
benign flows are charged to the monitor budget first, and the risk threshold gets whatever budget is left.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nids.utils import get_logger, load_json, save_json

log = get_logger("nids.fusion")

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

    def unknown_mask(self, probs: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
        """Classifier says Benign, VAE says anomalous -> labelled Unknown-Anomaly and never silently allowed."""
        return (probs.argmax(axis=1) == self.benign_idx) & (anomaly >= self.unknown_thr)

    def fuse(self, probs: np.ndarray, anomaly: np.ndarray) -> dict[str, np.ndarray]:
        risk = self.risk(probs, anomaly)
        pred_idx = probs.argmax(axis=1)
        labels = np.array(self.classes, dtype=object)[pred_idx]
        unknown = self.unknown_mask(probs, anomaly)
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
    @staticmethod
    def _monitor_threshold(benign_risk: np.ndarray, forced: np.ndarray, target_fpr: float) -> float | None:
        """Lowest risk threshold m such that mean(forced | risk >= m) over benign flows stays <= target_fpr.

        `forced` marks benign flows that are reviewed regardless of risk (the Unknown-Anomaly override).
        Returns None when the forced flows alone already use up the budget.
        """
        n = len(benign_risk)
        budget = int(np.floor(target_fpr * n)) - int(forced.sum())
        if budget <= 0:
            return None
        free = np.sort(benign_risk[~forced])[::-1]  # descending
        if budget >= len(free):
            return float(free[-1]) if len(free) else 1.0
        # strictly above the (budget+1)-th largest free risk -> at most `budget` free flows reach monitor
        return float(np.nextafter(free[budget], np.inf))

    def calibrate(self, probs_val: np.ndarray, anomaly_val: np.ndarray, y_val: np.ndarray, fusion_cfg: dict[str, Any]) -> dict:
        """Pick thresholds from benign validation flows and store them on the engine.

        `block` is a plain risk quantile (the Unknown override never blocks). `monitor` is chosen so the share of
        benign flows reaching monitor OR block, including those forced there by the Unknown override, hits
        `target_fpr_monitor`. The realised rates are measured on the actual `fuse()` actions.
        """
        risk = self.risk(probs_val, anomaly_val)
        is_benign = y_val == self.benign_idx
        benign_risk = risk[is_benign]
        forced = self.unknown_mask(probs_val, anomaly_val)[is_benign]

        block = max(float(fusion_cfg["min_block_threshold"]),
                    float(np.quantile(benign_risk, 1.0 - fusion_cfg["target_fpr_block"])))
        target_monitor = float(fusion_cfg["target_fpr_monitor"])
        m = self._monitor_threshold(benign_risk, forced, target_monitor)
        warning = None
        if m is None:
            warning = (f"Unknown-Anomaly override alone sends {forced.mean():.2%} of benign validation flows to "
                       f"monitor, above the {target_monitor:.2%} target. Monitor threshold set to the block threshold; "
                       "raise fusion.unknown_anomaly_threshold or the monitor target.")
            log.warning(warning)
            m = block
        elif m > block:
            warning = (f"Block target ({fusion_cfg['target_fpr_block']:.2%}) plus the Unknown-Anomaly override "
                       f"({forced.mean():.2%}) exceed the {target_monitor:.2%} monitor target; every blocked flow is "
                       "also reviewed, so the realised monitor rate will be above target.")
            log.warning(warning)
        monitor = min(max(float(fusion_cfg["min_monitor_threshold"]), m), block)
        self.thresholds = Thresholds(monitor=monitor, block=block)

        action = self.fuse(probs_val, anomaly_val)["action"][is_benign]
        realised = {
            "benign_flows_val": int(is_benign.sum()),
            "benign_fpr_at_block": float((action == "block").mean()),
            "benign_fpr_at_monitor_or_block": float((action != "allow").mean()),
            "benign_unknown_override_rate": float(forced.mean()),
            "benign_monitor_by_override_only": float((forced & (benign_risk < monitor)).mean()),
        }
        result = {"thresholds": asdict(self.thresholds), "realised_on_validation": realised}
        if warning:
            result["warning"] = warning
        return result

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
