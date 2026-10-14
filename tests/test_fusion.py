import numpy as np
import pytest

from nids.fusion import UNKNOWN, RiskEngine, Thresholds, alert_funnel

CLASSES = ["Benign", "DoS", "Bot"]
CFG = {
    "anomaly_weight": 0.8, "unknown_anomaly_threshold": 0.5, "benign_label": "Benign",
    "target_fpr_block": 0.01, "target_fpr_monitor": 0.05, "min_block_threshold": 0.5, "min_monitor_threshold": 0.2,
}


def engine():
    return RiskEngine(CLASSES, CFG, Thresholds(monitor=0.3, block=0.7))


def test_confident_attack_is_blocked_and_clean_benign_allowed():
    probs = np.array([[0.01, 0.98, 0.01], [0.99, 0.005, 0.005]])
    out = engine().fuse(probs, np.array([0.0, 0.0]))
    assert list(out["action"]) == ["block", "allow"]
    assert list(out["label"]) == ["DoS", "Benign"]


def test_zero_day_flagged_by_vae_when_classifier_says_benign():
    probs = np.array([[0.97, 0.02, 0.01]])
    out = engine().fuse(probs, np.array([1.0]))
    assert out["label"][0] == UNKNOWN
    assert out["action"][0] in ("monitor", "block")
    assert out["risk"][0] > 0.7  # noisy-OR: the anomaly alone is enough to raise the alarm


def test_unknown_anomaly_is_never_silently_allowed():
    eng = RiskEngine(CLASSES, CFG, Thresholds(monitor=0.95, block=0.99))
    out = eng.fuse(np.array([[0.99, 0.005, 0.005]]), np.array([0.6]))
    assert out["label"][0] == UNKNOWN and out["action"][0] == "monitor"


def test_risk_is_monotone_in_both_signals():
    eng = engine()
    base = np.array([[0.8, 0.1, 0.1]])
    r0 = eng.risk(base, np.array([0.0]))
    assert eng.risk(base, np.array([0.5])) > r0
    assert eng.risk(np.array([[0.4, 0.4, 0.2]]), np.array([0.0])) > r0


def test_calibration_hits_target_fpr_on_validation():
    rng = np.random.default_rng(0)
    n = 20000
    y = np.where(rng.random(n) < 0.8, 0, 1)
    p_attack = np.where(y == 0, rng.beta(1, 12, n), rng.beta(8, 2, n))
    probs = np.c_[1 - p_attack, p_attack * 0.6, p_attack * 0.4]
    eng = RiskEngine(CLASSES, CFG)
    res = eng.calibrate(probs, np.zeros(n), y, CFG)
    real = res["realised_on_validation"]
    assert eng.thresholds.monitor <= eng.thresholds.block
    assert real["benign_fpr_at_block"] <= CFG["target_fpr_block"] + 0.003
    assert real["benign_fpr_at_monitor_or_block"] <= CFG["target_fpr_monitor"] + 0.01


def test_alert_funnel_arithmetic():
    is_attack = np.array([True, True, False, False, False, False])
    action = np.array(["block", "monitor", "monitor", "allow", "allow", "allow"], dtype=object)
    f = alert_funnel(is_attack, action)
    assert f["attack_recall_at_block"] == pytest.approx(0.5)
    assert f["attack_recall_at_review"] == pytest.approx(1.0)
    assert f["review_precision"] == pytest.approx(2 / 3)
    assert f["false_alerts_per_1k_flows"] == pytest.approx(1 / 6 * 1000)
