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


def _override_heavy_validation(n=20000, seed=0):
    """Mirrors the failure seen in the smoke run: the monitor threshold sits above the risk of override flows.

    8% of benign flows are classifier-confusable (high p_attack), so the 95th-percentile benign risk is ~0.6+.
    A separate ~4% of benign flows are confidently Benign to the SNN but anomalous to the VAE (score 0.5-0.55),
    giving them risk ~0.4: below that threshold, yet forced to monitor by the Unknown override.
    """
    rng = np.random.default_rng(seed)
    y = np.where(rng.random(n) < 0.8, 0, 1)
    u = rng.random(n)
    confusable = (y == 0) & (u < 0.08)
    override = (y == 0) & (u >= 0.08) & (u < 0.12)
    p_attack = np.where(y == 0, rng.beta(1, 30, n), rng.beta(8, 2, n))
    p_attack = np.where(confusable, rng.uniform(0.5, 0.7, n), p_attack)
    p_attack = np.where(override, 0.01, p_attack)
    probs = np.c_[1 - p_attack, p_attack * 0.6, p_attack * 0.4]
    anomaly = np.where(override, rng.uniform(0.5, 0.55, n), 0.0)
    return probs, anomaly, y


def test_calibration_counts_unknown_override_against_monitor_budget():
    probs, anomaly, y = _override_heavy_validation()
    cfg = {**CFG, "target_fpr_block": 0.005}  # 0.5% block + ~4% override leaves room inside the 5% monitor budget
    eng = RiskEngine(CLASSES, cfg)
    res = eng.calibrate(probs, anomaly, y, cfg)
    real = res["realised_on_validation"]
    assert real["benign_unknown_override_rate"] > 0.03  # the override is a real share of the budget here
    # measured on the final fuse() actions, the monitor target holds despite the override
    action = eng.fuse(probs, anomaly)["action"][y == 0]
    assert real["benign_fpr_at_monitor_or_block"] == pytest.approx((action != "allow").mean())
    assert real["benign_fpr_at_monitor_or_block"] <= CFG["target_fpr_monitor"] + 1e-9
    assert real["benign_fpr_at_monitor_or_block"] >= CFG["target_fpr_monitor"] - 0.005  # budget is used, not wasted
    assert "warning" not in res


def test_calibration_warns_when_override_alone_exceeds_target():
    probs, anomaly, y = _override_heavy_validation()
    cfg = {**CFG, "target_fpr_monitor": 0.02}  # below the ~4% override rate
    eng = RiskEngine(CLASSES, cfg)
    res = eng.calibrate(probs, anomaly, y, cfg)
    assert "warning" in res
    assert eng.thresholds.monitor == eng.thresholds.block


def test_calibration_warns_when_block_target_and_override_crowd_out_monitor():
    probs, anomaly, y = _override_heavy_validation()  # 1% block + ~4% override > 5% monitor target
    res = RiskEngine(CLASSES, CFG).calibrate(probs, anomaly, y, CFG)
    assert "warning" in res and "Block target" in res["warning"]


def test_alert_funnel_arithmetic():
    is_attack = np.array([True, True, False, False, False, False])
    action = np.array(["block", "monitor", "monitor", "allow", "allow", "allow"], dtype=object)
    f = alert_funnel(is_attack, action)
    assert f["attack_recall_at_block"] == pytest.approx(0.5)
    assert f["attack_recall_at_review"] == pytest.approx(1.0)
    assert f["review_precision"] == pytest.approx(2 / 3)
    assert f["false_alerts_per_1k_flows"] == pytest.approx(1 / 6 * 1000)
