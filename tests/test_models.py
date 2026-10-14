import numpy as np
import pytest
import torch

from nids.models.gan import Critic, Generator, gradient_penalty, sample
from nids.models.snn import SpikingMLP, fit_temperature, softmax, synaptic_op_estimate
from nids.models.vae import VAE, AnomalyScorer, vae_loss
from nids.training.train_gan import find_atoms, snap_to_atoms

torch.manual_seed(0)


# ------------------------------------------------------------------ VAE
def test_vae_shapes_and_loss_decreases():
    x = torch.rand(256, 12)
    model = VAE(12, (16, 8), 4)
    recon, mu, logvar = model(x)
    assert recon.shape == x.shape and mu.shape == (256, 4) and logvar.shape == (256, 4)
    assert recon.min() >= 0 and recon.max() <= 1
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        r, m, lv = model(x)
        loss, _, _ = vae_loss(r, x, m, lv, beta=0.01)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first


def test_anomaly_scorer_is_calibrated_on_benign():
    rng = np.random.default_rng(0)
    mu_tr, mu_val = rng.normal(size=(2000, 4)), rng.normal(size=(2000, 4))
    rec_val = rng.gamma(2.0, 0.01, 2000)
    sc = AnomalyScorer.fit(mu_tr, mu_val, rec_val, 0.3, 0.95, 0.999)
    raw = sc.raw_score(mu_val, rec_val)
    s = sc.score(raw)
    assert 0 <= s.min() and s.max() <= 1
    assert (s > 0).mean() < 0.08          # ~5% of benign validation flows exceed the floor quantile
    far = sc.score(sc.raw_score(mu_val[:50] + 8.0, rec_val[:50] * 50))
    assert far.mean() > 0.95              # clearly anomalous flows saturate
    again = AnomalyScorer.from_dict(sc.to_dict())
    assert np.allclose(again.score(raw), s)


# ------------------------------------------------------------------ GAN
def test_gradient_penalty_is_finite_and_differentiable():
    critic = Critic(10, 3, (16,))
    real, fake = torch.rand(32, 10), torch.rand(32, 10)
    y = torch.randint(3, (32,))
    gp = gradient_penalty(critic, real, fake, y)
    assert torch.isfinite(gp)
    gp.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic.parameters())


def test_generator_conditioning_and_range():
    g = Generator(8, 4, 10, (16, 16), 4)
    out = sample(g, class_id=2, n=100, device=torch.device("cpu"))
    assert out.shape == (100, 10) and out.min() >= 0 and out.max() <= 1


def test_snap_to_atoms_restores_point_masses():
    rng = np.random.default_rng(0)
    real = np.c_[rng.integers(0, 2, 500).astype(float), np.where(rng.random(500) < 0.3, 0.0, rng.random(500))]
    atoms = find_atoms(real)
    fake = np.c_[rng.random(200) * 0.9, np.where(rng.random(200) < 0.3, 0.004, rng.random(200))]
    snapped = snap_to_atoms(fake.astype(np.float32), atoms)
    assert set(np.unique(snapped[:, 0])) <= {0.0, 1.0}      # binary column snapped
    assert (snapped[:, 1] == 0.0).sum() > 0                  # near-zero mass restored to exactly zero


# ------------------------------------------------------------------ SNN
@pytest.mark.parametrize("encoding", ["rate", "direct"])
def test_snn_forward_and_stats(encoding):
    model = SpikingMLP(12, 5, (16, 8), num_steps=5, encoding=encoding)
    x = torch.rand(20, 12)
    logits, stats = model(x, return_stats=True)
    assert logits.shape == (20, 5)
    assert all(0 <= r <= 1 for r in stats["hidden_rates"])
    est = synaptic_op_estimate(model, stats)
    assert est["ann_macs_per_flow"] == 12 * 16 + 16 * 8 + 8 * 5


def test_snn_rate_coding_is_deterministic_with_generator():
    model = SpikingMLP(12, 5, (16,), num_steps=8, encoding="rate").eval()
    x = torch.rand(10, 12)
    a = model(x, generator=torch.Generator().manual_seed(1))
    b = model(x, generator=torch.Generator().manual_seed(1))
    c = model(x, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_snn_gradients_flow_through_spikes():
    model = SpikingMLP(8, 3, (12,), num_steps=6)
    x, y = torch.rand(32, 8), torch.randint(3, (32,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    assert model.fcs[0].weight.grad is not None and model.fcs[0].weight.grad.abs().sum() > 0


def test_snn_clamps_out_of_range_inputs():
    model = SpikingMLP(6, 2, (8,), num_steps=4, encoding="direct").eval()
    x = torch.full((3, 6), 7.0)
    assert torch.allclose(model(x), model(torch.ones(3, 6)))


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 3000)
    logits = np.eye(3)[y] * 3 + rng.normal(0, 2.5, (3000, 3))
    logits *= 4.0  # deliberately overconfident
    t = fit_temperature(logits.astype(np.float32), y)
    def nll(temp):
        return -np.log(softmax(logits, temp)[np.arange(len(y)), y] + 1e-12).mean()

    assert t > 1.0 and nll(t) < nll(1.0)
