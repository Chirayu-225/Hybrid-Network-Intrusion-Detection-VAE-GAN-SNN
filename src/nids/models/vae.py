"""Variational autoencoder for unsupervised anomaly detection.

The VAE is trained on BENIGN flows only. At inference a flow is scored by two complementary signals:

* reconstruction error  - how badly the decoder reproduces the flow from its latent code
* latent distance       - Mahalanobis distance of the flow's latent mean from the benign latent cloud

Both are log-transformed, robust-standardised on benign *validation* flows and blended. The blend is then
mapped through the benign-validation empirical CDF so that "anomaly score = 0" means "indistinguishable
from the bulk of benign traffic" and "1" means "beyond almost every benign validation flow". That makes the
score threshold-free and comparable across retrains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from nids.utils import load_json, save_json

_QUANTILE_LEVELS = np.linspace(0.0, 1.0, 10001)


class VAE(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (128, 64), latent_dim: int = 16):
        super().__init__()
        self.in_dim, self.hidden, self.latent_dim = in_dim, tuple(hidden), latent_dim
        enc: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            enc += [nn.Linear(prev, h), nn.LeakyReLU(0.1)]
            prev = h
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        dec: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden):
            dec += [nn.Linear(prev, h), nn.LeakyReLU(0.1)]
            prev = h
        dec += [nn.Linear(prev, in_dim), nn.Sigmoid()]  # inputs live in [0, 1]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10.0, 10.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    def config(self) -> dict[str, Any]:
        return {"in_dim": self.in_dim, "hidden": list(self.hidden), "latent_dim": self.latent_dim}


def vae_loss(
    recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Negative ELBO with a beta-weighted KL term. Per-sample sums, averaged over the batch."""
    recon_loss = ((recon - x) ** 2).sum(dim=1).mean()
    kl = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    return recon_loss + beta * kl, recon_loss, kl


@torch.no_grad()
def encode_and_reconstruct(
    model: VAE, X: np.ndarray, device: torch.device, batch_size: int = 8192
) -> tuple[np.ndarray, np.ndarray]:
    """Return latent means [N, L] and per-flow mean squared reconstruction error [N] (decoded from the mean)."""
    model.eval()
    mus, errs = [], []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size]).to(device)
        mu, _ = model.encode(xb)
        err = ((model.decode(mu) - xb) ** 2).mean(dim=1)
        mus.append(mu.cpu().numpy())
        errs.append(err.cpu().numpy())
    return np.concatenate(mus).astype(np.float64), np.concatenate(errs).astype(np.float64)


class AnomalyScorer:
    """Turns (reconstruction error, latent) into a calibrated anomaly score in [0, 1]."""

    def __init__(
        self,
        mu_ref: np.ndarray,
        precision: np.ndarray,
        norm: dict[str, float],
        quantiles: np.ndarray,
        latent_weight: float,
        floor_q: float,
        ceil_q: float,
    ):
        self.mu_ref, self.precision = mu_ref, precision
        self.norm, self.quantiles = norm, quantiles
        self.latent_weight, self.floor_q, self.ceil_q = latent_weight, floor_q, ceil_q

    # -- components
    def latent_distance_sq(self, mu: np.ndarray) -> np.ndarray:
        diff = mu - self.mu_ref
        return np.einsum("ij,jk,ik->i", diff, self.precision, diff)

    def raw_score(self, mu: np.ndarray, recon_err: np.ndarray) -> np.ndarray:
        z_recon = (np.log(recon_err + 1e-9) - self.norm["recon_med"]) / self.norm["recon_scale"]
        z_lat = (np.log(self.latent_distance_sq(mu) + 1e-9) - self.norm["lat_med"]) / self.norm["lat_scale"]
        return (1.0 - self.latent_weight) * z_recon + self.latent_weight * z_lat

    def score(self, raw: np.ndarray) -> np.ndarray:
        """Map raw scores to [0, 1]: 0 up to the floor quantile of benign validation, 1 beyond the ceiling."""
        u = np.interp(raw, self.quantiles, _QUANTILE_LEVELS)
        return np.clip((u - self.floor_q) / (self.ceil_q - self.floor_q), 0.0, 1.0)

    def raw_threshold(self, benign_quantile: float) -> float:
        return float(np.interp(benign_quantile, _QUANTILE_LEVELS, self.quantiles))

    # -- fitting
    @classmethod
    def fit(
        cls,
        mu_train: np.ndarray,
        mu_val: np.ndarray,
        recon_val: np.ndarray,
        latent_weight: float,
        floor_q: float,
        ceil_q: float,
        ridge: float = 0.05,
    ) -> AnomalyScorer:
        mu_ref = mu_train.mean(axis=0)
        cov = np.cov(mu_train, rowvar=False) + ridge * np.eye(mu_train.shape[1])
        precision = np.linalg.inv(cov)
        tmp = cls(mu_ref, precision, {}, np.zeros(1), latent_weight, floor_q, ceil_q)

        log_recon = np.log(recon_val + 1e-9)
        log_lat = np.log(tmp.latent_distance_sq(mu_val) + 1e-9)

        def robust(v: np.ndarray) -> tuple[float, float]:
            med = float(np.median(v))
            return med, float(max(1.4826 * np.median(np.abs(v - med)), 1e-3))

        recon_med, recon_scale = robust(log_recon)
        lat_med, lat_scale = robust(log_lat)
        norm = {"recon_med": recon_med, "recon_scale": recon_scale, "lat_med": lat_med, "lat_scale": lat_scale}
        tmp.norm = norm
        raw_val = tmp.raw_score(mu_val, recon_val)
        tmp.quantiles = np.maximum.accumulate(np.quantile(raw_val, _QUANTILE_LEVELS))
        return tmp

    # -- persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "mu_ref": self.mu_ref.tolist(),
            "precision": self.precision.tolist(),
            "norm": self.norm,
            "quantiles": self.quantiles.tolist(),
            "latent_weight": self.latent_weight,
            "floor_q": self.floor_q,
            "ceil_q": self.ceil_q,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AnomalyScorer:
        return cls(
            np.asarray(d["mu_ref"]), np.asarray(d["precision"]), d["norm"], np.asarray(d["quantiles"]),
            d["latent_weight"], d["floor_q"], d["ceil_q"],
        )


class VAEDetector:
    """Model + scorer bundle with save/load, used by training, evaluation and serving."""

    def __init__(self, model: VAE, scorer: AnomalyScorer, device: torch.device):
        self.model, self.scorer, self.device = model.to(device).eval(), scorer, device

    def score(self, X: np.ndarray, batch_size: int = 8192) -> dict[str, np.ndarray]:
        mu, recon = encode_and_reconstruct(self.model, X, self.device, batch_size)
        raw = self.scorer.raw_score(mu, recon)
        return {"raw": raw, "score": self.scorer.score(raw), "recon_err": recon,
                "latent_d2": self.scorer.latent_distance_sq(mu)}

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.model.config(), "state_dict": self.model.state_dict()}, directory / "vae.pt")
        save_json(self.scorer.to_dict(), directory / "scorer.json")

    @classmethod
    def load(cls, directory: str | Path, device: torch.device) -> VAEDetector:
        directory = Path(directory)
        blob = torch.load(directory / "vae.pt", map_location=device, weights_only=True)
        model = VAE(**blob["config"])
        model.load_state_dict(blob["state_dict"])
        return cls(model, AnomalyScorer.from_dict(load_json(directory / "scorer.json")), device)
