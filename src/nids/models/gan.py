"""Class-conditional Wasserstein GAN with gradient penalty (WGAN-GP) for tabular flow features.

Why WGAN-GP rather than a vanilla GAN: rare attack classes give us tens to a few thousand samples, exactly
the regime where vanilla GANs mode-collapse or oscillate. The Wasserstein critic gives a usable training
signal even when real and fake distributions barely overlap, and the gradient penalty (no weight clipping,
no batch-norm in the critic) keeps it stable. Conditioning on the class lets one network share structure
across attack families while still generating each class on demand.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        n_classes: int,
        out_dim: int,
        hidden: tuple[int, ...] = (128, 256, 128),
        embed_dim: int = 16,
    ):
        super().__init__()
        self.noise_dim, self.n_classes, self.out_dim = noise_dim, n_classes, out_dim
        self.hidden, self.embed_dim = tuple(hidden), embed_dim
        self.embed = nn.Embedding(n_classes, embed_dim)
        layers: list[nn.Module] = []
        prev = noise_dim + embed_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.LeakyReLU(0.2)]
            prev = h
        layers += [nn.Linear(prev, out_dim), nn.Sigmoid()]  # features are min-max scaled to [0, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, self.embed(y)], dim=1))

    def config(self) -> dict[str, Any]:
        return {"noise_dim": self.noise_dim, "n_classes": self.n_classes, "out_dim": self.out_dim,
                "hidden": list(self.hidden), "embed_dim": self.embed_dim}


class Critic(nn.Module):
    """Wasserstein critic. No batch-norm (it would break the per-sample gradient penalty)."""

    def __init__(self, in_dim: int, n_classes: int, hidden: tuple[int, ...] = (256, 128), embed_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)
        layers: list[nn.Module] = []
        prev = in_dim + embed_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.embed(y)], dim=1)).squeeze(1)


def gradient_penalty(critic: Critic, real: torch.Tensor, fake: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """E[(||grad_x critic(x_hat)||_2 - 1)^2] on random interpolates between real and fake samples."""
    alpha = torch.rand(real.size(0), 1, device=real.device)
    interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    scores = critic(interp, y)
    grads = torch.autograd.grad(
        outputs=scores, inputs=interp, grad_outputs=torch.ones_like(scores), create_graph=True, retain_graph=True
    )[0]
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


@torch.no_grad()
def sample(generator: Generator, class_id: int, n: int, device: torch.device, batch_size: int = 4096) -> torch.Tensor:
    generator.eval()
    out = []
    for start in range(0, n, batch_size):
        m = min(batch_size, n - start)
        z = torch.randn(m, generator.noise_dim, device=device)
        y = torch.full((m,), class_id, dtype=torch.long, device=device)
        out.append(generator(z, y).clamp(0.0, 1.0).cpu())
    return torch.cat(out) if out else torch.empty(0, generator.out_dim)
