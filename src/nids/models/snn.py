"""Spiking neural network classifier (snnTorch).

Architecture: rate-coded input spikes -> [Linear -> LIF] x H -> Linear -> leaky-integrator readout.

* Input: each min-max-scaled feature in [0, 1] is the firing probability of one input neuron per time step
  (Bernoulli rate coding). `encoding: direct` feeds the analog value as input current instead, which is
  deterministic and often a bit more accurate but is no longer a pure spike train.
* Hidden layers: leaky integrate-and-fire neurons trained end to end with a surrogate gradient.
* Readout: the output layer integrates current without ever spiking or resetting; the class logits are its
  membrane potential averaged over time. This is the standard SNN classification readout, avoids ties
  between output spike counts, and maps to hardware as a simple accumulator.

Inference is stochastic under rate coding, so predictions use a fixed-seed generator for reproducibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import snntorch as snn
import torch
from snntorch import surrogate
from torch import nn

from nids.utils import load_json, save_json


class SpikingMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        hidden: tuple[int, ...] = (256, 128),
        num_steps: int = 20,
        beta: float = 0.9,
        learn_beta: bool = True,
        encoding: str = "rate",
        dropout: float = 0.1,
        slope: float = 25.0,
    ):
        super().__init__()
        if encoding not in ("rate", "direct"):
            raise ValueError("encoding must be 'rate' or 'direct'")
        self.in_dim, self.n_classes, self.hidden = in_dim, n_classes, tuple(hidden)
        self.num_steps, self.beta, self.learn_beta = num_steps, beta, learn_beta
        self.encoding, self.dropout_p, self.slope = encoding, dropout, slope

        grad = surrogate.fast_sigmoid(slope=slope)
        dims = [in_dim, *hidden]
        self.fcs = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.lifs = nn.ModuleList(
            [snn.Leaky(beta=beta, spike_grad=grad, learn_beta=learn_beta, threshold=1.0) for _ in hidden]
        )
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dims[-1], n_classes)
        self.out_lif = snn.Leaky(beta=beta, learn_beta=learn_beta, threshold=1.0, reset_mechanism="none")

    def forward(
        self, x: torch.Tensor, generator: torch.Generator | None = None, return_stats: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """x: [B, F] in [0, 1]. Returns time-averaged output membrane potential [B, C] (the logits)."""
        x = x.clamp(0.0, 1.0)
        batch = x.size(0)
        mems = [torch.zeros(batch, fc.out_features, device=x.device) for fc in self.fcs]
        mem_out = torch.zeros(batch, self.n_classes, device=x.device)
        logits = torch.zeros(batch, self.n_classes, device=x.device)
        spike_counts = [torch.zeros((), device=x.device) for _ in self.fcs]
        input_spikes = torch.zeros((), device=x.device)

        for _ in range(self.num_steps):
            if self.encoding == "rate":
                s = torch.bernoulli(x, generator=generator)
            else:
                s = x
            input_spikes = input_spikes + s.sum()
            for i, (fc, lif) in enumerate(zip(self.fcs, self.lifs)):
                spk, mems[i] = lif(fc(s), mems[i])
                spike_counts[i] = spike_counts[i] + spk.sum()
                s = self.drop(spk)
            _, mem_out = self.out_lif(self.out(s), mem_out)
            logits = logits + mem_out
        logits = logits / self.num_steps
        if self.training:  # differentiable mean firing rate per hidden layer, for the sparsity regulariser
            self.last_rates = [c / float(batch * self.num_steps * fc.out_features)
                               for c, fc in zip(spike_counts, self.fcs)]

        if not return_stats:
            return logits
        denom = float(batch * self.num_steps)
        stats = {
            "input_rate": float(input_spikes / (denom * self.in_dim)),
            "hidden_rates": [float(c / (denom * fc.out_features)) for c, fc in zip(spike_counts, self.fcs)],
            "spikes_per_flow_in": float(input_spikes / batch),
            "spikes_per_flow_hidden": [float(c / batch) for c in spike_counts],
        }
        return logits, stats

    def config(self) -> dict[str, Any]:
        return {"in_dim": self.in_dim, "n_classes": self.n_classes, "hidden": list(self.hidden),
                "num_steps": self.num_steps, "beta": self.beta, "learn_beta": self.learn_beta,
                "encoding": self.encoding, "dropout": self.dropout_p, "slope": self.slope}


E_MAC_PJ, E_AC_PJ = 4.6, 0.9


def synaptic_op_estimate(model: SpikingMLP, stats: dict[str, Any]) -> dict[str, float]:
    """Energy *proxy*: SNN accumulates (+ first-layer MACs under direct encoding) vs MACs of the same-shape ANN.

    Event-driven execution: every spike entering a layer triggers `fan_out` additions (AC). Under `rate`
    encoding the input spikes pay AC cost like any other layer. Under `direct` encoding the first layer
    consumes analog values, so it costs `in x hidden` MACs once per flow (its output current is reused for all
    time steps). A dense ANN costs `in x out` MACs per layer. Energy uses 45 nm / 32-bit float figures
    (MAC 4.6 pJ, AC 0.9 pJ; Horowitz, ISSCC 2014). Neuron-state updates and memory traffic are ignored, so
    this is an order-of-magnitude indicator for comparing configurations, not a measured energy figure.
    """
    dims = [model.in_dim, *model.hidden, model.n_classes]
    hidden_spikes = stats["spikes_per_flow_hidden"]
    ac, mac = 0.0, 0.0
    if model.encoding == "direct":
        mac += dims[0] * dims[1]
    else:
        ac += stats["spikes_per_flow_in"] * dims[1]
    for i, s in enumerate(hidden_spikes):  # spikes leaving hidden layer i feed layer i+1 (or the readout)
        ac += s * dims[i + 2]
    ann_mac = sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))
    snn_energy = mac * E_MAC_PJ + ac * E_AC_PJ
    return {
        "snn_accumulates_per_flow": float(ac),
        "snn_macs_per_flow": float(mac),
        "ann_macs_per_flow": float(ann_mac),
        "energy_ratio_snn_over_ann": float(snn_energy / (ann_mac * E_MAC_PJ)),
    }


@torch.no_grad()
def predict_logits(
    model: SpikingMLP, X: np.ndarray, device: torch.device, batch_size: int = 4096, seed: int | None = 1234
) -> np.ndarray:
    model.eval()
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
    out = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size]).to(device)
        out.append(model(xb, generator=gen).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, model.n_classes), np.float32)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Post-hoc temperature scaling on validation logits (minimises NLL). Makes probabilities usable in fusion."""
    lt = torch.from_numpy(logits).double()
    yt = torch.from_numpy(y).long()
    log_t = torch.zeros(1, dtype=torch.double, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lt / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 50.0).item())


class SNNClassifier:
    """Model + temperature + class names, with save/load. Used by training, evaluation and serving."""

    def __init__(self, model: SpikingMLP, classes: list[str], device: torch.device,
                 temperature: float = 1.0, seed: int | None = 1234):
        self.model, self.classes, self.device = model.to(device).eval(), classes, device
        self.temperature, self.seed = temperature, seed

    def logits(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        return predict_logits(self.model, X, self.device, batch_size, self.seed)

    def predict_proba(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        return softmax(self.logits(X, batch_size), self.temperature)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.model.config(), "state_dict": self.model.state_dict()}, directory / "snn.pt")
        save_json({"temperature": self.temperature, "classes": self.classes, "seed": self.seed},
                  directory / "calibration.json")

    @classmethod
    def load(cls, directory: str | Path, device: torch.device) -> SNNClassifier:
        directory = Path(directory)
        blob = torch.load(directory / "snn.pt", map_location=device, weights_only=True)
        cfg = blob["config"]
        cfg["hidden"] = tuple(cfg["hidden"])
        model = SpikingMLP(**cfg)
        model.load_state_dict(blob["state_dict"])
        cal = load_json(directory / "calibration.json")
        return cls(model, cal["classes"], device, cal["temperature"], cal.get("seed"))
