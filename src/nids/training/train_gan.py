"""Train the conditional WGAN-GP on rare attack classes and generate synthetic training samples.

Guard-rails against the usual ways GAN augmentation goes wrong:

* Only classes with at least `gan.min_real_samples` real train flows are modelled; below that a GAN simply
  memorises its handful of examples, so such classes are reported and skipped.
* Synthetic volume per class is capped at `max_synth_per_real` x the real count so synthetic data can never
  drown out the real signal.
* Only the TRAIN split is ever used. Validation and test stay 100% real.
* A fidelity report (marginal moments, real-vs-synthetic discriminability, nearest-neighbour copy check)
  is written next to the samples so quality is measured, not assumed.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors

from nids.config import Config, artifact_dir
from nids.data.io import load_meta, load_split
from nids.models.gan import Critic, Generator, gradient_penalty, sample
from nids.tracking import Tracker
from nids.utils import get_device, get_logger, save_json, set_seed

log = get_logger("nids.train_gan")


def find_atoms(X: np.ndarray, min_freq: float = 0.01, max_discrete: int = 8) -> list[tuple[np.ndarray, bool]]:
    """Per feature: the values real data piles up on ("atoms"), and whether the feature is low-cardinality.

    CICFlowMeter features are full of point masses (flag counts, zeros, the -1 window-size sentinel). A
    sigmoid-output generator produces 0.0003 where real data has exactly 0, which makes synthetic flows
    trivially detectable and, worse, teaches the classifier a spurious "exactly zero" shortcut. Snapping
    generated values onto the real atoms removes that artefact (the same job CTGAN's mode-specific
    normalisation does for discrete columns).
    """
    atoms = []
    for j in range(X.shape[1]):
        vals, counts = np.unique(np.round(X[:, j], 6), return_counts=True)
        if len(vals) <= max_discrete:
            atoms.append((vals, True))
        else:
            atoms.append((vals[counts / len(X) >= min_freq], False))
    return atoms


def snap_to_atoms(X: np.ndarray, atoms: list[tuple[np.ndarray, bool]], tol: float = 0.02) -> np.ndarray:
    out = X.copy()
    for j, (vals, always) in enumerate(atoms):
        if len(vals) == 0:
            continue
        nearest = vals[np.abs(out[:, j:j + 1] - vals[None, :]).argmin(axis=1)]
        close = np.abs(out[:, j] - nearest) <= (np.inf if always else tol)
        out[close, j] = nearest[close]
    return out


def fidelity_report(real: np.ndarray, synth: np.ndarray, max_samples: int, seed: int) -> dict[str, float]:
    """Compare synthetic vs real samples of ONE class (both in scaled feature space)."""
    rng = np.random.default_rng(seed)
    if len(real) > max_samples:
        real = real[rng.choice(len(real), max_samples, replace=False)]
    if len(synth) > max_samples:
        synth = synth[rng.choice(len(synth), max_samples, replace=False)]
    report: dict[str, float] = {
        "n_real": int(len(real)),
        "n_synth": int(len(synth)),
        "mean_abs_mean_diff": float(np.abs(real.mean(0) - synth.mean(0)).mean()),
        "median_std_ratio": float(np.median((synth.std(0) + 1e-6) / (real.std(0) + 1e-6))),
    }
    # 1) can a classifier tell real from synthetic? ideal AUC = 0.5
    n = min(len(real), len(synth))
    if n >= 10:
        Xd = np.vstack([real[:n], synth[:n]])
        yd = np.r_[np.zeros(n), np.ones(n)]
        folds = min(3, n)
        clf = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1, random_state=seed)
        auc = cross_val_score(clf, Xd, yd, cv=StratifiedKFold(folds, shuffle=True, random_state=seed), scoring="roc_auc")
        report["real_vs_synth_auc"] = float(auc.mean())
    # 2) is the generator copying training samples? ratio << 1 means yes
    if len(real) >= 3:
        nn_real = NearestNeighbors(n_neighbors=2).fit(real)
        real_loo = nn_real.kneighbors(real)[0][:, 1]
        synth_to_real = nn_real.kneighbors(synth[: min(len(synth), 2000)], n_neighbors=1)[0][:, 0]
        report["nn_distance_ratio"] = float(np.median(synth_to_real) / (np.median(real_loo) + 1e-9))
    return report


def train_gan(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    gcfg = cfg.gan
    meta = load_meta(cfg)
    classes = meta["classes"]
    X, y = load_split(cfg, "train")
    train_counts = meta["counts"]["train"]
    out_dir = artifact_dir(cfg, "gan")

    eligible, skipped = [], {}
    for name in meta["rare_classes"]:
        cid = classes.index(name)
        if train_counts[cid] >= gcfg.min_real_samples:
            eligible.append(cid)
        else:
            skipped[name] = int(train_counts[cid])
    if skipped:
        log.warning("Skipping classes with < %d real train flows (a GAN would memorise them): %s",
                    gcfg.min_real_samples, skipped)
    if not eligible:
        log.warning("No eligible rare classes; writing an empty synthetic set")
        np.savez_compressed(out_dir / "synthetic.npz", X=np.empty((0, X.shape[1]), np.float32), y=np.empty(0, np.int64))
        save_json({"skipped": skipped, "classes": {}}, out_dir / "fidelity.json")
        return {"skipped": skipped}

    atoms = find_atoms(np.clip(X, 0.0, 1.0))
    real_by_class = {cid: torch.from_numpy(np.clip(X[y == cid], 0.0, 1.0)).to(device) for cid in eligible}
    n_classes, n_feat = len(classes), X.shape[1]
    G = Generator(gcfg.noise_dim, n_classes, n_feat, tuple(gcfg.g_hidden), gcfg.embed_dim).to(device)
    D = Critic(n_feat, n_classes, tuple(gcfg.d_hidden), gcfg.embed_dim).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=gcfg.lr, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=gcfg.lr, betas=(0.0, 0.9))
    elig_t = torch.tensor(eligible, device=device)
    log.info("Training WGAN-GP on classes %s for %d iterations on %s",
             [classes[c] for c in eligible], gcfg.iterations, device)

    def real_batch(bs: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Class-balanced sampling: pick a class uniformly, then a real flow of that class."""
        labels = elig_t[torch.randint(len(eligible), (bs,), device=device)]
        batch = torch.empty(bs, n_feat, device=device)
        for cid in eligible:
            mask = labels == cid
            k = int(mask.sum())
            if k:
                data = real_by_class[cid]
                batch[mask] = data[torch.randint(len(data), (k,), device=device)]
        return batch, labels

    history = []
    with Tracker(cfg, "gan", {"gan": gcfg.to_dict()}) as tracker:
        for it in range(1, gcfg.iterations + 1):
            for _ in range(gcfg.n_critic):
                real, labels = real_batch(gcfg.batch_size)
                z = torch.randn(len(real), gcfg.noise_dim, device=device)
                with torch.no_grad():
                    fake = G(z, labels)
                d_real, d_fake = D(real, labels).mean(), D(fake, labels).mean()
                gp = gradient_penalty(D, real, fake, labels)
                loss_d = d_fake - d_real + gcfg.gp_lambda * gp
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()

            z = torch.randn(gcfg.batch_size, gcfg.noise_dim, device=device)
            _, labels = real_batch(gcfg.batch_size)
            loss_g = -D(G(z, labels), labels).mean()
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            if it % max(1, gcfg.iterations // 10) == 0 or it == 1:
                w_dist = float((d_real - d_fake).item())
                history.append({"iter": it, "w_distance": w_dist, "loss_g": float(loss_g.item()), "gp": float(gp.item())})
                log.info("iter %5d | W-dist %.4f | loss_G %.4f | GP %.4f", it, w_dist, loss_g.item(), gp.item())
                tracker.log_metrics({"w_distance": w_dist, "loss_g": float(loss_g.item())}, step=it)

        # ---- generate
        G.eval()
        xs, ys, per_class, fidelity = [], [], {}, {}
        for cid in eligible:
            n_real = train_counts[cid]
            need = min(max(0, gcfg.target_per_class - n_real), gcfg.max_synth_per_real * n_real)
            per_class[classes[cid]] = {"real": int(n_real), "synthetic": int(need)}
            if need == 0:
                continue
            synth = snap_to_atoms(sample(G, cid, need, device).numpy().astype(np.float32), atoms)
            xs.append(synth)
            ys.append(np.full(need, cid, dtype=np.int64))
            real_c = np.clip(X[y == cid], 0.0, 1.0)
            fidelity[classes[cid]] = fidelity_report(real_c, synth, gcfg.fidelity_max_samples, cfg.seed)
            log.info("%-12s real %5d -> +%5d synthetic | fidelity %s", classes[cid], n_real, need,
                     {k: round(v, 3) for k, v in fidelity[classes[cid]].items() if k not in ("n_real", "n_synth")})

        X_syn = np.concatenate(xs) if xs else np.empty((0, n_feat), np.float32)
        y_syn = np.concatenate(ys) if ys else np.empty(0, np.int64)
        np.savez_compressed(out_dir / "synthetic.npz", X=X_syn, y=y_syn)
        torch.save({"config": G.config(), "state_dict": G.state_dict(), "classes": eligible}, out_dir / "generator.pt")
        summary = {"skipped": skipped, "per_class": per_class, "classes": fidelity, "history": history}
        save_json(summary, out_dir / "fidelity.json")
        log.info("Generated %d synthetic flows", len(X_syn))
    return summary
