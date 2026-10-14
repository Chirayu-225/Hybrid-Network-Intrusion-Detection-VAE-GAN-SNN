"""Train the benign-only VAE and calibrate its anomaly scorer."""

from __future__ import annotations

import copy

import torch
from sklearn.metrics import roc_auc_score

from nids.config import Config, artifact_dir
from nids.data.io import load_meta, load_split
from nids.models.vae import VAE, AnomalyScorer, VAEDetector, encode_and_reconstruct, vae_loss
from nids.tracking import Tracker
from nids.utils import get_device, get_logger, save_json, set_seed

log = get_logger("nids.train_vae")


def _val_loss(model: VAE, X: torch.Tensor, beta: float, batch_size: int) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size]
            recon, mu, logvar = model(xb)
            loss, _, _ = vae_loss(recon, xb, mu, logvar, beta)
            total += loss.item() * len(xb)
            n += len(xb)
    return total / max(n, 1)


def train_vae(cfg: Config) -> dict[str, float]:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    vcfg = cfg.vae
    meta = load_meta(cfg)
    benign = meta["benign_index"]

    X_train, y_train = load_split(cfg, "train")
    X_val, y_val = load_split(cfg, "val")
    Xb_train, Xb_val = X_train[y_train == benign], X_val[y_val == benign]
    log.info("Training VAE on %d benign flows (val: %d benign) on %s", len(Xb_train), len(Xb_val), device)

    model = VAE(X_train.shape[1], tuple(vcfg.hidden), vcfg.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=vcfg.lr)
    xt = torch.from_numpy(Xb_train).to(device)
    xv = torch.from_numpy(Xb_val).to(device)

    best_loss, best_state, stale = float("inf"), None, 0
    with Tracker(cfg, "vae", {"vae": vcfg.to_dict()}) as tracker:
        for epoch in range(vcfg.epochs):
            model.train()
            beta = vcfg.beta * min(1.0, (epoch + 1) / max(1, vcfg.beta_warmup_epochs))
            perm = torch.randperm(len(xt), device=device)
            run_loss = run_rec = run_kl = 0.0
            for start in range(0, len(xt), vcfg.batch_size):
                xb = xt[perm[start:start + vcfg.batch_size]]
                recon, mu, logvar = model(xb)
                loss, rec, kl = vae_loss(recon, xb, mu, logvar, beta)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                run_loss += loss.item() * len(xb)
                run_rec += rec.item() * len(xb)
                run_kl += kl.item() * len(xb)
            n = len(xt)
            # early stopping always uses the final beta so epochs are comparable across the warm-up
            val = _val_loss(model, xv, vcfg.beta, vcfg.batch_size)
            log.info("epoch %02d | beta %.2f | train %.4f (rec %.4f kl %.3f) | val %.4f",
                     epoch + 1, beta, run_loss / n, run_rec / n, run_kl / n, val)
            tracker.log_metrics({"train_loss": run_loss / n, "val_loss": val, "kl": run_kl / n}, step=epoch)
            if val < best_loss - 1e-5:
                best_loss, best_state, stale = val, copy.deepcopy(model.state_dict()), 0
            elif epoch >= vcfg.beta_warmup_epochs:
                stale += 1
                if stale >= vcfg.patience:
                    log.info("Early stopping at epoch %d", epoch + 1)
                    break

        model.load_state_dict(best_state)
        mu_train, _ = encode_and_reconstruct(model, Xb_train, device)
        mu_val, recon_val = encode_and_reconstruct(model, Xb_val, device)
        scorer = AnomalyScorer.fit(
            mu_train, mu_val, recon_val, vcfg.latent_weight, vcfg.anomaly_floor_q, vcfg.anomaly_ceil_q
        )
        detector = VAEDetector(model, scorer, device)
        out_dir = artifact_dir(cfg, "vae")
        detector.save(out_dir)

        # sanity check only (not used for model selection): benign vs any-attack separability on validation
        raw_val = detector.score(X_val)["raw"]
        is_attack = (y_val != benign).astype(int)
        metrics = {"best_val_loss": best_loss}
        if is_attack.sum() > 0:
            metrics["val_auroc"] = float(roc_auc_score(is_attack, raw_val))
        # posterior-collapse diagnostic: how many latent dimensions actually carry information
        metrics["active_latent_dims"] = int((mu_train.var(axis=0) > 0.01).sum())
        save_json(metrics, out_dir / "train_metrics.json")
        tracker.log_metrics({k: v for k, v in metrics.items()})
        log.info("VAE done: %s", {k: round(v, 4) for k, v in metrics.items()})
    return metrics
