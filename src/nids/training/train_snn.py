"""Train the spiking-neural-network classifier (optionally on GAN-augmented data)."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from nids.config import Config, artifact_dir
from nids.data.io import load_meta, load_split
from nids.models.snn import SNNClassifier, SpikingMLP, fit_temperature, predict_logits
from nids.tracking import Tracker
from nids.utils import class_weights, get_device, get_logger, save_json, set_seed

log = get_logger("nids.train_snn")


def _load_training_set(cfg: Config, augment: bool) -> tuple[np.ndarray, np.ndarray, int]:
    X, y = load_split(cfg, "train")
    n_synth = 0
    if augment:
        path = Path(cfg.paths.artifacts) / "gan" / "synthetic.npz"
        if path.exists():
            with np.load(path) as blob:
                Xs, ys = blob["X"].astype(np.float32), blob["y"].astype(np.int64)
            if len(Xs):
                X, y, n_synth = np.concatenate([X, Xs]), np.concatenate([y, ys]), len(Xs)
        else:
            log.warning("GAN augmentation requested but %s not found; training on real data only", path)
    return X, y, n_synth


def _validation_subset(y_val: np.ndarray, benign: int, benign_cap: int, seed: int) -> np.ndarray:
    """All attack flows + a capped random sample of benign -> a stable signal for macro-F1 early stopping."""
    rng = np.random.default_rng(seed)
    attack = np.flatnonzero(y_val != benign)
    benign_idx = np.flatnonzero(y_val == benign)
    if len(benign_idx) > benign_cap:
        benign_idx = rng.choice(benign_idx, benign_cap, replace=False)
    return np.sort(np.concatenate([attack, benign_idx]))


def train_snn(cfg: Config, augment: bool | None = None, tag: str = "snn") -> dict[str, float]:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    scfg = cfg.snn
    augment = scfg.use_gan_augmentation if augment is None else augment
    meta = load_meta(cfg)
    classes, n_classes, benign = meta["classes"], len(meta["classes"]), meta["benign_index"]

    X_train, y_train, n_synth = _load_training_set(cfg, augment)
    X_val, y_val = load_split(cfg, "val")
    sub = _validation_subset(y_val, benign, 50_000, cfg.seed)
    log.info("SNN '%s': %d train flows (%d synthetic), %d val flows (%d used per epoch) on %s",
             tag, len(X_train), n_synth, len(X_val), len(sub), device)

    counts = np.bincount(y_train, minlength=n_classes)
    weights = torch.from_numpy(class_weights(counts, scfg.class_weighting)).to(device)
    model = SpikingMLP(
        X_train.shape[1], n_classes, tuple(scfg.hidden), scfg.num_steps, scfg.beta, scfg.learn_beta,
        scfg.encoding, scfg.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=scfg.lr, weight_decay=scfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)

    best_f1, best_state, best_epoch, stale = -1.0, None, 0, 0
    with Tracker(cfg, tag, {"snn": scfg.to_dict(), "augment": augment, "n_synth": n_synth}) as tracker:
        for epoch in range(1, scfg.epochs + 1):
            model.train()
            perm = torch.randperm(len(xt), device=device)
            run_loss, run_correct = 0.0, 0
            for start in range(0, len(xt), scfg.batch_size):
                idx = perm[start:start + scfg.batch_size]
                xb, yb = xt[idx], yt[idx]
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, weight=weights)
                reg_scale = scfg.rate_reg * min(1.0, (epoch - 1) / max(1, scfg.rate_reg_warmup_epochs))
                if reg_scale > 0:
                    # Hinge on the mean firing rate: only rates ABOVE the target are penalised, so the penalty can
                    # sparsify a busy network but can never push a layer to silence. Warm-up lets the network
                    # learn the task first (a penalty from step 0 can kill every neuron before it learns anything).
                    excess = torch.stack([F.relu(r - scfg.rate_target) for r in model.last_rates]).mean()
                    loss = loss + reg_scale * excess
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                run_loss += loss.item() * len(idx)
                run_correct += (logits.argmax(1) == yb).sum().item()

            val_logits = predict_logits(model, X_val[sub], device, seed=scfg.eval_seed)
            pred = val_logits.argmax(1)
            present = np.unique(y_val[sub])
            macro_f1 = f1_score(y_val[sub], pred, labels=present, average="macro", zero_division=0)
            sched.step(macro_f1)
            rates = [round(float(r), 3) for r in model.last_rates]  # firing rates of the last training batch
            if min(rates) < 1e-3:
                log.warning("Dead hidden layer (firing rate ~0). Lower snn.rate_reg or raise snn.rate_reg_warmup_epochs.")
            log.info("epoch %02d | loss %.4f | train acc %.4f | val macro-F1 %.4f | rates %s | lr %.5f",
                     epoch, run_loss / len(xt), run_correct / len(xt), macro_f1, rates, opt.param_groups[0]["lr"])
            tracker.log_metrics({"train_loss": run_loss / len(xt), "val_macro_f1": macro_f1}, step=epoch)
            if macro_f1 > best_f1 + 1e-4:
                best_f1, best_state, best_epoch, stale = macro_f1, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                stale += 1
                if stale >= scfg.patience:
                    log.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                    break

        model.load_state_dict(best_state)
        # temperature scaling on the FULL natural-prevalence validation set
        val_logits = predict_logits(model, X_val, device, seed=scfg.eval_seed)
        temperature = fit_temperature(val_logits, y_val)
        clf = SNNClassifier(model, classes, device, temperature, scfg.eval_seed)
        out_dir = artifact_dir(cfg, tag)
        clf.save(out_dir)
        metrics = {
            "best_val_macro_f1": float(best_f1),
            "best_epoch": best_epoch,
            "temperature": temperature,
            "augmented": bool(augment),
            "n_train_real": int(len(X_train) - n_synth),
            "n_train_synthetic": int(n_synth),
        }
        save_json(metrics, out_dir / "train_metrics.json")
        tracker.log_metrics({"best_val_macro_f1": best_f1, "temperature": temperature})
        log.info("SNN '%s' done: %s", tag, metrics)
    return metrics
