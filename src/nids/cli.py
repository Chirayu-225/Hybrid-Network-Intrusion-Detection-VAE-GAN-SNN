"""`nids` command-line interface.

Typical full run (real CICIDS2017 in data/raw/cicids2017):

    nids prepare && nids train-vae && nids train-gan && nids train-snn && nids calibrate && nids evaluate

or just `nids run-all`. CI / smoke test on synthetic data: `nids --config configs/smoke.yaml run-all --synthetic`.
"""

from __future__ import annotations

import argparse
import json
import sys

from nids.config import Config, load_config
from nids.utils import get_logger

log = get_logger("nids.cli")


def _synth(cfg: Config) -> None:
    from nids.data.synthetic import generate, write_csvs

    df = generate(cfg.data.synthetic.n_rows, cfg.data.synthetic.seed)
    write_csvs(df, cfg.paths.raw_dir, cfg.data.synthetic.n_files)


def _run_all(cfg: Config, args: argparse.Namespace) -> None:
    from nids.calibrate import calibrate
    from nids.data.preprocess import prepare
    from nids.evaluation import evaluate
    from nids.training.train_gan import train_gan
    from nids.training.train_snn import train_snn
    from nids.training.train_vae import train_vae

    if args.synthetic:
        _synth(cfg)
    prepare(cfg)
    train_vae(cfg)
    train_gan(cfg)
    train_snn(cfg, augment=None, tag="snn")
    calibrate(cfg, "snn")
    evaluate(cfg, "snn")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nids", description="Hybrid VAE + GAN + SNN network intrusion detection")
    p.add_argument("--config", default="configs/default.yaml", help="path to a YAML config")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("synth-data", help="write a synthetic CICIDS-schema dataset (CI/smoke tests only)")
    sub.add_parser("prepare", help="clean, split and scale the raw CSVs")
    sub.add_parser("train-vae", help="train the benign-only VAE anomaly detector")
    sub.add_parser("train-gan", help="train the conditional WGAN-GP and generate synthetic rare-class flows")

    s = sub.add_parser("train-snn", help="train the spiking classifier")
    s.add_argument("--no-augment", action="store_true", help="train on real data only (baseline)")
    s.add_argument("--tag", default="snn", help="artifact sub-folder name, e.g. 'snn_baseline'")

    c = sub.add_parser("calibrate", help="fit fusion thresholds on the validation split")
    c.add_argument("--tag", default="snn")

    e = sub.add_parser("evaluate", help="metrics + figures on the test split")
    e.add_argument("--tag", default="snn")

    b = sub.add_parser("benchmark", help="single-flow CPU latency")
    b.add_argument("--tag", default="snn")
    b.add_argument("--threads", type=int, default=1)

    r = sub.add_parser("run-all", help="prepare -> VAE -> GAN -> SNN -> calibrate -> evaluate")
    r.add_argument("--synthetic", action="store_true", help="generate synthetic data first (smoke test)")

    sub.add_parser("show-config", help="print the resolved configuration")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "synth-data":
        _synth(cfg)
    elif args.command == "prepare":
        from nids.data.preprocess import prepare

        prepare(cfg)
    elif args.command == "train-vae":
        from nids.training.train_vae import train_vae

        train_vae(cfg)
    elif args.command == "train-gan":
        from nids.training.train_gan import train_gan

        train_gan(cfg)
    elif args.command == "train-snn":
        from nids.training.train_snn import train_snn

        train_snn(cfg, augment=False if args.no_augment else None, tag=args.tag)
    elif args.command == "calibrate":
        from nids.calibrate import calibrate

        calibrate(cfg, args.tag)
    elif args.command == "evaluate":
        from nids.evaluation import evaluate

        evaluate(cfg, args.tag)
    elif args.command == "benchmark":
        from nids.benchmark import benchmark

        benchmark(cfg, args.tag, threads=args.threads)
    elif args.command == "run-all":
        _run_all(cfg, args)
    elif args.command == "show-config":
        print(json.dumps(cfg.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
