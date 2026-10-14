"""YAML configuration with attribute access and single-level inheritance (`_base_`)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/default.yaml")


class Config(dict):
    """A dict whose nested dicts are reachable as attributes: ``cfg.vae.latent_dim``."""

    def __init__(self, data: dict[str, Any] | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(key) from exc

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, Config):
                out[key] = value.to_dict()
            else:
                out[key] = copy.deepcopy(value)
        return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    base_name = data.pop("_base_", None)
    if base_name:
        base_path = (path.parent / base_name).resolve()
        data = _deep_merge(_read_yaml(base_path), data)
    return data


def load_config(path: str | Path | None = None) -> Config:
    """Load a config file (default: ``configs/default.yaml``), resolving ``_base_`` inheritance."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    return Config(_read_yaml(cfg_path))


def artifact_dir(cfg: Config, *parts: str) -> Path:
    """Return (and create) a sub-directory of the artifacts folder."""
    path = Path(cfg.paths.artifacts).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_dir(cfg: Config) -> Path:
    path = Path(cfg.paths.reports)
    path.mkdir(parents=True, exist_ok=True)
    return path
