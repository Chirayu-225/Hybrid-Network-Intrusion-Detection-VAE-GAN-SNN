"""Optional MLflow experiment tracking. Becomes a silent no-op if disabled or not installed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nids.config import Config
from nids.utils import get_logger

log = get_logger("nids.tracking")


class Tracker:
    def __init__(self, cfg: Config, run_name: str, params: dict[str, Any] | None = None):
        self._mlflow = None
        tcfg = cfg.get("tracking", {})
        if not tcfg.get("enabled", False):
            return
        try:
            import mlflow  # noqa: WPS433

            mlflow.set_tracking_uri(tcfg.get("tracking_uri", "sqlite:///mlflow.db"))
            mlflow.set_experiment(tcfg.get("experiment", "hybrid-nids"))
            mlflow.start_run(run_name=run_name)
            self._mlflow = mlflow
            if params:
                self.log_params(params)
        except ImportError:
            log.warning("tracking.enabled=true but mlflow is not installed; continuing without tracking")

    @staticmethod
    def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for key, value in d.items():
            name = f"{prefix}{key}"
            if isinstance(value, dict):
                flat.update(Tracker._flatten(value, name + "."))
            else:
                flat[name] = value
        return flat

    def log_params(self, params: dict[str, Any]) -> None:
        if self._mlflow:
            flat = {k: str(v)[:250] for k, v in self._flatten(params).items()}
            self._mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._mlflow:
            clean = {k: float(v) for k, v in metrics.items() if v is not None}
            self._mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str | Path) -> None:
        if self._mlflow and Path(path).exists():
            self._mlflow.log_artifact(str(path))

    def close(self) -> None:
        if self._mlflow:
            self._mlflow.end_run()
            self._mlflow = None

    def __enter__(self) -> Tracker:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
