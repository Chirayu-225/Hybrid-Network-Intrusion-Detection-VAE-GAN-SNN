"""FastAPI inference service.

    uvicorn nids.serving.api:app --host 0.0.0.0 --port 8000

Environment:
    NIDS_CONFIG   path to the YAML config (default configs/default.yaml)
    NIDS_SNN_TAG  artifact sub-folder of the SNN to serve (default 'snn')
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from nids import __version__
from nids.config import load_config
from nids.data.preprocess import MissingFeaturesError
from nids.inference import HybridNIDS
from nids.serving.schemas import FlowBatch, HealthResponse, PredictResponse
from nids.utils import get_logger

log = get_logger("nids.api")


def create_app(pipeline: HybridNIDS | None = None) -> FastAPI:
    """App factory. Pass a pre-built pipeline in tests; otherwise artifacts are loaded at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline
        app.state.load_error = None
        if pipeline is None:
            try:
                cfg = load_config(os.environ.get("NIDS_CONFIG", "configs/default.yaml"))
                app.state.pipeline = HybridNIDS.load(cfg, os.environ.get("NIDS_SNN_TAG", "snn"))
                log.info("Pipeline loaded: %d classes", len(app.state.pipeline.classes))
            except Exception as exc:  # keep the process up so /health can explain what is missing
                app.state.load_error = str(exc)
                log.error("Could not load pipeline: %s", exc)
        yield

    app = FastAPI(
        title="Hybrid NIDS API",
        version=__version__,
        description="Risk scoring for network flows: VAE anomaly score + SNN classification -> allow / monitor / block.",
        lifespan=lifespan,
    )

    def _pipeline(request: Request) -> HybridNIDS:
        pipe = request.app.state.pipeline
        if pipe is None:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {request.app.state.load_error}")
        return pipe

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        loaded = request.app.state.pipeline is not None
        return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded,
                              detail=request.app.state.load_error)

    @app.get("/model-info")
    def model_info(request: Request) -> dict[str, Any]:
        return {"version": __version__, **_pipeline(request).info()}

    @app.post("/predict", response_model=PredictResponse)
    def predict(batch: FlowBatch, request: Request) -> PredictResponse:
        pipe = _pipeline(request)
        start = time.perf_counter()
        try:
            out = pipe.predict_frame(pd.DataFrame(batch.flows))
        except MissingFeaturesError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        records = pipe.to_records(out)
        return PredictResponse(results=records, latency_ms=round((time.perf_counter() - start) * 1000, 3),
                               n_flows=len(records))

    return app


app = create_app()
