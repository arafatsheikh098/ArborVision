"""FastAPI server exposing ArborVision models.

Endpoints
---------

GET  /                 — health + version
GET  /health           — same, structured
GET  /models           — list registered models with metadata
POST /predict          — single image, multiple models, returns predictions
POST /benchmark        — run a structured experiment from on-disk image paths
GET  /experiments      — list past experiment directories
GET  /experiments/{id} — load a past experiment's full results
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__ as ARBORVISION_VERSION
from ..config import RESULTS_DIR, get_device
from ..data.loader import decode_upload
from ..experiments.runner import run_experiment
from ..models import get_model, list_models
from .schemas import (
    BenchmarkRequest,
    ExperimentSummary,
    ExperimentsListResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    MultiPredictResponse,
    PredictionResponse,
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="ArborVision",
        description="Computer Vision Benchmarking Platform API",
        version=ARBORVISION_VERSION,
    )

    # CORS: same-origin by default, plus common dev hosts
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://localhost:8501",  # streamlit default
            "http://127.0.0.1:8501",
            "*",  # dev convenience; tighten in production
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----------------------------------------------------------------- routes

    @app.get("/", response_model=HealthResponse, tags=["meta"])
    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            arborvision_version=ARBORVISION_VERSION,
            device=get_device(),
            n_models=len(list_models()),
        )

    @app.get("/models", response_model=ModelsResponse, tags=["models"])
    def get_models() -> ModelsResponse:
        out: list[ModelInfo] = []
        for name in list_models():
            adapter = get_model(name)
            meta = adapter.metadata()
            out.append(
                ModelInfo(
                    name=meta.name,
                    task=meta.task,
                    description=meta.description,
                    device=meta.device,
                    weights=meta.weights,
                    extra=meta.extra,
                )
            )
        return ModelsResponse(models=out)

    @app.post("/predict", response_model=MultiPredictResponse, tags=["inference"])
    async def predict(
        files: Annotated[list[UploadFile], File(description="One or more image files")],
        model_names: Annotated[str, Form(description="Comma-separated model names")],
    ) -> MultiPredictResponse:
        names = [m.strip() for m in model_names.split(",") if m.strip()]
        unknown = [n for n in names if n not in list_models()]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model(s): {unknown}. Available: {list_models()}",
            )
        if not files:
            raise HTTPException(status_code=400, detail="No files supplied.")

        # Lazy-instantiate models once and reuse for the whole batch
        models = {n: get_model(n) for n in names}
        for m in models.values():
            m.warmup()

        predictions: dict[str, list[PredictionResponse]] = {n: [] for n in names}
        total_latency = 0.0

        for f in files:
            raw = await f.read()
            try:
                image = decode_upload(raw)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not decode {f.filename}: {e}",
                )
            for n, m in models.items():
                pred = m.timed_predict(image)
                total_latency += pred.latency_ms
                predictions[n].append(
                    PredictionResponse(
                        model=n,
                        latency_ms=pred.latency_ms,
                        labels=pred.labels,
                        scores=pred.scores,
                        boxes=[list(b) for b in pred.boxes],
                        n_masks=len(pred.masks),
                        topk_labels=pred.topk_labels,
                        topk_scores=pred.topk_scores,
                    )
                )
        return MultiPredictResponse(predictions=predictions, total_latency_ms=total_latency)

    @app.post("/benchmark", tags=["experiment"])
    async def benchmark(req: BenchmarkRequest) -> JSONResponse:
        """Run a structured experiment. ``image_paths`` are read from disk."""
        if not req.image_paths:
            raise HTTPException(
                status_code=400,
                detail="image_paths is required and must be non-empty.",
            )
        unknown = [n for n in req.model_names if n not in list_models()]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model(s): {unknown}. Available: {list_models()}",
            )

        try:
            result = run_experiment(
                image_paths=[Path(p) for p in req.image_paths],
                model_names=req.model_names,
                ground_truth=req.ground_truth,
                output_dir=Path(req.output_dir) if req.output_dir else None,
                score_threshold=req.score_threshold,
                n_warmup=req.n_warmup,
                n_runs=req.n_runs,
                save_overlays=req.save_overlays,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return JSONResponse(content=result.to_dict())

    @app.get("/experiments", response_model=ExperimentsListResponse, tags=["experiment"])
    def list_experiments() -> ExperimentsListResponse:
        out: list[ExperimentSummary] = []
        if not RESULTS_DIR.exists():
            return ExperimentsListResponse(experiments=[])
        for child in sorted(RESULTS_DIR.iterdir(), reverse=True):
            if not child.is_dir():
                continue
            cfg_path = child / "config.json"
            res_path = child / "results.json"
            if not cfg_path.exists() or not res_path.exists():
                continue
            try:
                cfg = json.loads(cfg_path.read_text())
            except Exception:
                continue
            out.append(
                ExperimentSummary(
                    experiment_id=cfg.get("experiment_id", child.name),
                    created_at=cfg.get("created_at", ""),
                    output_dir=str(child),
                    models=list(cfg.get("model_names", [])),
                    n_images=len(cfg.get("image_paths", [])),
                )
            )
        return ExperimentsListResponse(experiments=out)

    @app.get("/experiments/{experiment_id}", tags=["experiment"])
    def get_experiment(experiment_id: str) -> JSONResponse:
        # experiment_id is also the directory name for our default scheme
        path = RESULTS_DIR / experiment_id
        results_path = path / "results.json"
        if not results_path.exists():
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id!r} not found.")
        try:
            return JSONResponse(content=json.loads(results_path.read_text()))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not load experiment: {e}")

    return app


# Module-level app for ``uvicorn arborvision.api.server:app``
app = create_app()