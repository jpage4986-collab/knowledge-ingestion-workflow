"""HTTP boundary for colleagues and other workflow agents.

The API is intentionally thin: domain decisions stay in the LangGraph
workflow, while this module validates input and serializes stable responses.
It uses local fixtures unless callers inject a provider in application code.
"""

from __future__ import annotations

import json
import time
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from starlette.background import BackgroundTask
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - exercised when the optional extra is absent
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore

from . import run_assistant, run_workflow
from .knowledge import SimulationSpec
from .adapters import Retriever, WebSearcher
from .pdf_lifecycle import PdfLeaseManager
from .query_translation import translate_for_retrieval


_configured_web_searcher: WebSearcher | None = None
_configured_retriever: Retriever | None = None
_configured_pdf_manager = PdfLeaseManager()


def configure_web_searcher(searcher: WebSearcher | None) -> None:
    """Inject an approved web adapter during service startup."""
    global _configured_web_searcher
    _configured_web_searcher = searcher


def configure_retriever(retriever: Retriever | None) -> None:
    """Inject the upstream literature index adapter during service startup."""
    global _configured_retriever
    _configured_retriever = retriever


def configure_pdf_manager(manager: PdfLeaseManager) -> None:
    """Configure allowlisted PDF sources and the temporary lease directory."""
    global _configured_pdf_manager
    _configured_pdf_manager = manager


if FastAPI is not None:
    class AnswerRequest(BaseModel):
        question: str = Field(min_length=1, max_length=20_000)
        allow_web: bool = False
        max_iterations: int = Field(default=2, ge=0, le=5)
        include_full_page: bool = True


    class SimulationRequest(BaseModel):
        request: str = Field(default="validate simulation", max_length=20_000)
        method: str = Field(default="FDTD", min_length=1, max_length=64)
        software: str = Field(default="mock", min_length=1, max_length=64)
        dimension: int = Field(default=3, ge=1, le=3)
        max_frequency_hz: float = Field(gt=0)
        dx_m: float = Field(gt=0)
        dy_m: float = Field(gt=0)
        dz_m: float | None = Field(default=None, gt=0)
        dt_s: float | None = Field(default=None, gt=0)
        relative_permittivity: float = Field(default=1.0, gt=0)
        boundary: str = Field(default="open", min_length=1, max_length=64)
        outputs: list[str] = Field(default_factory=lambda: ["field", "s_parameters"])


    class PdfLeaseRequest(BaseModel):
        source: str = Field(min_length=1, max_length=4_000)
        ttl_seconds: int = Field(default=900, ge=30, le=86_400)
        filename: str | None = Field(default=None, max_length=255)


    def _spec(request: SimulationRequest) -> SimulationSpec:
        values = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        values.pop("request", None)
        return SimulationSpec(**values)


    app = FastAPI(title="FDTD Knowledge Runtime", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        connected = _configured_retriever is not None
        return {"status": "ok", "service": "fdtd-knowledge-runtime", "mode": "connected" if connected else "offline", "execution": "dry_run_only", "retriever_configured": connected}

    @app.post("/answer")
    def answer(request: AnswerRequest) -> dict[str, Any]:
        retrieval_question, translated = translate_for_retrieval(request.question)
        result = dict(run_assistant(retrieval_question, retriever=_configured_retriever, web_searcher=_configured_web_searcher, allow_web=request.allow_web, max_iterations=request.max_iterations, include_full_page=request.include_full_page))
        result["question"] = request.question
        result["retrieval_question"] = retrieval_question
        result["translation_applied"] = translated
        return result

    @app.post("/pdf/lease")
    def lease_pdf(request: PdfLeaseRequest) -> dict[str, Any]:
        """Stage an approved PDF for a downstream drawing/LLM agent."""
        try:
            lease = _configured_pdf_manager.acquire(
                request.source,
                ttl_seconds=request.ttl_seconds,
                filename=request.filename,
            )
        except (FileNotFoundError, PermissionError, ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "lease_id": lease.lease_id,
            "filename": lease.filename,
            "size_bytes": lease.size_bytes,
            "expires_at": lease.expires_at,
            "download_endpoint": f"/pdf/lease/{lease.lease_id}",
            "release_endpoint": f"/pdf/lease/{lease.lease_id}",
            "cleanup_policy": "release after the downstream agent reads it; TTL cleanup is the fallback",
        }

    @app.get("/pdf/lease/{lease_id}")
    def download_pdf(lease_id: str):
        try:
            lease = _configured_pdf_manager.get(lease_id)
            if lease.expires_at <= time.time():
                _configured_pdf_manager.release(lease_id)
                raise FileNotFoundError("PDF lease expired and was released")
            return FileResponse(
                lease.path,
                media_type="application/pdf",
                filename=lease.filename,
                background=BackgroundTask(_configured_pdf_manager.release, lease_id),
            )
        except (FileNotFoundError, ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/pdf/lease/{lease_id}")
    def release_pdf(lease_id: str) -> dict[str, Any]:
        try:
            released = _configured_pdf_manager.release(lease_id)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"released": released, "lease_id": lease_id}

    @app.post("/pdf/cleanup")
    def cleanup_pdf_leases() -> dict[str, Any]:
        return {"removed": _configured_pdf_manager.cleanup_expired()}

    @app.post("/simulation/validate")
    def validate(request: SimulationRequest) -> dict[str, Any]:
        result = run_workflow(request.request, _spec(request))
        return {"simulation_plan": result["simulation_plan"], "validation_issues": result["validation_issues"], "knowledge": result["knowledge"], "trace": result["trace"]}

    @app.post("/visualization/task")
    def visualization_task(request: SimulationRequest) -> dict[str, Any]:
        result = run_workflow(request.request, _spec(request))
        return {"visualization_task": result["visualization_task"], "validation_issues": result["validation_issues"], "simulation_plan": result["simulation_plan"], "trace": result["trace"]}
else:
    app = None


def require_app():
    if app is None:
        raise RuntimeError('Install the API extra first: python -m pip install -e ".[api]"')
    return app
