"""HTTP gateway that runs next to the open-weight model on every edge node."""

from __future__ import annotations

import hmac
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .backends import build_backend
from .backends.base import ChatMessage
from .cloud.claude import ClaudeEscalator
from .config import Settings
from .gate.confidence import Calibrator, ConfidenceGate
from .orchestrator import HybridOrchestrator, ServiceUnavailable
from .telemetry import Metrics, RecordSink

log = logging.getLogger("hybrid_edge")


class Message(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class InferRequest(BaseModel):
    messages: Optional[List[Message]] = None
    prompt: Optional[str] = None
    route: str = Field("auto", pattern="^(auto|edge|cloud)$")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=8192)
    use_cache: bool = True
    include_signals: bool = False


class InferResponse(BaseModel):
    request_id: str
    answer: str
    source: str
    escalated: bool
    degraded: bool
    refused: bool
    confidence: Optional[float]
    threshold: float
    reasons: List[str]
    region: str
    edge_id: str
    model: str
    latency_ms: float
    edge_latency_ms: Optional[float] = None
    cloud_latency_ms: Optional[float] = None
    usage: Dict[str, int] = {}
    signals: Optional[Dict[str, Any]] = None


def build_orchestrator(settings: Settings, backend=None, escalator=None, records=None) -> HybridOrchestrator:
    calibrator = Calibrator.load(settings.gate_calibration_path) if settings.gate_calibration_path else None
    gate = ConfidenceGate(
        threshold=settings.gate_threshold,
        weights=settings.gate_weights,
        calibrator=calibrator,
        hedge_penalty=settings.gate_hedge_penalty,
        truncation_penalty=settings.gate_truncation_penalty,
    )
    return HybridOrchestrator(
        settings=settings,
        backend=backend or build_backend(settings),
        gate=gate,
        escalator=escalator or ClaudeEscalator(settings),
        metrics=Metrics(settings),
        records=records or RecordSink(settings),
    )


def create_app(settings: Optional[Settings] = None, orchestrator: Optional[HybridOrchestrator] = None) -> FastAPI:
    settings = settings or Settings.load()
    orch = orchestrator or build_orchestrator(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await orch.backend.aclose()
        await orch.escalator.aclose()

    app = FastAPI(title="Hybrid Edge Intelligence", version="0.1.0", lifespan=lifespan)
    app.state.orchestrator = orch
    app.state.settings = settings

    def auth(authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)) -> None:
        if not settings.api_keys:
            return
        supplied = x_api_key or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None)
        if not supplied or not any(hmac.compare_digest(supplied, k) for k in settings.api_keys):
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.post("/v1/infer", response_model=InferResponse, dependencies=[Depends(auth)])
    async def infer(req: InferRequest) -> InferResponse:
        if req.messages:
            msgs = [ChatMessage(m.role, m.content) for m in req.messages]
        elif req.prompt is not None:
            msgs = [ChatMessage("user", req.prompt)]
        else:
            raise HTTPException(status_code=422, detail="provide `messages` or `prompt`")
        try:
            res = await orch.infer(msgs, route=req.route, min_confidence=req.min_confidence,
                                   max_tokens=req.max_tokens, use_cache=req.use_cache)
        except ServiceUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        out = InferResponse(**{k: v for k, v in res.__dict__.items() if k in InferResponse.model_fields})
        if not req.include_signals:
            out.signals = None
        return out

    @app.get("/healthz")
    async def healthz() -> Dict[str, str]:
        # Liveness: the gateway process is up. Used by ECS container health checks.
        return {"status": "ok", "region": settings.region, "edge_id": settings.edge_id}

    @app.get("/readyz")
    async def readyz() -> Dict[str, Any]:
        # Readiness: the edge model answers. ALB / Global Accelerator health checks hit this,
        # so a region whose GPU model is down is drained and traffic shifts to the next-nearest edge.
        edge_ok = await orch.backend.ready()
        if not edge_ok:
            raise HTTPException(status_code=503, detail="edge model not ready")
        return {"status": "ready", "edge_model": settings.edge_model,
                "cloud_breaker_open": orch.escalator.breaker.open}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        return orch.metrics.prometheus()

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    settings = Settings.load()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
