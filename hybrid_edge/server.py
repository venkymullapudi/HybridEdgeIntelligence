"""HTTP gateway that runs next to the open-weight model on every edge node."""

from __future__ import annotations

import asyncio
import hmac
import json
import re
import uuid
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from .backends import build_backend
from .backends.base import ChatMessage, conversation_problem
from .cloud.claude import ClaudeEscalator
from .config import Settings
from .gate.confidence import Calibrator, ConfidenceGate
from .limits import BodySizeLimit
from .orchestrator import HybridOrchestrator, ServiceUnavailable, caller_id
from .telemetry import Metrics, RecordSink

log = logging.getLogger("hybrid_edge")

REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


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
    cloud_first_token_ms: Optional[float] = None
    usage: Dict[str, Union[int, float]] = {}
    signals: Optional[Dict[str, Any]] = None


def client_ip(request: Request, trust_forwarded_for: bool) -> str:
    """The client's address. Behind the ALB, the *rightmost* X-Forwarded-For entry is
    the one the ALB appended (Global Accelerator preserves the client IP); entries
    to its left came from the client and can be forged."""
    if trust_forwarded_for:
        xff = request.headers.get("x-forwarded-for", "")
        hops = [h.strip() for h in xff.split(",") if h.strip()]
        if hops:
            return hops[-1]
    return request.client.host if request.client else "unknown"


def build_orchestrator(settings: Settings, backend=None, escalator=None, records=None) -> HybridOrchestrator:
    calibrator = Calibrator.load(settings.gate_calibration_path) if settings.gate_calibration_path else None
    if calibrator is not None and calibrator.edge_model != settings.edge_model:
        msg = (f"calibration {settings.gate_calibration_path} was fit on edge model "
               f"{calibrator.edge_model!r}, but the edge model is {settings.edge_model!r}")
        if not settings.gate_allow_calibration_mismatch:
            # Fail fast: ECS's deployment circuit breaker rolls back instead of serving
            # with a threshold that means nothing for this model.
            raise ValueError(msg + "; recalibrate or set HEI_GATE_ALLOW_CALIBRATION_MISMATCH=true")
        log.warning("%s (allowed by gate_allow_calibration_mismatch)", msg)
    gate = ConfidenceGate(
        threshold=settings.gate_threshold,
        weights=settings.gate_weights,
        calibrator=calibrator,
        hedge_penalty=settings.gate_hedge_penalty,
        truncation_penalty=settings.gate_truncation_penalty,
    )
    metrics = Metrics(settings)
    return HybridOrchestrator(
        settings=settings,
        backend=backend or build_backend(settings),
        gate=gate,
        escalator=escalator or ClaudeEscalator(settings),
        metrics=metrics,
        records=records or RecordSink(settings, metrics=metrics),
    )


def create_app(settings: Optional[Settings] = None, orchestrator: Optional[HybridOrchestrator] = None) -> FastAPI:
    settings = settings or Settings.load()
    orch = orchestrator or build_orchestrator(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await orch.drain()
        orch.records.flush()
        await orch.backend.aclose()
        await orch.escalator.aclose()

    app = FastAPI(title="Hybrid Edge Intelligence", version="0.2.0", lifespan=lifespan)
    app.state.orchestrator = orch
    app.state.settings = settings
    app.add_middleware(BodySizeLimit, max_bytes=settings.effective_max_body_bytes)

    def auth(request: Request, authorization: Optional[str] = Header(None),
             x_api_key: Optional[str] = Header(None)) -> str:
        supplied = x_api_key or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None)
        if not settings.api_keys:
            # An unchecked key is just a client-chosen label: keying budgets on it would
            # let a client mint a fresh bucket per request. Identify by address instead.
            return caller_id("ip:" + client_ip(request, settings.trust_forwarded_for))
        if not supplied or not any(hmac.compare_digest(supplied, k) for k in settings.api_keys):
            raise HTTPException(status_code=401, detail="invalid API key")
        return caller_id(supplied)

    def request_id(x_request_id: Optional[str]) -> str:
        # Keep a client's correlation ID if it's safe to echo and log; otherwise mint one.
        return x_request_id if x_request_id and REQUEST_ID_RE.match(x_request_id) else uuid.uuid4().hex

    def parse(req: InferRequest, hdrs: Dict[str, str]) -> List[ChatMessage]:
        if req.messages:
            msgs = [ChatMessage(m.role, m.content) for m in req.messages]
        elif req.prompt is not None:
            msgs = [ChatMessage("user", req.prompt)]
        else:
            raise HTTPException(status_code=422, detail="provide `messages` or `prompt`", headers=hdrs)
        problem = conversation_problem(msgs)
        if problem:
            raise HTTPException(status_code=422, detail=problem, headers=hdrs)
        if len(msgs) > settings.max_messages:
            raise HTTPException(status_code=413, detail=f"too many messages (max {settings.max_messages})", headers=hdrs)
        if sum(len(m.content) for m in msgs) > settings.max_request_chars:
            raise HTTPException(status_code=413, detail=f"request too large (max {settings.max_request_chars} chars)", headers=hdrs)
        return msgs

    def to_response(res: Any, include_signals: bool) -> InferResponse:
        out = InferResponse(**{k: v for k, v in res.__dict__.items() if k in InferResponse.model_fields})
        if not include_signals:
            out.signals = None
        return out

    @app.post("/v1/infer", response_model=InferResponse)
    async def infer(req: InferRequest, response: Response, caller: str = Depends(auth),
                    x_request_id: Optional[str] = Header(None)) -> InferResponse:
        rid = request_id(x_request_id)
        response.headers["X-Request-ID"] = rid
        hdrs = {"X-Request-ID": rid}
        msgs = parse(req, hdrs)
        try:
            res = await orch.infer(msgs, route=req.route, min_confidence=req.min_confidence,
                                   max_tokens=req.max_tokens, use_cache=req.use_cache, caller=caller,
                                   request_id=rid)
        except ServiceUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers=hdrs) from exc
        return to_response(res, req.include_signals)

    @app.post("/v1/infer/stream")
    async def infer_stream(req: InferRequest, caller: str = Depends(auth),
                           x_request_id: Optional[str] = Header(None)) -> StreamingResponse:
        """Server-sent events. Edge-served and cached answers arrive as one `done`
        event. An escalation first sends `escalating`, then `delta` events as Claude
        writes, then `done`. The `done` event's answer is always authoritative. On
        failure the stream ends with an `error` event."""
        rid = request_id(x_request_id)
        hdrs = {"X-Request-ID": rid}
        msgs = parse(req, hdrs)  # validation errors are still plain HTTP errors
        events: "asyncio.Queue[Optional[Dict[str, Any]]]" = asyncio.Queue()

        async def run() -> None:
            try:
                res = await orch.infer(msgs, route=req.route, min_confidence=req.min_confidence,
                                       max_tokens=req.max_tokens, use_cache=req.use_cache, caller=caller,
                                       request_id=rid, on_event=events.put)
                await events.put({"type": "done", **to_response(res, req.include_signals).model_dump()})
            except ServiceUnavailable as exc:
                await events.put({"type": "error", "status": 503, "detail": str(exc), "request_id": rid})
            except Exception:  # noqa: BLE001 - report, don't hang the stream
                log.exception("streamed inference failed")
                await events.put({"type": "error", "status": 500, "detail": "internal error", "request_id": rid})
            finally:
                await events.put(None)

        async def sse() -> AsyncIterator[str]:
            task = asyncio.ensure_future(run())
            try:
                while True:
                    try:
                        ev = await asyncio.wait_for(events.get(), timeout=settings.sse_heartbeat_s)
                    except asyncio.TimeoutError:
                        # SSE comment: keeps proxies and clients from timing out an idle
                        # stream while the edge gates or Claude thinks before its first text.
                        yield ": ping\n\n"
                        continue
                    if ev is None:
                        return
                    yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            finally:
                # Client went away: stop the work (and Claude's billed stream) too.
                if not task.done():
                    task.cancel()

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={**hdrs, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

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
                "cloud_breaker": orch.escalator.breaker.state}

    # Counters reveal traffic, escalation rate and spend: same auth as inference.
    @app.get("/metrics", response_class=PlainTextResponse, dependencies=[Depends(auth)])
    async def metrics() -> str:
        return orch.metrics.prometheus()

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    settings = Settings.load()
    # On SIGTERM (deploy / scale-in) let in-flight escalations finish: they can run up
    # to cloud_deadline_s. ECS stopTimeout and the ALB deregistration delay are sized to match.
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info",
                timeout_graceful_shutdown=int(settings.cloud_deadline_s) + 5)


if __name__ == "__main__":
    main()
