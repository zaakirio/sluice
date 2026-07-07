from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry import trace

from .backends import AnthropicBackend, OpenAIBackend
from .config import Config
from .engine import AllBackendsFailedError, BudgetExhaustedError, Engine
from .ledger import Ledger
from .lf import build_langfuse
from .obs import setup_logging
from .routing import NoRouteError, cost_usd, route

tracer = trace.get_tracer("sluice")
log = logging.getLogger("sluice")

BACKEND_TYPES = {"openai": OpenAIBackend, "anthropic": AnthropicBackend}


def _error(status: int, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "sluice_error"}},
        status_code=status,
        headers=headers,
    )


def _float_header(request: Request, name: str) -> float | None:
    raw = request.headers.get(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"header {name} must be a number, got {raw!r}") from None


def create_app(
    config: Config,
    transport_factory=None,
    ledger_path: str | None = None,
    langfuse=None,
) -> FastAPI:
    setup_logging()
    lf = langfuse if langfuse is not None else build_langfuse()

    backends = {}
    for name, cfg in config.backends.items():
        transport = transport_factory(cfg) if transport_factory else None
        client = httpx.AsyncClient(transport=transport)
        backends[name] = BACKEND_TYPES[cfg.type](cfg, client)

    engine = Engine(backends, config.reliability)
    ledger = Ledger(ledger_path or config.ledger_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        for backend in backends.values():
            await backend.client.aclose()
        ledger.close()

    app = FastAPI(title="sluice", lifespan=lifespan)
    app.state.engine = engine
    app.state.ledger = ledger
    app.state.config = config

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        id_header = {"X-Request-Id": request_id}
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "request body is not valid JSON", id_header)
        if not isinstance(body, dict):
            return _error(400, "request body must be a JSON object", id_header)
        policy = request.headers.get("x-sluice-policy") or config.default_policy
        try:
            max_cost = _float_header(request, "x-sluice-max-cost-usd")
            budget_ms = _float_header(request, "x-sluice-latency-budget-ms")
        except ValueError as e:
            return _error(400, str(e), id_header)
        if budget_ms is not None and budget_ms <= 0:
            return _error(400, "header x-sluice-latency-budget-ms must be positive", id_header)
        stream = bool(body.get("stream"))

        lf_trace = (
            lf.start_observation(
                name="gateway-request",
                input={"policy": policy, "stream": stream},
                metadata={"request_id": request_id},
            )
            if lf
            else None
        )

        with tracer.start_as_current_span("route-decision") as span:
            lf_route = lf_trace.start_observation(name="route-decision") if lf_trace else None
            try:
                decision = route(
                    config,
                    policy,
                    body.get("messages", []),
                    bool(body.get("tools")),
                    max_cost,
                    body.get("max_tokens"),
                )
            except NoRouteError as e:
                if lf_route:
                    lf_route.update(output={"error": str(e)})
                    lf_route.end()
                if lf_trace:
                    lf_trace.update(output={"status": "no_route"})
                    lf_trace.end()
                return _error(400, str(e), id_header)
            span.set_attribute("sluice.request_id", request_id)
            span.set_attribute("sluice.policy", decision.policy)
            span.set_attribute("sluice.tier", decision.tier)
            span.set_attribute("sluice.chain", ">".join(decision.chain))
            if lf_route:
                lf_route.update(
                    output={
                        "policy": decision.policy,
                        "tier": decision.tier,
                        "chain": decision.chain,
                        "reason": decision.reason,
                    }
                )
                lf_route.end()

        deadline = time.monotonic() + budget_ms / 1000.0 if budget_ms else None
        started = time.perf_counter()
        reason = decision.reason.replace("\n", " ")

        def response_headers(backend_name: str, cost: float) -> dict[str, str]:
            return {
                "X-Sluice-Backend": backend_name,
                "X-Sluice-Est-Cost-USD": f"{cost:.8f}",
                "X-Sluice-Route-Reason": reason,
                "X-Request-Id": request_id,
            }

        def finish(
            backend_name: str | None,
            model: str | None,
            prompt_tokens: int,
            completion_tokens: int,
            cost: float,
            status: str,
            hops: int,
            is_stream: bool,
        ) -> None:
            latency_ms = (time.perf_counter() - started) * 1000.0
            ledger.record(
                request_id=request_id,
                policy=decision.policy,
                tier=decision.tier,
                backend=backend_name,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                est_cost_usd=cost,
                latency_ms=latency_ms,
                route_reason=reason,
                fallback_hops=hops,
                status=status,
                stream=is_stream,
            )
            log.info(
                "request completed",
                extra={
                    "request_id": request_id,
                    "policy": decision.policy,
                    "tier": decision.tier,
                    "backend": backend_name,
                    "status": status,
                    "latency_ms": round(latency_ms, 1),
                    "fallback_hops": hops,
                    "est_cost_usd": round(cost, 8),
                    "stream": is_stream,
                },
            )
            if lf_trace:
                lf_trace.update(
                    output={"status": status, "backend": backend_name},
                    metadata={
                        "request_id": request_id,
                        "model": model,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "est_cost_usd": cost,
                        "fallback_hops": hops,
                        "latency_ms": round(latency_ms, 1),
                        "stream": is_stream,
                    },
                )
                lf_trace.end()

        if not stream:
            try:
                name, result, hops, _ = await engine.execute(
                    decision.chain, lambda b, t: b.chat(body, t), deadline, lf_parent=lf_trace
                )
            except BudgetExhaustedError as e:
                finish(None, None, 0, 0, 0.0, "budget_exhausted", 0, False)
                return _error(504, str(e), {"X-Sluice-Route-Reason": reason, **id_header})
            except AllBackendsFailedError as e:
                finish(None, None, 0, 0, 0.0, "error", len(e.errors), False)
                return _error(
                    502,
                    "all backends failed: " + "; ".join(e.errors),
                    {"X-Sluice-Route-Reason": reason, **id_header},
                )
            cfg = config.backends[name]
            cost = cost_usd(cfg, result.prompt_tokens, result.completion_tokens)
            finish(
                name, cfg.model, result.prompt_tokens, result.completion_tokens,
                cost, "ok", hops, False,
            )
            return JSONResponse(result.body, headers=response_headers(name, cost))

        try:
            name, handle, hops, _ = await engine.execute(
                decision.chain, lambda b, t: b.open_stream(body, t), deadline, lf_parent=lf_trace
            )
        except BudgetExhaustedError as e:
            finish(None, None, 0, 0, 0.0, "budget_exhausted", 0, True)
            return _error(504, str(e), {"X-Sluice-Route-Reason": reason, **id_header})
        except AllBackendsFailedError as e:
            finish(None, None, 0, 0, 0.0, "error", len(e.errors), True)
            return _error(
                502,
                "all backends failed: " + "; ".join(e.errors),
                {"X-Sluice-Route-Reason": reason, **id_header},
            )
        cfg = config.backends[name]

        async def relay():
            status = "ok"
            try:
                async for chunk in handle.chunks:
                    yield chunk
            except Exception:
                status = "stream_error"
                raise
            finally:
                cost = cost_usd(cfg, handle.prompt_tokens, handle.completion_tokens)
                finish(
                    name, cfg.model, handle.prompt_tokens, handle.completion_tokens,
                    cost, status, hops, True,
                )

        # Headers must be sent before the body, so the cost header on a
        # streamed response is the pre-flight estimate; the ledger records
        # the actual usage-based cost once the stream completes.
        return StreamingResponse(
            relay(),
            media_type="text/event-stream",
            headers=response_headers(name, decision.est_costs[name]),
        )

    return app
