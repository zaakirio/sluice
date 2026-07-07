from __future__ import annotations

import httpx
import pytest

from sluice.app import create_app
from sluice.lf import build_langfuse


class SpySpan:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.children: list[SpySpan] = []
        self.events: list[tuple[str, dict]] = []
        self.updates: list[dict] = []
        self.ended = False

    def start_observation(self, name: str, **kwargs) -> SpySpan:
        child = SpySpan(name, **kwargs)
        self.children.append(child)
        return child

    def update(self, **kwargs) -> SpySpan:
        self.updates.append(kwargs)
        return self

    def create_event(self, name: str, **kwargs) -> None:
        self.events.append((name, kwargs))

    def end(self) -> None:
        self.ended = True


class SpyLangfuse:
    def __init__(self):
        self.traces: list[SpySpan] = []

    def start_observation(self, name: str, **kwargs) -> SpySpan:
        span = SpySpan(name, **kwargs)
        self.traces.append(span)
        return span


@pytest.fixture
def spy():
    return SpyLangfuse()


@pytest.fixture
def instrumented_client(config, fakes, spy):
    def transport_factory(cfg):
        return httpx.ASGITransport(app=fakes[cfg.name].app)

    app = create_app(config, transport_factory=transport_factory, langfuse=spy)
    return httpx.ASGITransport(app=app)


def payload(text: str = "hi") -> dict:
    return {"messages": [{"role": "user", "content": text}], "max_tokens": 50}


async def post(transport, json, headers=None):
    async with httpx.AsyncClient(transport=transport, base_url="http://sluice") as c:
        return await c.post("/v1/chat/completions", json=json, headers=headers or {})


async def test_trace_mirrors_route_and_backend_spans(instrumented_client, spy):
    r = await post(instrumented_client, payload(), {"X-Sluice-Policy": "balanced"})
    assert r.status_code == 200

    assert len(spy.traces) == 1
    trace = spy.traces[0]
    assert trace.name == "gateway-request"
    assert [c.name for c in trace.children] == ["route-decision", "backend-call"]
    assert trace.ended
    assert all(c.ended for c in trace.children)

    route = trace.children[0]
    route_out = route.updates[-1]["output"]
    assert route_out["policy"] == "balanced"
    assert route_out["tier"] == "simple"
    assert route_out["chain"] == ["primary", "secondary"]

    call = trace.children[1]
    assert call.kwargs["metadata"] == {"backend": "primary", "model": "primary-model"}

    meta = trace.updates[-1]["metadata"]
    assert meta["model"] == "primary-model"
    assert meta["prompt_tokens"] == 12
    assert meta["completion_tokens"] == 7
    assert meta["est_cost_usd"] == pytest.approx((12 * 1.0 + 7 * 5.0) / 1e6)
    assert meta["fallback_hops"] == 0
    assert trace.updates[-1]["output"]["status"] == "ok"


async def test_retries_recorded_as_events(instrumented_client, spy, fakes):
    fakes["primary"].fail_times = 2
    fakes["primary"].fail_status = 500

    r = await post(instrumented_client, payload(), {"X-Sluice-Policy": "balanced"})
    assert r.status_code == 200

    call = spy.traces[0].children[1]
    assert call.name == "backend-call"
    retry_events = [e for e in call.events if e[0] == "retry"]
    assert len(retry_events) == 2
    assert retry_events[0][1]["metadata"]["attempt"] == 1


async def test_fallback_produces_span_per_backend(instrumented_client, spy, fakes):
    fakes["primary"].fail_times = 10
    fakes["primary"].fail_status = 500

    r = await post(instrumented_client, payload(), {"X-Sluice-Policy": "balanced"})
    assert r.status_code == 200
    assert r.headers["X-Sluice-Backend"] == "secondary"

    trace = spy.traces[0]
    calls = [c for c in trace.children if c.name == "backend-call"]
    assert [c.kwargs["metadata"]["backend"] for c in calls] == ["primary", "secondary"]
    assert "error" in calls[0].updates[-1]["output"]
    assert trace.updates[-1]["metadata"]["fallback_hops"] == 1


async def test_streaming_trace_ends_after_body(instrumented_client, spy):
    r = await post(
        instrumented_client,
        dict(payload(), stream=True),
        {"X-Sluice-Policy": "balanced"},
    )
    assert r.status_code == 200
    r.read()

    trace = spy.traces[0]
    assert trace.ended
    assert trace.updates[-1]["metadata"]["stream"] is True


async def test_no_langfuse_is_a_noop(config, fakes, monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    def transport_factory(cfg):
        return httpx.ASGITransport(app=fakes[cfg.name].app)

    app = create_app(config, transport_factory=transport_factory)
    r = await post(httpx.ASGITransport(app=app), payload(), {"X-Sluice-Policy": "balanced"})
    assert r.status_code == 200
    assert r.headers["X-Sluice-Backend"] == "primary"


def test_build_langfuse_requires_env(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert build_langfuse() is None


def test_build_langfuse_requires_package(monkeypatch):
    import importlib.util

    if importlib.util.find_spec("langfuse") is not None:
        pytest.skip("langfuse is installed; the ImportError path is not reachable")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    assert build_langfuse() is None
