from __future__ import annotations

import time

import pytest

from conftest import backend_cfg, reliability_cfg
from sluice.backends import BackendCallError
from sluice.engine import AllBackendsFailedError, CircuitBreaker, Engine


class StubBackend:
    def __init__(self, name: str, fail_times: int = 0, retryable: bool = True, timeout_s: float = 5.0):
        self.cfg = backend_cfg(name, timeout_s=timeout_s)
        self.fail_times = fail_times
        self.retryable = retryable
        self.calls = 0

    async def call(self, timeout: float):
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise BackendCallError("induced", retryable=self.retryable, status=500)
        return f"{self.cfg.name}-result"


def make_engine(backends: dict[str, StubBackend], **rel_overrides):
    sleeps: list[float] = []

    async def fake_sleep(delay: float):
        sleeps.append(delay)

    engine = Engine(backends, reliability_cfg(**rel_overrides), sleep=fake_sleep, rng=lambda: 0.5)
    return engine, sleeps


async def test_retries_then_success():
    b = StubBackend("a", fail_times=2)
    engine, sleeps = make_engine({"a": b}, max_retries=2, backoff_base_s=0.1, backoff_max_s=10.0)
    name, result, hops, errors = await engine.execute(["a"], lambda be, t: be.call(t))
    assert result == "a-result"
    assert b.calls == 3
    assert hops == 0
    # Exponential backoff with rng pinned at 0.5 -> jitter factor 1.0.
    assert sleeps == pytest.approx([0.1, 0.2])


async def test_backoff_is_capped():
    b = StubBackend("a", fail_times=3)
    engine, sleeps = make_engine({"a": b}, max_retries=3, backoff_base_s=1.0, backoff_max_s=1.5)
    await engine.execute(["a"], lambda be, t: be.call(t))
    assert sleeps == pytest.approx([1.0, 1.5, 1.5])


async def test_non_retryable_fails_fast():
    b = StubBackend("a", fail_times=5, retryable=False)
    engine, sleeps = make_engine({"a": b})
    with pytest.raises(AllBackendsFailedError):
        await engine.execute(["a"], lambda be, t: be.call(t))
    assert b.calls == 1
    assert sleeps == []


async def test_retries_exhausted_raises():
    b = StubBackend("a", fail_times=10)
    engine, _ = make_engine({"a": b}, max_retries=2)
    with pytest.raises(AllBackendsFailedError) as exc:
        await engine.execute(["a"], lambda be, t: be.call(t))
    assert b.calls == 3
    assert "a:" in exc.value.errors[0]


async def test_fallback_to_next_backend():
    a = StubBackend("a", fail_times=10)
    b = StubBackend("b")
    engine, _ = make_engine({"a": a, "b": b}, max_retries=1)
    name, result, hops, errors = await engine.execute(["a", "b"], lambda be, t: be.call(t))
    assert name == "b"
    assert result == "b-result"
    assert hops == 1
    assert len(errors) == 1


def test_breaker_opens_after_threshold():
    br = CircuitBreaker(threshold=3, reset_s=100.0)
    for _ in range(2):
        br.record_failure()
    assert br.allow()
    br.record_failure()
    assert br.state == CircuitBreaker.OPEN
    assert not br.allow()


def test_breaker_half_open_probe_then_close():
    clock = [0.0]
    br = CircuitBreaker(threshold=1, reset_s=10.0, clock=lambda: clock[0])
    br.record_failure()
    assert not br.allow()
    clock[0] = 11.0
    assert br.allow()  # single half-open probe
    assert not br.allow()  # no second probe while first is in flight
    br.record_success()
    assert br.state == CircuitBreaker.CLOSED
    assert br.allow()


def test_breaker_lost_probe_self_heals():
    clock = [0.0]
    br = CircuitBreaker(threshold=1, reset_s=10.0, clock=lambda: clock[0])
    br.record_failure()
    clock[0] = 11.0
    # Probe dispatched but never resolves (e.g. request cancelled mid-call).
    assert br.allow()
    clock[0] = 15.0
    assert not br.allow()
    clock[0] = 21.0
    assert br.allow()


def test_breaker_half_open_failure_reopens():
    clock = [0.0]
    br = CircuitBreaker(threshold=1, reset_s=10.0, clock=lambda: clock[0])
    br.record_failure()
    clock[0] = 11.0
    assert br.allow()
    br.record_failure()
    assert br.state == CircuitBreaker.OPEN
    clock[0] = 20.0
    assert not br.allow()
    clock[0] = 21.0
    assert br.allow()


async def test_engine_skips_open_breaker():
    a = StubBackend("a", fail_times=100)
    b = StubBackend("b")
    engine, _ = make_engine({"a": a, "b": b}, max_retries=0, circuit_failure_threshold=1)
    await engine.execute(["a", "b"], lambda be, t: be.call(t))
    assert a.calls == 1
    # Second request: breaker for a is open, a is not called again.
    name, _, hops, errors = await engine.execute(["a", "b"], lambda be, t: be.call(t))
    assert name == "b"
    assert a.calls == 1
    assert errors == ["a: circuit open"]


async def test_engine_half_open_recovery():
    a = StubBackend("a", fail_times=1)
    engine, _ = make_engine(
        {"a": a}, max_retries=0, circuit_failure_threshold=1, circuit_reset_s=0.02
    )
    with pytest.raises(AllBackendsFailedError):
        await engine.execute(["a"], lambda be, t: be.call(t))
    with pytest.raises(AllBackendsFailedError):  # still open
        await engine.execute(["a"], lambda be, t: be.call(t))
    assert a.calls == 1
    time.sleep(0.03)
    name, _, _, _ = await engine.execute(["a"], lambda be, t: be.call(t))
    assert name == "a"
    assert engine.breakers["a"].state == CircuitBreaker.CLOSED
