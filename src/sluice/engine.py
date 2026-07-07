from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable

from opentelemetry import trace

from .backends import Backend, BackendCallError
from .config import ReliabilityConfig

tracer = trace.get_tracer("sluice")


class AllBackendsFailedError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class BudgetExhaustedError(Exception):
    pass


class CircuitBreaker:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"

    def __init__(self, threshold: int, reset_s: float, clock: Callable[[], float] = time.monotonic):
        self.threshold = threshold
        self.reset_s = reset_s
        self.clock = clock
        self.state = self.CLOSED
        self.failures = 0
        self.opened_at = 0.0
        self.probe_at = 0.0

    def allow(self) -> bool:
        if self.state == self.OPEN:
            if self.clock() - self.opened_at >= self.reset_s:
                self.state = self.HALF_OPEN
                self.probe_at = self.clock()
                return True
            return False
        if self.state == self.HALF_OPEN:
            # A probe is already in flight; refuse further traffic until it
            # resolves. A probe can also vanish without resolving (cancelled
            # request, exhausted latency budget), so after reset_s assume it
            # was lost and allow a new one - otherwise the breaker would stay
            # half-open forever and blackhole the backend.
            if self.clock() - self.probe_at >= self.reset_s:
                self.probe_at = self.clock()
                return True
            return False
        return True

    def record_success(self) -> None:
        self.state = self.CLOSED
        self.failures = 0

    def record_failure(self) -> None:
        self.failures += 1
        if self.state == self.HALF_OPEN or self.failures >= self.threshold:
            self.state = self.OPEN
            self.opened_at = self.clock()


class Engine:
    def __init__(
        self,
        backends: dict[str, Backend],
        rel: ReliabilityConfig,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
    ):
        self.backends = backends
        self.rel = rel
        self.sleep = sleep
        self.rng = rng
        self.breakers = {
            name: CircuitBreaker(rel.circuit_failure_threshold, rel.circuit_reset_s)
            for name in backends
        }

    def _timeout_for(self, backend: Backend, deadline: float | None) -> float:
        timeout = backend.cfg.timeout_s
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BudgetExhaustedError("latency budget exhausted")
            timeout = min(timeout, remaining)
        return timeout

    async def _call_with_retries(
        self, backend: Backend, call, deadline: float | None, span, lf_span=None
    ):
        attempt = 0
        while True:
            timeout = self._timeout_for(backend, deadline)
            try:
                return await call(backend, timeout)
            except BackendCallError as err:
                if not err.retryable or attempt >= self.rel.max_retries:
                    raise
                delay = min(self.rel.backoff_base_s * (2**attempt), self.rel.backoff_max_s)
                delay *= 0.5 + self.rng()  # jitter: uniform in [0.5x, 1.5x)
                if deadline is not None and time.monotonic() + delay >= deadline:
                    raise
                span.add_event(
                    "retry",
                    {"attempt": attempt + 1, "delay_s": round(delay, 4), "error": str(err)},
                )
                if lf_span:
                    lf_span.create_event(
                        name="retry",
                        metadata={
                            "attempt": attempt + 1,
                            "delay_s": round(delay, 4),
                            "error": str(err),
                        },
                    )
                await self.sleep(delay)
                attempt += 1

    async def execute(self, chain: list[str], call, deadline: float | None = None, lf_parent=None):
        """Try each backend in order; returns (name, result, fallback_hops, errors)."""
        errors: list[str] = []
        for name in chain:
            backend = self.backends[name]
            breaker = self.breakers[name]
            if not breaker.allow():
                errors.append(f"{name}: circuit open")
                if lf_parent:
                    lf_parent.create_event(
                        name="circuit-open", metadata={"backend": name}
                    )
                continue
            with tracer.start_as_current_span(
                "backend-call",
                attributes={"sluice.backend": name, "sluice.model": backend.cfg.model},
            ) as span:
                lf_span = (
                    lf_parent.start_observation(
                        name="backend-call",
                        metadata={"backend": name, "model": backend.cfg.model},
                    )
                    if lf_parent
                    else None
                )
                try:
                    result = await self._call_with_retries(backend, call, deadline, span, lf_span)
                except BudgetExhaustedError:
                    if lf_span:
                        lf_span.update(output={"error": "latency budget exhausted"})
                        lf_span.end()
                    raise
                except BackendCallError as err:
                    breaker.record_failure()
                    span.set_attribute("sluice.failed", True)
                    errors.append(f"{name}: {err}")
                    if lf_span:
                        lf_span.update(output={"error": str(err)})
                        lf_span.end()
                    continue
                breaker.record_success()
                if lf_span:
                    lf_span.update(output={"backend": name})
                    lf_span.end()
                return name, result, len(errors), errors
        raise AllBackendsFailedError(errors or ["empty chain"])
