from __future__ import annotations

import httpx
import pytest

from fake_backend import FakeAnthropicBackend, FakeOpenAIBackend
from sluice.app import create_app
from sluice.config import BackendConfig, Config, ReliabilityConfig


def backend_cfg(
    name: str,
    type_: str = "openai",
    input_cost: float = 0.0,
    output_cost: float = 0.0,
    timeout_s: float = 5.0,
) -> BackendConfig:
    return BackendConfig(
        name=name,
        type=type_,
        model=f"{name}-model",
        base_url="http://fake/v1" if type_ == "openai" else "http://fake",
        input_cost_per_mtok=input_cost,
        output_cost_per_mtok=output_cost,
        timeout_s=timeout_s,
        api_key_env=None,
    )


def reliability_cfg(**overrides) -> ReliabilityConfig:
    defaults = dict(
        max_retries=2,
        backoff_base_s=0.001,
        backoff_max_s=0.01,
        circuit_failure_threshold=3,
        circuit_reset_s=0.05,
    )
    defaults.update(overrides)
    return ReliabilityConfig(**defaults)


@pytest.fixture
def fakes():
    return {
        "primary": FakeOpenAIBackend(),
        "secondary": FakeOpenAIBackend(),
        "claude": FakeAnthropicBackend(),
    }


@pytest.fixture
def config(tmp_path):
    backends = {
        "primary": backend_cfg("primary", input_cost=1.0, output_cost=5.0),
        "secondary": backend_cfg("secondary", input_cost=0.0, output_cost=0.0),
        "claude": backend_cfg("claude", type_="anthropic", input_cost=2.0, output_cost=10.0),
    }
    policies = {
        "cheap": {
            "simple": ["secondary"],
            "moderate": ["secondary"],
            "complex": ["secondary", "primary"],
        },
        "balanced": {
            "simple": ["primary", "secondary"],
            "moderate": ["primary", "secondary"],
            "complex": ["primary", "secondary"],
        },
        "quality": {
            "simple": ["claude", "primary", "secondary"],
            "moderate": ["claude", "primary", "secondary"],
            "complex": ["claude", "primary", "secondary"],
        },
    }
    return Config(
        backends=backends,
        policies=policies,
        reliability=reliability_cfg(),
        default_policy="balanced",
        ledger_path=str(tmp_path / "ledger.db"),
    )


@pytest.fixture
def sluice_app(config, fakes):
    def transport_factory(cfg: BackendConfig) -> httpx.ASGITransport:
        return httpx.ASGITransport(app=fakes[cfg.name].app)

    return create_app(config, transport_factory=transport_factory)


@pytest.fixture
async def client(sluice_app):
    transport = httpx.ASGITransport(app=sluice_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://sluice") as c:
        yield c
