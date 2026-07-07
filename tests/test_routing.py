from __future__ import annotations

import pytest

from sluice.routing import NoRouteError, classify, cost_usd, route


def msg(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def test_short_prompt_is_simple():
    tier, signals = classify(msg("What is the capital of France?"), has_tools=False)
    assert tier == "simple"
    assert any(s.startswith("prompt~") for s in signals)


def test_code_prompt_is_moderate():
    tier, signals = classify(msg("Fix this:\n```python\ndef f(x):\n    return x\n```"), False)
    assert tier == "moderate"
    assert "code-detected" in signals


def test_long_prompt_is_complex():
    tier, _ = classify(msg("word " * 2000), False)
    assert tier == "complex"


def test_tools_force_complex():
    tier, signals = classify(msg("hi"), has_tools=True)
    assert tier == "complex"
    assert "tools-requested" in signals


def test_multipart_content_counted():
    messages = [{"role": "user", "content": [{"type": "text", "text": "word " * 2000}]}]
    tier, _ = classify(messages, False)
    assert tier == "complex"


def test_route_picks_policy_chain(config):
    decision = route(config, "quality", msg("hi"), False, None, None)
    assert decision.tier == "simple"
    assert decision.chain == ["claude", "primary", "secondary"]
    assert "policy=quality" in decision.reason
    assert "tier=simple" in decision.reason


def test_route_unknown_policy(config):
    with pytest.raises(NoRouteError, match="unknown policy"):
        route(config, "nope", msg("hi"), False, None, None)


def test_max_cost_drops_expensive_backends(config):
    decision = route(config, "quality", msg("hi"), False, 0.0, None)
    assert decision.chain == ["secondary"]
    assert "dropped=" in decision.reason
    assert "claude" in decision.reason


def test_max_cost_no_backend_fits(config):
    policies = dict(config.policies)
    policies["paid-only"] = {t: ["claude"] for t in ("simple", "moderate", "complex")}
    cfg = type(config)(
        backends=config.backends,
        policies=policies,
        reliability=config.reliability,
        default_policy=config.default_policy,
        ledger_path=config.ledger_path,
    )
    with pytest.raises(NoRouteError, match="fits"):
        route(cfg, "paid-only", msg("hi"), False, 0.0, None)


def test_cost_usd_exact(config):
    cfg = config.backends["claude"]  # $2 in / $10 out per MTok
    assert cost_usd(cfg, 1_000_000, 0) == pytest.approx(2.0)
    assert cost_usd(cfg, 0, 1_000_000) == pytest.approx(10.0)
    assert cost_usd(cfg, 500, 200) == pytest.approx((500 * 2 + 200 * 10) / 1e6)


def test_est_cost_uses_max_tokens(config):
    small = route(config, "quality", msg("hi"), False, None, 10)
    large = route(config, "quality", msg("hi"), False, None, 10_000)
    assert large.est_costs["claude"] > small.est_costs["claude"]
