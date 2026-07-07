from __future__ import annotations

import sqlite3

import pytest


def payload(text: str = "hi", **extra) -> dict:
    return {"model": "anything", "messages": [{"role": "user", "content": text}], **extra}


def ledger_rows(config) -> list[sqlite3.Row]:
    conn = sqlite3.connect(config.ledger_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM requests ORDER BY id").fetchall()
    conn.close()
    return rows


async def test_basic_request_headers_and_ledger(client, config, fakes):
    r = await client.post(
        "/v1/chat/completions", json=payload(), headers={"X-Sluice-Policy": "balanced"}
    )
    assert r.status_code == 200
    assert r.headers["X-Sluice-Backend"] == "primary"
    assert "policy=balanced" in r.headers["X-Sluice-Route-Reason"]
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Hello from fake backend."

    rows = ledger_rows(config)
    assert len(rows) == 1
    row = rows[0]
    assert row["backend"] == "primary"
    assert row["policy"] == "balanced"
    assert row["tier"] == "simple"
    assert row["prompt_tokens"] == 12
    assert row["completion_tokens"] == 7
    assert row["status"] == "ok"
    assert row["fallback_hops"] == 0
    assert row["latency_ms"] > 0


async def test_cost_math_exact(client, config, fakes):
    # primary: $1/MTok in, $5/MTok out; fake usage: 12 prompt + 7 completion.
    r = await client.post(
        "/v1/chat/completions", json=payload(), headers={"X-Sluice-Policy": "balanced"}
    )
    expected = (12 * 1.0 + 7 * 5.0) / 1e6
    assert float(r.headers["X-Sluice-Est-Cost-USD"]) == pytest.approx(expected)
    assert ledger_rows(config)[0]["est_cost_usd"] == pytest.approx(expected)


async def test_free_backend_costs_zero(client, config):
    r = await client.post(
        "/v1/chat/completions", json=payload(), headers={"X-Sluice-Policy": "cheap"}
    )
    assert r.headers["X-Sluice-Backend"] == "secondary"
    assert float(r.headers["X-Sluice-Est-Cost-USD"]) == 0.0


async def test_fallback_on_backend_failure(client, config, fakes):
    fakes["primary"].fail_times = 100
    r = await client.post(
        "/v1/chat/completions", json=payload(), headers={"X-Sluice-Policy": "balanced"}
    )
    assert r.status_code == 200
    assert r.headers["X-Sluice-Backend"] == "secondary"
    # max_retries=2 -> 3 attempts against primary before falling back.
    assert fakes["primary"].calls == 3
    assert fakes["secondary"].calls == 1
    assert ledger_rows(config)[0]["fallback_hops"] == 1


async def test_no_fallback_retries_on_non_retryable_status(client, fakes):
    fakes["primary"].fail_times = 100
    fakes["primary"].fail_status = 401
    r = await client.post(
        "/v1/chat/completions", json=payload(), headers={"X-Sluice-Policy": "balanced"}
    )
    assert r.status_code == 200
    assert r.headers["X-Sluice-Backend"] == "secondary"
    assert fakes["primary"].calls == 1


async def test_all_backends_fail_returns_502(client, config, fakes):
    fakes["primary"].fail_times = 100
    fakes["secondary"].fail_times = 100
    r = await client.post(
        "/v1/chat/completions", json=payload(), headers={"X-Sluice-Policy": "balanced"}
    )
    assert r.status_code == 502
    assert "all backends failed" in r.json()["error"]["message"]
    assert r.headers["X-Request-Id"]
    row = ledger_rows(config)[0]
    assert row["status"] == "error"
    assert row["backend"] is None


async def test_max_cost_header_reroutes(client, fakes):
    r = await client.post(
        "/v1/chat/completions",
        json=payload(),
        headers={"X-Sluice-Policy": "balanced", "X-Sluice-Max-Cost-USD": "0"},
    )
    assert r.status_code == 200
    # primary is paid, secondary is free: cap of $0 drops primary.
    assert r.headers["X-Sluice-Backend"] == "secondary"
    assert "dropped=" in r.headers["X-Sluice-Route-Reason"]
    assert fakes["primary"].calls == 0


async def test_max_cost_unroutable_returns_400(client):
    r = await client.post(
        "/v1/chat/completions",
        json=payload(),
        headers={"X-Sluice-Policy": "quality", "X-Sluice-Max-Cost-USD": "-1"},
    )
    assert r.status_code == 400


async def test_unknown_policy_returns_400(client):
    r = await client.post(
        "/v1/chat/completions", json=payload(), headers={"X-Sluice-Policy": "nope"}
    )
    assert r.status_code == 400
    assert "unknown policy" in r.json()["error"]["message"]


async def test_invalid_cost_header_returns_400(client):
    r = await client.post(
        "/v1/chat/completions",
        json=payload(),
        headers={"X-Sluice-Max-Cost-USD": "cheap-please"},
    )
    assert r.status_code == 400


async def test_malformed_json_body_returns_400(client):
    r = await client.post(
        "/v1/chat/completions",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400
    assert "JSON" in r.json()["error"]["message"]


async def test_non_object_json_body_returns_400(client):
    r = await client.post("/v1/chat/completions", json=["hi"])
    assert r.status_code == 400


async def test_non_positive_latency_budget_returns_400(client):
    r = await client.post(
        "/v1/chat/completions",
        json=payload(),
        headers={"X-Sluice-Latency-Budget-Ms": "0"},
    )
    assert r.status_code == 400


async def test_default_policy_applies(client):
    r = await client.post("/v1/chat/completions", json=payload())
    assert "policy=balanced" in r.headers["X-Sluice-Route-Reason"]


async def test_request_id_propagates(client, config):
    r = await client.post(
        "/v1/chat/completions", json=payload(), headers={"X-Request-Id": "req-abc"}
    )
    assert r.headers["X-Request-Id"] == "req-abc"
    assert ledger_rows(config)[0]["request_id"] == "req-abc"


async def test_circuit_breaker_via_app(client, config, fakes):
    fakes["primary"].fail_times = 100
    # threshold=3 backend-level failures opens the breaker.
    for _ in range(3):
        await client.post(
            "/v1/chat/completions", json=payload(), headers={"X-Sluice-Policy": "balanced"}
        )
    calls_when_open = fakes["primary"].calls
    r = await client.post(
        "/v1/chat/completions", json=payload(), headers={"X-Sluice-Policy": "balanced"}
    )
    assert r.status_code == 200
    assert r.headers["X-Sluice-Backend"] == "secondary"
    assert fakes["primary"].calls == calls_when_open


async def test_latency_budget_exhausted_returns_504(client, config, fakes):
    fakes["primary"].delay_s = 0.08
    fakes["primary"].fail_times = 100
    r = await client.post(
        "/v1/chat/completions",
        json=payload(),
        headers={"X-Sluice-Policy": "balanced", "X-Sluice-Latency-Budget-Ms": "50"},
    )
    assert r.status_code == 504
    assert "budget" in r.json()["error"]["message"]
    assert ledger_rows(config)[0]["status"] == "budget_exhausted"
