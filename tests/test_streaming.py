from __future__ import annotations

import json
import sqlite3

import pytest


def payload(text: str = "hi") -> dict:
    return {
        "model": "anything",
        "messages": [{"role": "user", "content": text}],
        "stream": True,
    }


def ledger_rows(config) -> list[sqlite3.Row]:
    conn = sqlite3.connect(config.ledger_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM requests ORDER BY id").fetchall()
    conn.close()
    return rows


def parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[6:]))
    return events


async def test_stream_passthrough(client, config, fakes):
    async with client.stream(
        "POST", "/v1/chat/completions", json=payload(), headers={"X-Sluice-Policy": "balanced"}
    ) as r:
        assert r.status_code == 200
        assert r.headers["X-Sluice-Backend"] == "primary"
        body = (await r.aread()).decode()

    assert "data: [DONE]" in body
    chunks = parse_sse(body)
    text = "".join(
        c["choices"][0]["delta"].get("content", "") for c in chunks if c.get("choices")
    )
    assert text.strip() == "Hello from fake backend."

    row = ledger_rows(config)[0]
    assert row["stream"] == 1
    assert row["prompt_tokens"] == 12
    assert row["completion_tokens"] == 7
    assert row["status"] == "ok"


async def test_stream_header_cost_is_preflight_estimate(client, config):
    async with client.stream(
        "POST", "/v1/chat/completions", json=payload(), headers={"X-Sluice-Policy": "balanced"}
    ) as r:
        header_cost = float(r.headers["X-Sluice-Est-Cost-USD"])
        await r.aread()
    # Pre-flight estimate assumes DEFAULT_COMPLETION_ESTIMATE output tokens.
    assert header_cost > 0
    # Ledger has the actual usage-based cost, which differs from the estimate.
    row = ledger_rows(config)[0]
    assert row["est_cost_usd"] != header_cost
    assert row["est_cost_usd"] == (12 * 1.0 + 7 * 5.0) / 1e6


async def test_stream_fallback_before_first_byte(client, config, fakes):
    fakes["primary"].fail_times = 100
    async with client.stream(
        "POST", "/v1/chat/completions", json=payload(), headers={"X-Sluice-Policy": "balanced"}
    ) as r:
        assert r.status_code == 200
        assert r.headers["X-Sluice-Backend"] == "secondary"
        body = (await r.aread()).decode()
    assert "data: [DONE]" in body
    assert ledger_rows(config)[0]["fallback_hops"] == 1


async def test_stream_usage_injection_forwarded(client, fakes):
    async with client.stream(
        "POST", "/v1/chat/completions", json=payload(), headers={"X-Sluice-Policy": "cheap"}
    ) as r:
        body = (await r.aread()).decode()
    # Gateway injects stream_options.include_usage upstream; the usage chunk
    # is forwarded to the client.
    assert fakes["secondary"].last_body["stream_options"] == {"include_usage": True}
    usage_chunks = [c for c in parse_sse(body) if c.get("usage")]
    assert usage_chunks and usage_chunks[-1]["usage"]["completion_tokens"] == 7


async def test_client_stream_options_cannot_disable_usage(client, fakes):
    body = payload()
    body["stream_options"] = {"include_usage": False}
    async with client.stream(
        "POST", "/v1/chat/completions", json=body, headers={"X-Sluice-Policy": "cheap"}
    ) as r:
        await r.aread()
    assert fakes["secondary"].last_body["stream_options"] == {"include_usage": True}


async def test_anthropic_mid_stream_error_recorded(client, config, fakes):
    fakes["claude"].stream_error = True
    with pytest.raises(Exception):
        async with client.stream(
            "POST", "/v1/chat/completions", json=payload(), headers={"X-Sluice-Policy": "quality"}
        ) as r:
            assert r.status_code == 200
            await r.aread()
    assert ledger_rows(config)[0]["status"] == "stream_error"
