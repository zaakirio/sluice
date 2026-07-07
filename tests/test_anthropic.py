from __future__ import annotations

import json

QUALITY = {"X-Sluice-Policy": "quality"}


def payload(**extra) -> dict:
    return {
        "model": "anything",
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hi"},
        ],
        **extra,
    }


async def test_request_translation(client, fakes):
    r = await client.post(
        "/v1/chat/completions", json=payload(max_tokens=64, temperature=0.2), headers=QUALITY
    )
    assert r.status_code == 200
    sent = fakes["claude"].last_body
    assert sent["model"] == "claude-model"
    assert sent["system"] == "Be terse."
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert sent["max_tokens"] == 64
    assert sent["temperature"] == 0.2
    assert fakes["claude"].last_headers["anthropic-version"] == "2023-06-01"


async def test_response_translation(client, fakes):
    r = await client.post("/v1/chat/completions", json=payload(), headers=QUALITY)
    body = r.json()
    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert choice["message"]["content"] == "Hello from fake claude."
    assert choice["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29}
    # $2/MTok in, $10/MTok out on the test claude backend.
    expected = (20 * 2.0 + 9 * 10.0) / 1e6
    assert abs(float(r.headers["X-Sluice-Est-Cost-USD"]) - expected) < 1e-12


async def test_tools_translation(client, fakes):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    fakes["claude"].tool_use = {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "get_weather",
        "input": {"city": "Paris"},
    }
    r = await client.post("/v1/chat/completions", json=payload(tools=tools), headers=QUALITY)
    sent = fakes["claude"].last_body
    assert sent["tools"] == [
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": tools[0]["function"]["parameters"],
        }
    ]
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}


async def test_stream_translation(client, fakes):
    async with client.stream(
        "POST", "/v1/chat/completions", json=payload(stream=True), headers=QUALITY
    ) as r:
        assert r.status_code == 200
        body = (await r.aread()).decode()

    assert "data: [DONE]" in body
    chunks = [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text.strip() == "Hello from fake claude."
    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["usage"] == {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29}


async def test_stream_with_tools_rejected_falls_back(client, fakes):
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    async with client.stream(
        "POST", "/v1/chat/completions", json=payload(stream=True, tools=tools), headers=QUALITY
    ) as r:
        # Anthropic streaming+tools unsupported -> non-retryable -> falls back
        # to the next backend in the quality chain.
        assert r.status_code == 200
        assert r.headers["X-Sluice-Backend"] == "primary"
        await r.aread()
    assert fakes["claude"].calls == 0


async def test_429_is_retried(client, fakes):
    fakes["claude"].fail_times = 2
    fakes["claude"].fail_status = 429
    r = await client.post("/v1/chat/completions", json=payload(), headers=QUALITY)
    assert r.status_code == 200
    assert r.headers["X-Sluice-Backend"] == "claude"
    assert fakes["claude"].calls == 3
