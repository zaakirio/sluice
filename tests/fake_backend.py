"""In-process ASGI fakes for OpenAI-compatible and Anthropic backends.

Each fake can be told to fail the next N requests, add latency, and stream.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


class FakeOpenAIBackend:
    def __init__(self):
        self.calls = 0
        self.fail_times = 0
        self.fail_status = 500
        self.delay_s = 0.0
        self.completion_text = "Hello from fake backend."
        self.usage = {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
        self.last_body: dict | None = None
        self.app = FastAPI()
        self.app.post("/v1/chat/completions")(self._handler)

    async def _handler(self, request: Request):
        self.calls += 1
        body = await request.json()
        self.last_body = body
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail_times > 0:
            self.fail_times -= 1
            return JSONResponse({"error": "induced failure"}, status_code=self.fail_status)
        if body.get("stream"):
            return StreamingResponse(self._stream(body), media_type="text/event-stream")
        return JSONResponse(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "fake"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": self.completion_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": self.usage,
            }
        )

    async def _stream(self, body: dict):
        model = body.get("model", "fake")
        for word in self.completion_text.split(" "):
            chunk = {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        final = {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        if body.get("stream_options", {}).get("include_usage"):
            usage_chunk = {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [],
                "usage": self.usage,
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"
        yield "data: [DONE]\n\n"


class FakeAnthropicBackend:
    def __init__(self):
        self.calls = 0
        self.fail_times = 0
        self.fail_status = 429
        self.completion_text = "Hello from fake claude."
        self.usage = {"input_tokens": 20, "output_tokens": 9}
        self.tool_use: dict | None = None
        self.stream_error = False
        self.last_body: dict | None = None
        self.last_headers: dict | None = None
        self.app = FastAPI()
        self.app.post("/v1/messages")(self._handler)

    async def _handler(self, request: Request):
        self.calls += 1
        body = await request.json()
        self.last_body = body
        self.last_headers = dict(request.headers)
        if self.fail_times > 0:
            self.fail_times -= 1
            return JSONResponse(
                {"type": "error", "error": {"type": "induced", "message": "induced failure"}},
                status_code=self.fail_status,
            )
        if body.get("stream"):
            return StreamingResponse(self._stream(), media_type="text/event-stream")
        content: list[dict] = [{"type": "text", "text": self.completion_text}]
        stop_reason = "end_turn"
        if self.tool_use:
            content.append(self.tool_use)
            stop_reason = "tool_use"
        return JSONResponse(
            {
                "id": "msg_fake",
                "type": "message",
                "role": "assistant",
                "model": body.get("model", "fake-claude"),
                "content": content,
                "stop_reason": stop_reason,
                "usage": self.usage,
            }
        )

    async def _stream(self):
        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        yield sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_fake",
                    "usage": {"input_tokens": self.usage["input_tokens"], "output_tokens": 0},
                },
            },
        )
        yield sse(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        )
        for word in self.completion_text.split(" "):
            yield sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": word + " "},
                },
            )
        if self.stream_error:
            yield sse(
                "error",
                {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "Overloaded"},
                },
            )
            return
        yield sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": self.usage["output_tokens"]},
            },
        )
        yield sse("message_stop", {"type": "message_stop"})
