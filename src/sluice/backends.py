from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from .config import BackendConfig


class BackendCallError(Exception):
    def __init__(self, message: str, retryable: bool, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@dataclass
class ChatResult:
    body: dict
    prompt_tokens: int
    completion_tokens: int


@dataclass
class StreamHandle:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    chunks: AsyncIterator[bytes] = field(default=None, repr=False)  # type: ignore[assignment]


def _retryable_status(status: int) -> bool:
    return status == 429 or status >= 500


def _wrap_transport_error(exc: httpx.HTTPError) -> BackendCallError:
    # Timeouts and connection failures are transient by nature; retry them.
    return BackendCallError(f"{type(exc).__name__}: {exc}", retryable=True)


async def _raise_for_status(resp: httpx.Response, streamed: bool) -> None:
    if resp.status_code == 200:
        return
    if streamed:
        body = (await resp.aread()).decode(errors="replace")
        await resp.aclose()
    else:
        body = resp.text
    raise BackendCallError(
        f"HTTP {resp.status_code}: {body[:200]}",
        retryable=_retryable_status(resp.status_code),
        status=resp.status_code,
    )


class OpenAIBackend:
    def __init__(self, cfg: BackendConfig, client: httpx.AsyncClient):
        self.cfg = cfg
        self.client = client
        self._url = f"{cfg.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        key = self.cfg.api_key
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _prepare(self, payload: dict) -> dict:
        body = dict(payload)
        body["model"] = self.cfg.model
        return body

    async def chat(self, payload: dict, timeout: float) -> ChatResult:
        try:
            resp = await self.client.post(
                self._url, json=self._prepare(payload), headers=self._headers(), timeout=timeout
            )
        except httpx.HTTPError as e:
            raise _wrap_transport_error(e) from e
        await _raise_for_status(resp, streamed=False)
        body = resp.json()
        usage = body.get("usage") or {}
        return ChatResult(
            body=body,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    async def open_stream(self, payload: dict, timeout: float) -> StreamHandle:
        body = self._prepare(payload)
        body["stream"] = True
        # Ask upstream for a final usage chunk so the ledger records real
        # token counts; the chunk is spec-compliant and forwarded to the client.
        # Merge rather than setdefault so a client-supplied stream_options
        # cannot silently disable usage reporting.
        body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
        req = self.client.build_request(
            "POST", self._url, json=body, headers=self._headers(), timeout=timeout
        )
        try:
            resp = await self.client.send(req, stream=True)
        except httpx.HTTPError as e:
            raise _wrap_transport_error(e) from e
        await _raise_for_status(resp, streamed=True)
        handle = StreamHandle()
        handle.chunks = self._relay(resp, handle)
        return handle

    async def _relay(self, resp: httpx.Response, handle: StreamHandle) -> AsyncIterator[bytes]:
        try:
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        obj = json.loads(line[6:])
                    except ValueError:
                        obj = None
                    if obj and obj.get("usage"):
                        handle.prompt_tokens = obj["usage"].get("prompt_tokens", 0)
                        handle.completion_tokens = obj["usage"].get("completion_tokens", 0)
                yield (line + "\n").encode()
        finally:
            await resp.aclose()


ANTHROPIC_FINISH_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


class AnthropicBackend:
    def __init__(self, cfg: BackendConfig, client: httpx.AsyncClient):
        self.cfg = cfg
        self.client = client
        self._url = f"{cfg.base_url}/v1/messages"

    def _headers(self) -> dict[str, str]:
        headers = {"anthropic-version": "2023-06-01"}
        key = self.cfg.api_key
        if key:
            headers["x-api-key"] = key
        return headers

    def _prepare(self, payload: dict) -> dict:
        system_parts: list[str] = []
        messages: list[dict] = []
        for m in payload["messages"]:
            role = m["role"]
            text = _content_text(m.get("content"))
            if role == "system":
                system_parts.append(text)
            elif role in ("user", "assistant"):
                messages.append({"role": role, "content": text})
            else:
                raise BackendCallError(
                    f"role {role!r} not supported by anthropic backend", retryable=False
                )
        body: dict = {
            "model": self.cfg.model,
            "max_tokens": payload.get("max_tokens") or 1024,
            "messages": messages,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if payload.get("temperature") is not None:
            body["temperature"] = payload["temperature"]
        if payload.get("stop"):
            stop = payload["stop"]
            body["stop_sequences"] = stop if isinstance(stop, list) else [stop]
        if payload.get("tools"):
            body["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"].get("parameters", {"type": "object"}),
                }
                for t in payload["tools"]
            ]
        return body

    def _translate_response(self, resp: dict) -> dict:
        text = "".join(b.get("text", "") for b in resp["content"] if b["type"] == "text")
        tool_calls = [
            {
                "id": b["id"],
                "type": "function",
                "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
            }
            for b in resp["content"]
            if b["type"] == "tool_use"
        ]
        message: dict = {"role": "assistant", "content": text or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        usage = resp.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        return {
            "id": resp.get("id", ""),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.cfg.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": ANTHROPIC_FINISH_MAP.get(resp.get("stop_reason"), "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def chat(self, payload: dict, timeout: float) -> ChatResult:
        try:
            resp = await self.client.post(
                self._url, json=self._prepare(payload), headers=self._headers(), timeout=timeout
            )
        except httpx.HTTPError as e:
            raise _wrap_transport_error(e) from e
        await _raise_for_status(resp, streamed=False)
        body = self._translate_response(resp.json())
        usage = body["usage"]
        return ChatResult(
            body=body,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
        )

    async def open_stream(self, payload: dict, timeout: float) -> StreamHandle:
        if payload.get("tools"):
            # Streaming tool-call translation (partial input_json deltas) is
            # deliberately out of scope; see README limitations.
            raise BackendCallError(
                "streaming with tools is not supported on anthropic backends",
                retryable=False,
            )
        body = self._prepare(payload)
        body["stream"] = True
        req = self.client.build_request(
            "POST", self._url, json=body, headers=self._headers(), timeout=timeout
        )
        try:
            resp = await self.client.send(req, stream=True)
        except httpx.HTTPError as e:
            raise _wrap_transport_error(e) from e
        await _raise_for_status(resp, streamed=True)
        handle = StreamHandle()
        handle.chunks = self._relay(resp, handle)
        return handle

    def _chunk(self, msg_id: str, delta: dict, finish_reason: str | None, usage: dict | None = None) -> bytes:
        obj: dict = {
            "id": msg_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.cfg.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            obj["usage"] = usage
        return f"data: {json.dumps(obj)}\n\n".encode()

    async def _relay(self, resp: httpx.Response, handle: StreamHandle) -> AsyncIterator[bytes]:
        msg_id = ""
        stop_reason = None
        try:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except ValueError:
                    continue
                etype = event.get("type")
                if etype == "message_start":
                    msg = event.get("message", {})
                    msg_id = msg.get("id", "")
                    handle.prompt_tokens = msg.get("usage", {}).get("input_tokens", 0)
                    yield self._chunk(msg_id, {"role": "assistant", "content": ""}, None)
                elif etype == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield self._chunk(msg_id, {"content": delta["text"]}, None)
                elif etype == "message_delta":
                    stop_reason = event.get("delta", {}).get("stop_reason")
                    handle.completion_tokens = event.get("usage", {}).get(
                        "output_tokens", handle.completion_tokens
                    )
                elif etype == "message_stop":
                    usage = {
                        "prompt_tokens": handle.prompt_tokens,
                        "completion_tokens": handle.completion_tokens,
                        "total_tokens": handle.prompt_tokens + handle.completion_tokens,
                    }
                    yield self._chunk(
                        msg_id, {}, ANTHROPIC_FINISH_MAP.get(stop_reason, "stop"), usage
                    )
                    yield b"data: [DONE]\n\n"
                elif etype == "error":
                    # Anthropic reports mid-stream failures (e.g. overloaded)
                    # as an error event; surface it instead of ending the
                    # stream silently with an "ok" ledger status.
                    err = event.get("error", {})
                    raise BackendCallError(
                        f"anthropic stream error: {err.get('type')}: {err.get('message')}",
                        retryable=False,
                    )
        finally:
            await resp.aclose()


Backend = OpenAIBackend | AnthropicBackend
