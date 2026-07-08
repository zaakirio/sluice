<p align="center"><img src="assets/banner.svg" alt="" width="100%"></p>

# Sluice

A cost-aware LLM gateway.
Sluice exposes one OpenAI-compatible endpoint and, per request, decides which backend should serve it: a free local llama.cpp server, Claude Haiku, or Claude Sonnet.
Every request gets a deterministic, explainable routing decision, per-request cost accounting in a SQLite ledger, retries with backoff, fallback chains, circuit breaking, and OpenTelemetry traces.

It is built for teams putting LLM traffic behind one endpoint who need per-request answers to "what did this cost and why was it routed there".
Headline numbers from the reference run on 2026-07-07 (full detail and caveats in [Measured numbers](#measured-numbers)): 7/7 requests served for $0.000000 total, p50 233.0 ms, p95 697.0 ms, including one real Anthropic 401 absorbed by fallback to local inference in the same request.

## The problem

Teams that only ever call one hosted model API tend to have no answer to two questions:

1. What does a single request cost, and which requests are worth a frontier model?
2. What happens when the provider rate-limits or goes down?

Sluice is a small, readable implementation of the production answer: route by request complexity and policy, account for every token at real per-MTok prices, and degrade gracefully through a fallback chain instead of failing.
The unit-economics angle is concrete here because one backend is a local llama.cpp server with zero marginal cost, so the router can express "use the free model unless the request earns a paid one".

Hosted gateways (Vercel AI Gateway, OpenRouter) already do provider failover and cost/latency routing, so Sluice is not trying to be a SaaS gateway.
It is the self-hosted, auditable version for teams that cannot send every request through a third party (on-prem, VPC, data-residency, a local model in the mix) and want the routing policy and the cost ledger to be code they own and can reason about, not a dashboard.

## Quickstart

```bash
uv sync
uv run sluice serve           # gateway on http://127.0.0.1:8091
uv run sluice report          # cost/latency report from the ledger
uv run pytest                 # 61 tests
MODEL=path/to/model.gguf ./scripts/demo.sh   # end-to-end against a real local llama-server
```

Point any OpenAI-compatible client at `http://127.0.0.1:8091/v1` and it just works.
`sluice.yaml` is the routing policy: it declares the backends and maps `(policy, tier)` to backend chains.
For the local tier, point the `local` backend at any running llama-server (any GGUF); for the Claude tiers, set `ANTHROPIC_API_KEY`.
Either works alone: with no API key, everything routes local; with no local server, everything routes cloud.

Set per-request routing behaviour with headers:

```bash
curl http://127.0.0.1:8091/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Sluice-Policy: balanced' \
  -H 'X-Sluice-Max-Cost-USD: 0.002' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":100}'
```

## How we know it works

Two artifacts back every claim in this README.
First, 61 tests drive the gateway through an in-process fake backend with injected sleep and RNG, so backoff sequences, breaker open/half-open/reopen transitions, fallback ordering, cost-cap drops, deadline 504s, streaming, and both directions of the Anthropic translation are asserted exactly, not approximately.
Second, `./scripts/demo.sh` runs 7 real requests end-to-end through a real llama-server, including one request that takes a real 401 from `api.anthropic.com` and falls back to local; the [Measured numbers](#measured-numbers) section is that run's ledger output, verbatim.
Rerunning the demo regenerates every number below on any machine with a GGUF and a built llama.cpp.

## Architecture

```
                 POST /v1/chat/completions
                            |
                   +-----------------+
                   |  route-decision |   complexity heuristic -> tier
                   |     (span)      |   policy + tier -> backend chain
                   +-----------------+   max-cost cap filters the chain
                            |
              chain: [claude-haiku, local]
                            |
          +-----------------v------------------+
          |         execution engine           |
          |  per-backend circuit breaker       |
          |  retries: exp backoff + jitter     |
          |  fallback to next backend in chain |
          |  per-backend timeout / deadline    |
          +----+----------------+---------+----+
               |                |         |
        +------v-----+   +------v-----+   +------------+
        |   local    |   |  anthropic |   |  openai-   |
        | llama.cpp  |   |  (sonnet,  |   |  generic   |
        | $0/MTok    |   |   haiku)   |   |            |
        +------------+   +------------+   +------------+
               |
        response + headers:            SQLite ledger (every request):
        X-Sluice-Backend               ts, backend, model, tokens,
        X-Sluice-Est-Cost-USD          cost, latency, route reason,
        X-Sluice-Route-Reason          fallback hops, status
```

The Anthropic backend translates OpenAI-format requests to the Anthropic Messages API and translates responses (including streaming SSE events and tool calls) back to OpenAI format, so clients only ever speak one dialect.

## Routing policy

Routing is deterministic: the same request with the same headers always produces the same decision, and the decision is returned in `X-Sluice-Route-Reason`.

Step 1, classify the request into a tier using a complexity heuristic:

- `complex` if the request asks for tool use, or the estimated prompt size is at least 1500 tokens (chars/4).
- `moderate` if the prompt contains code markers (fences, `def `, `SELECT `, tracebacks, ...) or is at least 300 estimated tokens.
- `simple` otherwise.

Step 2, look up the backend chain for `(policy, tier)` in `sluice.yaml`.
The policy comes from `X-Sluice-Policy` (`cheap`, `balanced`, `quality`); the default is `balanced`.
For example, `balanced` maps `simple -> [local, claude-haiku]` and `complex -> [claude-sonnet, claude-haiku, local]`.
Every default chain ends in `local`, so the gateway degrades to free local inference instead of failing.

Step 3, apply constraints from headers:

- `X-Sluice-Max-Cost-USD` drops any backend whose pre-flight cost estimate (estimated prompt tokens plus `max_tokens`, priced at the backend's per-MTok rates) exceeds the cap. Dropped backends are named in the route reason.
- `X-Sluice-Latency-Budget-Ms` sets a request deadline. Per-attempt timeouts are clamped to the remaining budget, retries that would land past the deadline are skipped, and a request that exhausts the budget gets a 504.

Example route reason, verbatim from the reference run:

```
policy=quality tier=simple signals=[prompt~16tok] chain=local
  dropped=[claude-haiku(est=$0.000616>cap=$0.000000),claude-sonnet(est=$0.001232>cap=$0.000000)]
```

## Reliability semantics

- Per-backend timeout from `sluice.yaml`, clamped by the latency budget when one is set.
- Retries on 429, 5xx, and transport timeouts: up to `max_retries` extra attempts with exponential backoff (`base * 2^attempt`, capped) and uniform jitter in [0.5x, 1.5x). 4xx errors other than 429 are not retried.
- Fallback: when a backend exhausts its retries (or fails non-retryably), the engine moves to the next backend in the chain. The number of backends tried or skipped before the winner is recorded as `fallback_hops`.
- Circuit breaker per backend: after N consecutive backend-level failures the breaker opens and the backend is skipped without being called. After `circuit_reset_s` it allows exactly one half-open probe; success closes it, failure reopens it.
- Streaming: retries and fallback apply until the first byte of a successful upstream response. Once a stream has started, a mid-stream failure propagates to the client (the ledger records it as `stream_error`).

All of this is unit-tested against an in-process fake backend that can be told to fail N times with a given status, add latency, or stream.

## Cost ledger and observability

Every request is recorded in SQLite: timestamp, backend, model, prompt/completion tokens, estimated cost in USD, latency, route reason, fallback hops, status, and whether it streamed.
Token counts come from upstream `usage` fields; for streaming, Sluice injects `stream_options.include_usage` upstream so llama.cpp reports real token counts in the final chunk.
`uv run sluice report` prints totals by backend, policy, and day, plus p50/p95 latency and throughput.

Tracing uses OpenTelemetry with two spans per request: `route-decision` (policy, tier, chain) and `backend-call` per backend attempted, with each retry attached as a span event.
The exporter is selected with `SLUICE_TRACE_EXPORTER`: `console` (default), `otlp` (honours `OTEL_EXPORTER_OTLP_ENDPOINT`, default `http://127.0.0.1:4318`), or `none`.
Logs are structured JSON on stdout with request ids; a client-supplied `X-Request-Id` is honoured and echoed back.

On non-streaming responses, `X-Sluice-Est-Cost-USD` is computed from actual upstream token usage.
On streaming responses, headers must be sent before the body, so the header carries the pre-flight estimate while the ledger records the actual usage-based cost after the stream completes.

### Langfuse

Sluice can additionally emit one Langfuse trace per gateway request, mirroring the OTel span structure: a `gateway-request` root, a `route-decision` span (policy, tier, chain, verbatim reason), and one `backend-call` span per backend attempted, with each retry attached as an event.
The root trace records the serving model, prompt/completion tokens, estimated cost in USD, fallback hops, and latency as metadata, so the Langfuse UI answers "what did this request cost and why was it routed there".

It is off by default and strictly optional: the `langfuse` package ships in the `obs` extra, the import is guarded, and when the env vars are unset every instrumentation site is a no-op with zero behavior change.

```bash
uv sync --extra obs
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=http://127.0.0.1:3000   # optional; defaults to Langfuse Cloud
uv run sluice serve
```

This works against Langfuse Cloud or a self-hosted Langfuse; only the three env vars differ.

## Docker

```bash
docker build -t sluice .
docker run --rm -p 8091:8091 -v sluice-data:/data sluice
curl http://127.0.0.1:8091/healthz
```

The image is multi-stage (uv builder, `python:3.13-slim` runtime), runs as a non-root user, ships a container healthcheck against `/healthz`, and writes the ledger to the `/data` volume.
Mount your own config over `/app/sluice.yaml` to change backends or policies; pass `LANGFUSE_*` env vars with `-e` to enable tracing.

## Measured numbers

From a real run of `./scripts/demo.sh` on 2026-07-07 (Apple M4 Pro, 24 GB, LFM2.5-1.2B Q4_K_M served by llama-server with full Metal offload, `ANTHROPIC_API_KEY` unset).
The batch was 7 requests across all three policies, including one streamed request and one tool-use request.

- All 7 requests served by `local`; total cost $0.000000.
- The `balanced` request tried `claude-haiku` first, got a non-retryable 401 (no API key), and fell back to `local` in the same request: `fallback_hops=1`. This is the real fallback path exercised against the real Anthropic endpoint.
- p50 latency 233.0 ms, p95 latency 697.0 ms (end-to-end through the gateway, including prompt processing).
- Aggregate local throughput 130.2 tok/s (completion tokens divided by total request wall time, so it understates pure decode speed; short responses are dominated by prompt processing).
- 240 prompt tokens and 297 completion tokens across the batch, measured by llama.cpp's tokenizer.

Cost for the same batch if it had been routed to a cloud backend, priced at the July 2026 per-MTok rates configured in `sluice.yaml`.
These two rows are estimates: the token counts are from the local model's tokenizer and Anthropic tokenizers differ, so treat them as order-of-magnitude.

| Backend | Input $/MTok | Output $/MTok | Batch cost | Source |
|---|---|---|---|---|
| local (measured) | 0.00 | 0.00 | $0.000000 | reference run |
| claude-haiku-4-5 (estimate) | 1.00 | 5.00 | $0.001725 | Anthropic pricing docs |
| claude-sonnet-5 (estimate) | 2.00 | 10.00 | $0.003450 | Anthropic intro pricing through 2026-08-31; sticker $3/$15 |

The absolute numbers are tiny because the batch is tiny; the point is the mechanism.
The ledger prices every request at the serving backend's real rates, so at production volume the report directly answers "what did routing policy X cost us today".

Report output from the reference run, verbatim:

```
By backend
  key                     reqs    ok  prompt_tok  compl_tok    cost_usd    p50_ms    p95_ms   tok_s
  local                      7     7         240        297    0.000000     233.0     697.0   130.2

Totals
  requests            7
  ok                  7
  with fallback hops  1
  total cost usd      0.000000
  p50 latency ms      233.0
  p95 latency ms      697.0
```

## Technical decisions

- **Raw httpx instead of provider SDKs.** The gateway's job is protocol translation and reliability control, and it needs one uniform HTTP layer that tests can replace with an in-process ASGI transport. Owning the retry/breaker logic is the point of the project; an SDK's built-in retries would fight the engine's.
- **Retryability is decided by the backend, orchestration by the engine.** Backends classify errors (429/5xx/timeouts retryable, other 4xx not) into one exception type; the engine owns attempts, backoff, breakers, and fallback. Retry tests run against stubs with injected sleep and RNG, so backoff sequences are asserted exactly.
- **Circuit breaker counts backend-level failures, not attempts.** "Open after N consecutive failures" means N failed calls (each possibly with retries), which matches how an operator reasons about a flapping backend.
- **stdlib sqlite3 for the ledger.** One table, one writer process, a lock around inserts. An ORM or external store adds nothing here.
- **Pre-flight cost estimates use chars/4.** The cap check needs a deterministic, model-agnostic estimate before any tokenizer runs. Actual accounting always uses upstream usage numbers, so the approximation only affects the cap decision, and the estimate formula is stated in the route reason.
- **Chains end in local by design.** A cost-aware gateway should degrade to the free backend, not return 502, when cloud auth or availability fails. The demo demonstrates this against the real Anthropic API with no key configured.
- **Policy config is data, not code.** `sluice.yaml` maps `policy x tier -> chain`, validated at startup. Adding a backend or policy is a config change; the routing engine never special-cases a backend name.
