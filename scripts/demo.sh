#!/usr/bin/env bash
# End-to-end demo: real llama-server + sluice, mixed batch of policies,
# then the ledger report.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_BIN="${LLAMA_BIN:-llama-server}"
MODEL="${MODEL:?set MODEL to a local GGUF path (any small chat model works)}"
LLAMA_PORT=8092
SLUICE_PORT=8091
DB="$ROOT/demo-ledger.db"
LLAMA_LOG="$(mktemp -t sluice-llama)"
SLUICE_LOG="$(mktemp -t sluice-gw)"

rm -f "$DB"

cleanup() {
    [ -n "${SLUICE_PID:-}" ] && kill "$SLUICE_PID" 2>/dev/null || true
    [ -n "${LLAMA_PID:-}" ] && kill "$LLAMA_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

wait_for() {
    local url=$1 name=$2
    for _ in $(seq 1 120); do
        if curl -sf "$url" >/dev/null 2>&1; then return 0; fi
        sleep 0.5
    done
    echo "FATAL: $name did not become healthy" >&2
    exit 1
}

echo "== starting llama-server (port $LLAMA_PORT) =="
"$LLAMA_BIN" -m "$MODEL" --port "$LLAMA_PORT" --jinja -ngl 99 --ctx-size 4096 \
    >"$LLAMA_LOG" 2>&1 &
LLAMA_PID=$!
wait_for "http://127.0.0.1:$LLAMA_PORT/health" "llama-server"

echo "== starting sluice (port $SLUICE_PORT) =="
cd "$ROOT"
SLUICE_TRACE_EXPORTER="${SLUICE_TRACE_EXPORTER:-none}" \
    uv run sluice serve --config sluice.yaml --port "$SLUICE_PORT" --db "$DB" \
    >"$SLUICE_LOG" 2>&1 &
SLUICE_PID=$!
wait_for "http://127.0.0.1:$SLUICE_PORT/healthz" "sluice"

BASE="http://127.0.0.1:$SLUICE_PORT/v1/chat/completions"
HDRS="$(mktemp -t sluice-hdrs)"
BODY="$(mktemp -t sluice-body)"

show() {
    echo "  backend:  $(grep -i '^x-sluice-backend:' "$HDRS" | tr -d '\r' | cut -d' ' -f2-)"
    echo "  cost:     $(grep -i '^x-sluice-est-cost-usd:' "$HDRS" | tr -d '\r' | cut -d' ' -f2-)"
    echo "  reason:   $(grep -i '^x-sluice-route-reason:' "$HDRS" | tr -d '\r' | cut -d' ' -f2-)"
}

run_request() {
    local title=$1; shift
    echo
    echo "-- $title"
    curl -s -D "$HDRS" -o "$BODY" "$BASE" -H 'Content-Type: application/json' "$@"
    show
    python3 -c "
import json, sys
body = open('$BODY').read()
if body.lstrip().startswith('{'):
    d = json.loads(body)
    if 'choices' in d:
        msg = d['choices'][0]['message']
        content = msg.get('content') or ''
        if content:
            print('  reply:    ' + content.replace(chr(10), ' ')[:110])
        elif msg.get('tool_calls'):
            call = msg['tool_calls'][0]['function']
            print(f\"  reply:    tool_call {call['name']}({call['arguments']})\")
    else:
        print('  body:     ' + body[:110])
else:
    text = ''.join(
        (json.loads(line[6:])['choices'][0].get('delta', {}).get('content') or '')
        for line in body.splitlines()
        if line.startswith('data: ') and line != 'data: [DONE]'
        and json.loads(line[6:]).get('choices')
    )
    print('  reply:    ' + text.replace(chr(10), ' ')[:110] + '  (streamed)')
"
}

run_request "cheap / simple prompt -> local" \
    -H 'X-Sluice-Policy: cheap' \
    -d '{"messages":[{"role":"user","content":"In one sentence, what is a hash map?"}],"max_tokens":80}'

run_request "cheap / code prompt -> moderate tier, still local" \
    -H 'X-Sluice-Policy: cheap' \
    -d '{"messages":[{"role":"user","content":"Explain this briefly:\n```python\ndef f(xs):\n    return {x: len(x) for x in xs}\n```"}],"max_tokens":80}'

run_request "balanced / moderate -> claude-haiku first, falls back to local without API key" \
    -H 'X-Sluice-Policy: balanced' \
    -d '{"messages":[{"role":"user","content":"Review this function and name one bug:\n```python\ndef avg(xs):\n    return sum(xs) / len(xs)\n```"}],"max_tokens":100}'

run_request "quality + max-cost cap \$0 -> paid backends dropped at routing time" \
    -H 'X-Sluice-Policy: quality' -H 'X-Sluice-Max-Cost-USD: 0' \
    -d '{"messages":[{"role":"user","content":"Summarise the tradeoffs of write-ahead logging in two sentences."}],"max_tokens":120}'

run_request "cheap / streaming" \
    -H 'X-Sluice-Policy: cheap' \
    -d '{"messages":[{"role":"user","content":"Count from 1 to 10, comma separated."}],"max_tokens":60,"stream":true}'

run_request "cheap / tools requested -> complex tier" \
    -H 'X-Sluice-Policy: cheap' \
    -d '{"messages":[{"role":"user","content":"What is the weather in Paris?"}],"max_tokens":60,"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}'

run_request "default policy (balanced) / simple, larger completion" \
    -d '{"messages":[{"role":"user","content":"Write a limerick about caching."}],"max_tokens":150}'

echo
echo "== ledger report =="
uv run sluice report --db "$DB"

echo
echo "== done; servers stopping =="
