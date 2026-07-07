from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * q)
    return ordered[idx]


def _print_group(title: str, groups: dict[str, list[sqlite3.Row]]) -> None:
    print(f"\n{title}")
    header = (
        f"  {'key':<22}{'reqs':>6}{'ok':>6}{'prompt_tok':>12}{'compl_tok':>11}"
        f"{'cost_usd':>12}{'p50_ms':>10}{'p95_ms':>10}{'tok_s':>8}"
    )
    print(header)
    for key in sorted(groups):
        rows = groups[key]
        ok_rows = [r for r in rows if r["status"] == "ok"]
        latencies = [r["latency_ms"] for r in ok_rows]
        prompt = sum(r["prompt_tokens"] for r in ok_rows)
        compl = sum(r["completion_tokens"] for r in ok_rows)
        cost = sum(r["est_cost_usd"] for r in ok_rows)
        gen_ms = sum(r["latency_ms"] for r in ok_rows if r["completion_tokens"] > 0)
        tok_s = compl / (gen_ms / 1000.0) if gen_ms > 0 else 0.0
        print(
            f"  {key:<22}{len(rows):>6}{len(ok_rows):>6}{prompt:>12}{compl:>11}"
            f"{cost:>12.6f}{percentile(latencies, 0.5):>10.1f}"
            f"{percentile(latencies, 0.95):>10.1f}{tok_s:>8.1f}"
        )


def cmd_report(args: argparse.Namespace) -> int:
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT ts, policy, tier, backend, prompt_tokens, completion_tokens,"
        " est_cost_usd, latency_ms, fallback_hops, status FROM requests ORDER BY ts"
    ).fetchall()
    conn.close()
    if not rows:
        print("ledger is empty")
        return 0

    by_backend: dict[str, list[sqlite3.Row]] = defaultdict(list)
    by_policy: dict[str, list[sqlite3.Row]] = defaultdict(list)
    by_day: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        by_backend[r["backend"] or "(none)"].append(r)
        by_policy[r["policy"]].append(r)
        by_day[r["ts"][:10]].append(r)

    _print_group("By backend", by_backend)
    _print_group("By policy", by_policy)
    _print_group("By day", by_day)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    latencies = [r["latency_ms"] for r in ok_rows]
    fallbacks = sum(1 for r in rows if r["fallback_hops"] > 0)
    print("\nTotals")
    print(f"  requests            {len(rows)}")
    print(f"  ok                  {len(ok_rows)}")
    print(f"  with fallback hops  {fallbacks}")
    print(f"  total cost usd      {sum(r['est_cost_usd'] for r in ok_rows):.6f}")
    print(f"  p50 latency ms      {percentile(latencies, 0.5):.1f}")
    print(f"  p95 latency ms      {percentile(latencies, 0.95):.1f}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .app import create_app
    from .config import ConfigError, load_config
    from .obs import setup_tracing

    try:
        config = load_config(args.config)
    except (ConfigError, FileNotFoundError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    setup_tracing()
    app = create_app(config, ledger_path=args.db)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sluice", description="Cost-aware LLM gateway")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="run the gateway")
    p_serve.add_argument("--config", default="sluice.yaml")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8091)
    p_serve.add_argument("--db", default=None, help="override ledger path from config")
    p_serve.set_defaults(func=cmd_serve)

    p_report = sub.add_parser("report", help="print cost/latency report from the ledger")
    p_report.add_argument("--db", default="sluice.db")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
