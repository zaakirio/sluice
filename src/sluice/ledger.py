from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    request_id TEXT NOT NULL,
    policy TEXT NOT NULL,
    tier TEXT NOT NULL,
    backend TEXT,
    model TEXT,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    est_cost_usd REAL NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL,
    route_reason TEXT NOT NULL,
    fallback_hops INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    stream INTEGER NOT NULL DEFAULT 0
)
"""


class Ledger:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self.conn.execute(SCHEMA)
            self.conn.commit()

    def record(
        self,
        *,
        request_id: str,
        policy: str,
        tier: str,
        backend: str | None,
        model: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        est_cost_usd: float,
        latency_ms: float,
        route_reason: str,
        fallback_hops: int,
        status: str,
        stream: bool,
    ) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO requests (ts, request_id, policy, tier, backend, model,"
                " prompt_tokens, completion_tokens, est_cost_usd, latency_ms,"
                " route_reason, fallback_hops, status, stream)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    request_id,
                    policy,
                    tier,
                    backend,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    est_cost_usd,
                    latency_ms,
                    route_reason,
                    fallback_hops,
                    status,
                    int(stream),
                ),
            )
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()
