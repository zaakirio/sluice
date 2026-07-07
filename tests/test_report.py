from __future__ import annotations

from sluice.cli import main, percentile
from sluice.ledger import Ledger


def test_percentile_nearest_rank():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 0.5) == 30.0
    assert percentile(values, 0.95) == 50.0
    assert percentile([7.0], 0.95) == 7.0
    assert percentile([], 0.5) == 0.0


def test_report_output(tmp_path, capsys):
    db = str(tmp_path / "ledger.db")
    ledger = Ledger(db)
    for i in range(4):
        ledger.record(
            request_id=f"r{i}",
            policy="cheap" if i % 2 else "quality",
            tier="simple",
            backend="local" if i % 2 else "claude-sonnet",
            model="m",
            prompt_tokens=100,
            completion_tokens=50,
            est_cost_usd=0.0 if i % 2 else 0.0007,
            latency_ms=100.0 + i,
            route_reason="test",
            fallback_hops=1 if i == 3 else 0,
            status="ok",
            stream=False,
        )
    ledger.record(
        request_id="rerr",
        policy="quality",
        tier="simple",
        backend=None,
        model=None,
        prompt_tokens=0,
        completion_tokens=0,
        est_cost_usd=0.0,
        latency_ms=5.0,
        route_reason="test",
        fallback_hops=2,
        status="error",
        stream=False,
    )
    ledger.close()

    assert main(["report", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "By backend" in out
    assert "local" in out
    assert "claude-sonnet" in out
    assert "By policy" in out
    assert "By day" in out
    assert "requests            5" in out
    assert "ok                  4" in out
    assert "with fallback hops  2" in out
    assert "0.001400" in out  # 2 x 0.0007


def test_report_empty_db(tmp_path, capsys):
    db = str(tmp_path / "empty.db")
    Ledger(db).close()
    assert main(["report", "--db", db]) == 0
    assert "ledger is empty" in capsys.readouterr().out
