from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

LOG_FIELDS = (
    "request_id",
    "policy",
    "tier",
    "backend",
    "status",
    "latency_ms",
    "fallback_hops",
    "est_cost_usd",
    "stream",
)


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        for field in LOG_FIELDS:
            if hasattr(record, field):
                out[field] = getattr(record, field)
        return json.dumps(out)


def setup_logging() -> None:
    logger = logging.getLogger("sluice")
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def setup_tracing() -> None:
    """Exporter selection: SLUICE_TRACE_EXPORTER=console (default) | otlp | none."""
    mode = os.environ.get("SLUICE_TRACE_EXPORTER", "console")
    if mode == "none":
        return

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    if mode == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
        exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
    else:
        exporter = ConsoleSpanExporter()

    provider = TracerProvider(resource=Resource.create({"service.name": "sluice"}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
