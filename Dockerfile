FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.13-slim-bookworm
RUN useradd --create-home --uid 10001 sluice \
    && mkdir /data && chown sluice:sluice /data
WORKDIR /app
COPY --from=build /app/.venv ./.venv
COPY sluice.yaml ./sluice.yaml
ENV PATH="/app/.venv/bin:$PATH"
USER sluice
VOLUME /data
EXPOSE 8091
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8091/healthz', timeout=2).status == 200 else 1)"]
CMD ["sluice", "serve", "--host", "0.0.0.0", "--port", "8091", "--config", "/app/sluice.yaml", "--db", "/data/sluice.db"]
